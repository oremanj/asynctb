import attr
import collections
import dis
import gc
import sys
import traceback
import types
import warnings
from typing import (
    Any,
    Deque,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    cast,
)


@attr.s(auto_attribs=True, slots=True, frozen=True, kw_only=True)
class ContextInfo:
    """Information about a sync or async context manager that's
    currently active in a frame."""

    #: True for an async context manager, False for a sync context manager.
    is_async: bool

    #: True if this context manager is currently exiting. *manager* will be
    #: None in this case; look at the next thing in the traceback.
    is_exiting: bool = False

    #: The context manager object itself (the ``foo`` in ``async with foo:``)
    manager: object = None

    #: The name to which the result of the context manager was assigned
    #: (``"bar"`` in ``async with foo as bar:``), or ``None`` if it wasn't
    #: assigned anywhere or if we couldn't determine where it was assigned.
    varname: Optional[str] = None

    #: The line number of the ``with`` or ``async with`` statement that
    #: entered the context manager, or ``None`` if we couldn't determine it.
    start_line: Optional[int] = None


class InspectionWarning(RuntimeWarning):
    """Warning raised if something goes awry during frame inspection."""


_can_use_trickery: Optional[bool] = None


def contexts_active_in_frame(frame: types.FrameType) -> List[ContextInfo]:
    """Inspects the frame object ``frame`` to try to determine which
    context managers are currently active; returns a list of
    `ContextInfo` objects describing the active context managers
    from outermost to innermost.
    """
    global _can_use_trickery

    if _can_use_trickery is None:
        _can_use_trickery = sys.implementation.name == "cpython" or (
            sys.implementation.name == "pypy"
            and sys.pypy_translation_info["translation.gc"]  # type: ignore
            == "incminimark"
        )
        if _can_use_trickery:
            from contextlib import contextmanager

            @contextmanager
            def noop() -> Iterator[None]:
                yield

            noop_cm = noop()
            contexts_live = []

            def fn() -> Generator[None, None, None]:
                with noop_cm as xyzzy:  # noqa: F841
                    contexts_live.extend(_contexts_active_by_trickery(sys._getframe(0)))
                    yield

            gen = fn()
            try:
                gen.send(None)
                contexts = _contexts_active_by_trickery(gen.gi_frame)
                assert contexts == contexts_live
                assert len(contexts) == 1
                assert contexts[0].varname == "xyzzy" and contexts[0].manager is noop_cm
            except Exception as ex:
                warnings.warn(
                    "Inspection trickery doesn't work on this interpreter: {!r}. "
                    "Enhanced tracebacks will be less detailed. Please file a "
                    "bug.".format(ex),
                    InspectionWarning,
                )
                traceback.print_exc()
                _can_use_trickery = False
        else:
            warnings.warn(
                "Inspection trickery is not supported on this interpreter: "
                "need either CPython, or PyPy with incminimark GC. "
                "Enhanced tracebacks will be less detailed.",
                InspectionWarning,
            )

    if _can_use_trickery:
        try:
            return _contexts_active_by_trickery(frame)
        except Exception as ex:
            warnings.warn(
                "Inspection trickery failed on frame {!r}: {!r}. "
                "Enhanced tracebacks will be less detailed. Please file a bug.".format(
                    frame, ex
                ),
                InspectionWarning,
            )
            traceback.print_exc()
            return _contexts_active_by_referents(frame)
    else:
        return _contexts_active_by_referents(frame)


def _contexts_active_by_referents(frame: types.FrameType) -> List[ContextInfo]:
    """Version of `contexts_active_in_frame` that relies only on
    `gc.get_referents`, and thus can be used on any Python interpreter
    that supports the `gc` module.

    On CPython, this doesn't support running frames -- only those that
    are suspended at a yield or await. It can't determine the
    `~ContextInfo.varname` or `~ContextInfo.start_line` members of
    `ContextInfo`, and it's possible to fool it in some unlikely
    circumstances (e.g., if you have a local variable that points
    directly to an ``__exit__`` or ``__aexit__`` method, or if a context
    manager's ``__exit__`` method is a static method or thinks its name
    is something other than ``__exit__``).
    """
    ret: List[ContextInfo] = []
    for referent in gc.get_referents(frame):
        if isinstance(referent, types.MethodType) and referent.__func__.__name__ in (
            "__exit__",
            "__aexit__",
        ):
            # 'with' and 'async with' statements push a reference to the
            # __exit__ or __aexit__ method that they'll call when exiting.
            ret.append(
                ContextInfo(
                    is_async="a" in referent.__func__.__name__,
                    manager=referent.__self__,
                )
            )
    exiting = _currently_exiting_context(frame)
    if exiting is not None:
        ret.append(ContextInfo(is_async=exiting.is_async, is_exiting=True))
    return ret


def _contexts_active_by_trickery(frame: types.FrameType) -> List[ContextInfo]:
    """Version of `contexts_active_in_frame` that provides full information
    on tested versions of CPython and PyPy by accessing the block stack.
    This is an internal implementation detail so it may stop working as
    Python's internals change. The inspectors use lots of assertions so
    such failures will hopefully downgrade to the by_referents version,
    but there are no guarantees -- they might just segfault if we get
    really unlucky.
    """
    with_block_info = analyze_with_blocks(frame.f_code)
    frame_details = inspect_frame(frame)
    with_blocks = [
        block for block in frame_details.blocks if block.handler in with_block_info
    ]
    exiting = _currently_exiting_context(frame)
    ret = [
        attr.evolve(
            with_block_info[block.handler],
            manager=frame_details.stack[block.level - 1].__self__,  # type: ignore
        )
        for block in with_blocks
    ]
    if exiting is not None:
        ret.append(
            attr.evolve(with_block_info[exiting.cleanup_offset], is_exiting=True)
        )
    locals_by_id = {}
    for name, value in frame.f_locals.items():
        locals_by_id[id(value)] = name
    for idx, info in enumerate(ret):
        if info.manager is not None and info.varname is None:
            ret[idx] = attr.evolve(info, varname=locals_by_id.get(id(info.manager)))
    return ret


@attr.s(auto_attribs=True)
class FrameDetails:
    """A collection of internal interpreter details relating to a currently
    executing or suspended frame.
    """

    @attr.s(auto_attribs=True)
    class FinallyBlock:
        #: The bytecode offset to which control will be transferred if an
        #: exception is raised
        handler: int

        #: The value stack depth at which the handler begins execution
        level: int

    #: Currently active finally blocks in this frame (includes context managers too)
    #: in order from outermost to innermost
    blocks: List[FinallyBlock] = attr.Factory(list)

    #: All values on this frame's value stack
    stack: List[object] = attr.Factory(list)


def inspect_frame(frame: types.FrameType) -> FrameDetails:
    """Return a `FrameDetails` object describing the block stack and value stack
    for the currently executing or suspended frame *frame*.
    """
    # Overwrite this function with the version that's applicable to the
    # running interpreter
    global inspect_frame
    if sys.implementation.name == "cpython":
        from ._frame_tricks_cpython import inspect_frame
    elif sys.implementation.name == "pypy":
        from ._frame_tricks_pypy import inspect_frame
    else:
        raise NotImplementedError("frame details not supported on this interpreter")
    return inspect_frame(frame)


def analyze_with_blocks(code: types.CodeType) -> Dict[int, ContextInfo]:
    """Analyze the bytecode of the given code object, returning a
    partially filled-in `ContextInfo` object for each ``with`` or
    ``async with`` block.

    Each key in the returned mapping uniquely identifies one ``with``
    or ``async with`` block in the function, by specifying the
    bytecode offset of the ``WITH_CLEANUP_START`` (3.8 and earlier) or
    ``WITH_EXCEPT_START`` (3.9 and later) instruction that begins its
    associated exception handler.  The corresponding value is a
    `ContextInfo` object appropriate to that block, with all fields
    except ``manager`` filled in.
    """
    with_block_info: Dict[int, ContextInfo] = {}
    current_line = -1
    insns = list(dis.Bytecode(code))
    for idx, insn in enumerate(insns):
        if insn.starts_line is not None:
            current_line = insn.starts_line
        if insn.opname in ("SETUP_WITH", "SETUP_ASYNC_WITH"):
            store_to = _describe_assignment_target(insns, idx + 1)
            cleanup_offset = insn.argval
            with_block_info[cleanup_offset] = ContextInfo(
                is_async=(insn.opname == "SETUP_ASYNC_WITH"),
                varname=store_to,
                start_line=current_line,
            )
    return with_block_info


@attr.s(auto_attribs=True, slots=True, frozen=True)
class _ExitingContext:
    """Information about the sync or async context manager that's
    currently being exited in a frame.
    """

    #: True for an async context manager, False for a sync context manager.
    is_async: bool

    #: The bytecode offset of the WITH_CLEANUP_START or WITH_EXCEPT_START instruction
    #: that begins the exception handler associated with this context manager.
    cleanup_offset: int


def _currently_exiting_context(frame: types.FrameType) -> Optional[_ExitingContext]:
    """If *frame* is currently suspended waiting for one of its context managers'
    __exit__ or __aexit__ methods to complete, then return an object indicating
    which context manager is exiting and whether it's async or not.
    Otherwise return None.
    """
    code = frame.f_code.co_code
    op = dis.opmap
    offs = frame.f_lasti
    if offs < 0:
        return None

    # Our task here is twofold:
    # - figure out whether `frame` is in the middle of a call to a context
    #   manager __exit__ or __aexit__
    # - if so, figure out *which* context manager, in terms that
    #   can be matched up with the result of analyze_with_blocks()
    #
    # This is rather challenging, because the block stack gets popped
    # before the __exit__ method is called, so we can't just consult
    # the block stack like we do for the context managers that aren't
    # currently exiting. But we can do it if we know something about
    # how the Python bytecode compiler compiles 'with' and 'async
    # with' blocks.
    #
    # There are basically three ways to exit a 'with' block:
    # - falling off the bottom
    # - jumping out (using return, break, or continue)
    # - unwinding due to an exception
    #
    # On 3.7 and earlier, "jumping out" uses the exception-unwinding
    # mechanism, and falling off the bottom falls through into the
    # exception handling block, so there is only one bytecode location
    # where __exit__ or __aexit__ is called. It looks like:
    #
    #     POP_BLOCK          \__ these may be absent if fallthrough is impossible
    #     LOAD_CONST None    /
    #     WITH_CLEANUP_START <-- block stack for exception unwinding points here
    #     GET_AWAITABLE      \
    #     LOAD_CONST None    |-- only if 'async with'
    #     YIELD_FROM         /
    #     WITH_CLEANUP_FINISH
    #     END_FINALLY
    #
    # f_lasti will be at WITH_CLEANUP_START for a synchronous call,
    # LOAD_CONST None for an async call on CPython, or YIELD_FROM for an
    # async call on pypy.  Note that the LOAD_CONST may have some
    # EXTENDED_ARGs before it, in weird cases where None is not one of
    # the first 256 constants.
    #
    # On 3.8, falling off the bottom still falls through into the
    # exception handling block, which looks like:
    #
    #     POP_BLOCK          \__ these may be absent if fallthrough is impossible
    #     BEGIN_FINALLY      /
    #     WITH_CLEANUP_START <-- block stack for exception unwinding points here
    #     GET_AWAITABLE      \
    #     LOAD_CONST None    |-- only if 'async with'
    #     YIELD_FROM         /
    #     WITH_CLEANUP_FINISH
    #     END_FINALLY
    #
    # But, now each instance of jumping-out inlines its own cleanup. The cleanup
    # sequence is the same as the terminal sequence except that it may have a
    # ROT_TWO after POP_BLOCK (for non-constant 'return' jumps only, not 'break' or
    # 'continue') and it ends with POP_FINALLY rather than END_FINALLY.
    # Since there can be multiple WITH_CLEANUP_START opcodes that clean up
    # the same 'with' block, we can't assume the WITH_CLEANUP_START near f_lasti is
    # the one whose offset is named in the SETUP_WITH that analyze_with_blocks()
    # found. Instead, we'll build a basic control-flow graph to see what offset
    # was in the block that just got POP_BLOCK'ed.
    #
    # On 3.9, this was further split up so that "falling off the bottom" and
    # "unwinding due to an exception" execute different code. Falling off the bottom:
    #
    #     POP_BLOCK
    #     LOAD_CONST None
    #     DUP_TOP
    #     DUP_TOP
    #     CALL_FUNCTION 3    <-- calls __exit__(None, None, None)
    #     GET_AWAITABLE      \
    #     LOAD_CONST None    |-- only if 'async with'
    #     YIELD_FROM         /
    #     POP_TOP            <-- return value of __exit__ ignored
    #
    # Jumping out: same as falling off the bottom, except with possible ROT_TWO
    # after POP_BLOCK.
    #
    # Unwinding on exception:
    #
    #     WITH_EXCEPT_START  <-- block stack points here
    #     GET_AWAITABLE      \
    #     LOAD_CONST None    |-- only if 'async with'
    #     YIELD_FROM         /
    #     POP_JUMP_IF_TRUE x <-- jumps over the RERAISE
    #     RERAISE
    #     POP_TOP
    #     POP_TOP
    #     POP_TOP
    #     POP_EXCEPT
    #     POP_TOP
    #
    # Armed with that context, the below code will hopefully make a bit more sense!

    # See if we're at a context manager __exit__ or __aexit__ call
    is_async = False
    if code[offs] == op["YIELD_FROM"] or (
        offs + 2 < len(code) and code[offs + 2] == op["YIELD_FROM"]
    ):
        # Async calls have lasti pointing at YIELD_FROM or LOAD_CONST
        is_async = True
        if code[offs] == op["YIELD_FROM"]:
            # If lasti points to YIELD_FROM (pypy convention), move backward
            # to point to LOAD_CONST (cpython convention)
            offs -= 2
        if code[offs] != op["LOAD_CONST"]:  # pragma: no cover
            warnings.warn(
                f"Surprise during analysis of {frame.f_code!r}: YIELD_FROM at {offs} "
                f"not preceded by LOAD_CONST -- please file a bug",
                InspectionWarning,
            )
            return None
        # Backtrack one more to find GET_AWAITABLE
        offs -= 2
        while offs and code[offs] == op["EXTENDED_ARG"]:
            # If LOAD_CONST had an EXTENDED_ARG then skip over those.
            # This is very unlikely -- would require none of the first
            # 256 constants used in a function to be None.
            offs -= 2
        if code[offs] != op["GET_AWAITABLE"]:
            # Non-awaity use of 'yield from' --> must not be an __aexit__ call we're in
            return None
        # And finally go back one more to reach a CALL_FUNCTION,
        # WITH_CLEANUP_START, or WITH_EXCEPT_START, which can be handled
        # the same as in the synchronous case
        offs -= 2

    if sys.version_info < (3, 8):
        # 3.7 and below: every exit call is done from a single WITH_CLEANUP_START
        # location per 'with' block
        if code[offs] == op["WITH_CLEANUP_START"]:
            return _ExitingContext(is_async=is_async, cleanup_offset=offs)
        return None
    elif sys.version_info < (3, 9):
        # 3.8: they all use WITH_CLEANUP_START, but there might be multiple instances;
        # backtrack to the preceding POP_BLOCK
        if offs < 4 or code[offs] != op["WITH_CLEANUP_START"]:
            return None
        if code[offs - 2] != op["BEGIN_FINALLY"]:
            # Every jumping-out exit uses BEGIN_FINALLY before WITH_CLEANUP_START,
            # but it's possible for the end-of-block exit to not have a preceding
            # BEGIN_FINALLY. If we're in that situation, then we're in the
            # exception handler, so we already know its offset.
            return _ExitingContext(is_async=is_async, cleanup_offset=offs)
        offs -= 4
        if offs and code[offs] == op["ROT_TWO"]:
            offs -= 2
    else:
        # 3.9 and above: either WITH_EXCEPT_START at the handler
        # offset, or LOAD_CONST DUP_TOP DUP_TOP CALL_FUNCTION
        # somewhere else (that particular sequence is not produced by
        # anything else, as far as I can tell)
        if code[offs] == op["WITH_EXCEPT_START"]:
            return _ExitingContext(is_async=is_async, cleanup_offset=offs)
        if offs < 8 or code[offs - 6 : offs + 2 : 2] != bytes(
            [op["LOAD_CONST"], op["DUP_TOP"], op["DUP_TOP"], op["CALL_FUNCTION"]]
        ):
            return None
        # Backtrack from CALL_FUNCTION to the preceding POP_BLOCK
        offs -= 8
        while offs and code[offs] == op["EXTENDED_ARG"]:
            offs -= 2
        if offs and code[offs] == op["ROT_TWO"]:
            offs -= 2

    # If we get here, we're on 3.8 or later and offs is the offset of a
    # POP_BLOCK opcode that popped the context manager block whose offset
    # we want to return.
    if code[offs] != op["POP_BLOCK"]:  # pragma: no cover
        warnings.warn(
            f"Surprise during analysis of {frame.f_code!r}: __exit__ call at {offs} "
            f"not preceded by POP_BLOCK -- please file a bug",
            InspectionWarning,
        )
        return None

    pop_block_offs = offs

    # The block stack on 3.8+ is pretty simple: there's only one
    # type of block used outside exception handling, it's
    # pushed by any of SETUP_FINALLY, SETUP_WITH, SETUP_ASYNC_WITH
    # and popped by POP_BLOCK. This is the block type that will
    # let us match up our POP_BLOCK with its corresponding
    # SETUP_WITH or SETUP_ASYNC_WITH, so it's the only one we need
    # to worry about.

    # We represent the state of the block stack at each bytecode offset
    # as a list of the bytecode offsets of exception handlers for
    # each 'finally:'.
    BlockStack = List[int]

    # List of (bytecode offset, block stack that exists just before the
    # instruction at that offset is executed)
    todo: Deque[Tuple[int, BlockStack]] = collections.deque([(0, [])])

    # Bytecode offsets we've already visited
    seen: Set[int] = set()

    while todo:  # pragma: no branch
        offs, stack = todo.popleft()
        if offs in seen:
            continue
        seen.add(offs)
        arg = code[offs + 1]
        while code[offs] == op["EXTENDED_ARG"]:
            offs += 2
            arg = (arg << 8) | code[offs + 1]
        if code[offs] in dis.hasjabs:
            todo.append((arg, stack[:]))
        if code[offs] in dis.hasjrel:
            todo.append((offs + 2 + arg, stack[:]))
            # All three SETUP_* opcodes are in the hasjrel list,
            # because they indicate a possible jump to the handler
            # whose relative offset is named in their argument.
            # That handler is entered with the finally block already
            # popped, so it's correct that we record it in todo
            # before updating the block stack.
            if code[offs] in (
                op["SETUP_FINALLY"],
                op["SETUP_WITH"],
                op["SETUP_ASYNC_WITH"],
            ):
                stack.append(offs + 2 + arg)
        if code[offs] == op["POP_BLOCK"]:
            if offs == pop_block_offs:
                # We found the one we're looking for!
                return _ExitingContext(is_async=is_async, cleanup_offset=stack[-1])
            stack.pop()
        if code[offs] not in (
            op["JUMP_FORWARD"],
            op["JUMP_ABSOLUTE"],
            op["RETURN_VALUE"],
            op["RAISE_VARARGS"],
            op.get("RERAISE"),  # 3.9 only
        ):
            # The above are the unconditional control transfer opcodes.
            # If we're not one of those, then we'll continue to the following
            # line at least sometimes.
            todo.append((offs + 2, stack))

    warnings.warn(
        f"Surprise during analysis of {frame.f_code!r}: POP_BLOCK at offset "
        f"{pop_block_offs} doesn't appear reachable -- please file a bug",
        InspectionWarning,
    )
    return None


def _describe_assignment_target(
    insns: List[dis.Instruction], start_idx: int,
) -> Optional[str]:
    """Given that insns[start_idx] and beyond constitute a series of
    instructions that assign the top-of-stack value somewhere, this
    function returns a string description of where it's getting
    assigned, or None if we can't figure it out.  Understands simple
    names, attributes, subscripting, unpacking, and
    positional-only function calls.
    """

    if start_idx >= len(insns) or insns[start_idx].opname == "POP_TOP":
        return None
    if insns[start_idx].opname == "STORE_FAST":
        return cast(str, insns[start_idx].argval)

    def format_tuple(values: Sequence[str]) -> str:
        if len(values) == 1:
            return "({},)".format(values[0])
        return "({})".format(", ".join(values))

    idx = start_idx

    def next_target() -> str:
        nonlocal idx
        stack: List[str] = []
        while True:
            insn = insns[idx]
            idx += 1
            if insn.opname == "EXTENDED_ARG":
                continue
            if insn.opname in (
                # fmt: off
                "LOAD_GLOBAL", "LOAD_FAST", "LOAD_NAME", "LOAD_DEREF",
                "STORE_GLOBAL", "STORE_FAST", "STORE_NAME", "STORE_DEREF",
                # fmt: on
            ):
                stack.append(insn.argval)
            elif insn.opname in (
                # LOOKUP_METHOD is pypy-only
                "LOAD_ATTR",
                "LOAD_METHOD",
                "LOOKUP_METHOD",
                "STORE_ATTR",
            ):
                obj = stack.pop()
                stack.append(f"{obj}.{insn.argval}")
            elif insn.opname == "LOAD_CONST":
                stack.append(insn.argrepr)
            elif insn.opname in ("BINARY_SUBSCR", "STORE_SUBSCR"):
                index = stack.pop()
                container = stack.pop()
                stack.append(f"{container}[{index}]")
            elif insn.opname == "UNPACK_SEQUENCE":
                values = [next_target() for _ in range(insn.argval)]
                stack.append(format_tuple(values))
            elif insn.opname == "UNPACK_EX":
                before = [next_target() for _ in range(insn.argval & 0xFF)]
                rest = next_target()
                after = [next_target() for _ in range(insn.argval >> 8)]
                stack.append(format_tuple(before + [f"*{rest}"] + after))
            elif insn.opname in ("CALL_FUNCTION", "CALL_METHOD"):
                if insn.argval == 0:
                    args = []
                else:
                    args = stack[-insn.argval :]
                    del stack[-insn.argval :]
                func = stack.pop()
                stack.append("{}({})".format(func, ", ".join(args)))
            elif insn.opname == "DUP_TOP":
                # Walrus assignments get here
                stack.append(stack[-1])
            elif insn.opname == "POP_TOP":  # pragma: no cover
                # No known way to get here -- POP_TOP as sole insn is
                # handled at the top of this function
                stack.pop()
            else:
                raise ValueError(f"{insn.opname} in assignment target not supported")
            if insn.opname.startswith(("STORE_", "UNPACK_")):
                break
        if len(stack) != 1:
            # Walrus assignments can get here
            raise ValueError("Assignment occurred at unsupported stack depth")
        return stack[0]

    try:
        return next_target()
    except (ValueError, IndexError):
        return None
