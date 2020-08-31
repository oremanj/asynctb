import attr
import dis
import gc
import sys
import traceback
import types
import warnings
from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence, Tuple, cast


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


_can_use_trickery: Optional[bool] = None


def contexts_active_in_frame(frame: types.FrameType) -> List[ContextInfo]:
    """Inspects the frame object ``frame`` to try to determine which
    context managers are currently active; returns a list of
    `ContextInfo` objects describing the active context managers
    from outermost to innermost.
    """
    global _can_use_trickery

    if _can_use_trickery is None:
        _can_use_trickery = (
            sys.implementation.name == "cpython"
            or (
                sys.implementation.name == "pypy"
                and sys.pypy_translation_info[  # type: ignore
                    "translation.gc"
                ] == "incminimark"
            )
        )
        if _can_use_trickery:
            from contextlib import contextmanager

            @contextmanager
            def noop() -> Iterator[None]:
                yield

            noop_cm = noop()
            contexts_live = []

            def fn() -> Generator[None, None, None]:
                with noop_cm as xyzzy:
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
                    "bug.".format(ex)
                )
                traceback.print_exc()
                _can_use_trickery = False
        else:
            warnings.warn(
                "Inspection trickery is not supported on this interpreter: "
                "need either CPython, or PyPy with incminimark GC. "
                "Enhanced tracebacks will be less detailed."
            )

    if _can_use_trickery:
        try:
            return _contexts_active_by_trickery(frame)
        except Exception as ex:
            import pdb; pdb.post_mortem(ex.__traceback__)
            warnings.warn(
                "Inspection trickery failed on frame {!r}: {!r}. "
                "Enhanced tracebacks will be less detailed. Please file a bug.".format(
                    frame, ex
                )
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
    directly to an ``__exit__`` or ``__aexit__`` method).
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
        handler: int  # bytecode offset where the finally handler starts
        level: int  # value stack depth at which handler begins execution

    # Currently active finally blocks in this frame (includes context managers too)
    # in order from outermost to innermost
    blocks: List[FinallyBlock] = attr.Factory(list)

    # All values on this frame's value stack
    stack: List[object] = attr.Factory(list)


def inspect_frame(frame: types.FrameType) -> FrameDetails:
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

    Each key in the returned mapping is the bytecode offset of a
    ``WITH_CLEANUP_START`` instruction that ends a ``with`` or ``async
    with`` block. The corresponding value is a `ContextInfo` object
    appropriate to that block, with all fields except ``manager``
    filled in.
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

    #: The bytecode offset of the WITH_CLEANUP_START instruction corresponding
    #: to this context manager.
    cleanup_offset: int


def _currently_exiting_context(frame: types.FrameType) -> Optional[_ExitingContext]:
    """If *frame* is currently suspended waiting for one of its context managers'
    __exit__ or __aexit__ methods to complete, then return a tuple
    (corresponding WITH_CLEANUP_START bytecode offset, is_async).
    Otherwise return None.
    """
    code = frame.f_code.co_code
    op = dis.opmap
    if frame.f_lasti < 0:
        return None
    if (
        code[frame.f_lasti] == op["WITH_CLEANUP_START"]
        and code[frame.f_lasti + 2] == op["WITH_CLEANUP_FINISH"]
    ):
        return _ExitingContext(is_async=False, cleanup_offset=frame.f_lasti)

    # PyPy suspends with lasti pointing at the YIELD_FROM; CPython suspends
    # with lasti pointing just before it (at LOAD_CONST).
    if bytes([op["YIELD_FROM"], op["WITH_CLEANUP_FINISH"]]) in code[
        frame.f_lasti : frame.f_lasti + 6 : 2
    ]:
        offs = frame.f_lasti - (4 if code[frame.f_lasti] == op["YIELD_FROM"] else 2)
        if code[offs + 2] != op["LOAD_CONST"]:  # pragma: no cover
            return None
        while code[offs] == op["EXTENDED_ARG"]:
            offs -= 2
        if (
            code[offs] == op["GET_AWAITABLE"]
            and code[offs - 2] == op["WITH_CLEANUP_START"]
        ):
            return _ExitingContext(is_async=True, cleanup_offset=offs - 2)
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
            done = insn.opname.startswith(("STORE_", "UNPACK_"))
            if insn.opname == "EXTENDED_ARG":
                continue
            if insn.opname in (
                # fmt: off
                "LOAD_GLOBAL", "LOAD_FAST", "LOAD_NAME", "LOAD_DEREF",
                "STORE_GLOBAL", "STORE_FAST", "STORE_NAME", "STORE_DEREF",
                # fmt: on
            ):
                stack.append(insn.argval)
            elif insn.opname in ("LOAD_ATTR", "LOAD_METHOD", "STORE_ATTR"):
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
                args = stack[-insn.argval :]
                del stack[-insn.argval :]
                func = stack.pop()
                stack.append("{}({})".format(func, ", ".join(args)))
            elif insn.opname == "POP_TOP":
                stack.pop()
            else:
                raise ValueError(f"{insn.opname} in assignment target not supported")
            if insn.opname.startswith(("STORE_", "UNPACK_")):
                break
        if len(stack) != 1:
            raise ValueError(f"Assignment occurred at unsupported stack depth")
        return stack[0]

    try:
        return next_target()
    except (ValueError, IndexError):
        return None
