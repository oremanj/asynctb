import attr
import contextlib
import gc
import importlib
import inspect
import linecache
import math
import sys
import threading
import traceback
import types
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from ._frames import ContextInfo, contexts_active_in_frame
from ._util import IdentitySet

try:
    from greenlet import greenlet as GreenletType
except ImportError:
    class GreenletType:
        pass
else:
    try:
        from greenback._impl import _greenback_shim, await_ as greenback_await
    except ImportError:
        greenback_await = _greenback_shim = lambda: None


@attr.s(auto_attribs=True, slots=True, eq=True, frozen=True)
class FrameInfo:
    """Information about one frame in a `Traceback`."""

    owner: object
    frame: types.FrameType
    lineno: int = attr.Factory(lambda self: self.frame.f_lineno, takes_self=True)
    _override_line: Optional[str] = None
    context_manager: object = None
    context_name: Optional[str] = None

    @property
    def filename(self) -> str:
        """The filename of the Python file from which this frame's code was
        imported."""
        return self.frame.f_code.co_filename

    @property
    def funcname(self) -> str:
        """The name of the function whose code this frame is executing."""
        return self.frame.f_code.co_name

    @property
    def linetext(self) -> str:
        """The text of the line of source code that this frame is executing.

        The return value has leading and trailing whitespace stripped, and
        does not end in a newline. In some cases (such as callbacks in an
        `~contextlib.ExitStack`), where we can't determine the original
        source line, we return a synthesized line beginning with ``#``.
        """
        if self._override_line is not None:
            return self._override_line
        if self.lineno == 0:
            return ""
        return linecache.getline(
            self.filename, self.lineno, self.frame.f_globals
        ).strip()

    @property
    def _context_descr(self) -> str:
        if self.context_manager is not None:
            typename = type(self.context_manager).__name__
            varname = self.context_name or "_"
            return f" ({varname}: {typename})"
        elif self.context_name is not None:
            return f" ({self.context_name})"
        return ""

    def __str__(self) -> str:
        ret = (
            f"  File {self.filename}, line {self.lineno}, in {self.funcname}"
            f"{self._context_descr}"
        )
        line = self.linetext
        if line:
            ret += f"\n    {line}"
        return ret + "\n"

    def as_stdlib_summary(
        self, *, capture_locals: bool = False
    ) -> traceback.FrameSummary:
        """Return a representation of this frame as a standard
        `traceback.FrameSummary`. The result can be pickled and will
        not keep frames alive, at the expense of some loss of information.

        If this frame introduces a context manager in an ref:`enhanced traceback
        <enhanced-tb>`, information about the name and type of the context manager
        will be appended to the function *name* in the returned
        `~traceback.FrameSummary`, in parentheses after a space. This results
        in reasonable output from :meth:`traceback.StackSummary.format`.

        If *capture_locals* is True, then the returned `~traceback.FrameSummary`
        will contain the stringified object representations of all local variables
        in this frame.
        """
        funcname = self.funcname + self._context_descr
        if not capture_locals:
            save_locals = None
        elif self.context_manager is not None:
            save_locals = {"<context manager>": self.context_manager}
        else:
            save_locals = self.frame.f_locals

        return traceback.FrameSummary(
            self.frame.f_code.co_filename,
            self.lineno,
            funcname,
            locals=save_locals,
            line=self._override_line,
        )


# We have two modes in which we might need to iterate over frames:
# "inward" (when working with a suspended coroutine/generator,
# where the outer object has a cr_await/gi_yieldfrom member that
# refers to the inner) or "outward" (when working with a greenlet or
# with running frames, in which the inner frame has an f_back that
# refers to the outer). We sometimes need to switch between these.
# This is implemented by defining two generator functions, iterate_suspended
# and iterate_running, each of which produces a generator of type
# FrameProducer. Mode switching is implemented trampoline-style,
# by allowing each FrameProducer to return another FrameProducer
# that should take over for it.
#
# The return type should be Optional[FrameProducer] but mypy doesn't
# support recursive types yet.
FrameProducer = Generator[FrameInfo, None, Any]


@attr.s(auto_attribs=True, slots=True, eq=True, frozen=True)
class Traceback:
    """A summary of the current execution context of a running or
    suspended function, coroutine, etc.

    A `Traceback` consists of a series of frames (each represented as a
    `FrameInfo` object), ordered from outermost (earlier call) to
    innermost (more recent call).

    You can get a `Traceback` for a coroutine, greenlet, or (sync or
    async) generator using :meth:`Traceback.of`, or for the current
    stack using :meth:`Traceback.since`.
    """

    frames: Sequence[FrameInfo]
    error: Optional[Exception] = None

    @classmethod
    def of(cls, owner: object, *, with_context_info: bool = True) -> "Traceback":
        """Return a traceback reflecting the current stack of *owner*, which
        must be a coroutine object, greenlet object, or (sync or
        async) generator iterator. A generator or coroutine may be
        either running or suspended. If it's running in a different
        thread, we'll still attempt to extract a traceback, but might
        not be able to. A greenlet must be suspended.

        Produce an :ref:`enhanced traceback <enhanced-tb>` if *with_context_info*
        is True (the default), or a basic traceback if *with_context_info* is False.

        """
        if isinstance(owner, GreenletType):
            if owner.gr_frame is None:
                return Traceback(
                    frames=(),
                    error=RuntimeError(
                        "Traceback.of(greenlet) requires that the greenlet be suspended"
                    ),
                )
            producer = iterate_running(None, owner.gr_frame, owner, with_context_info)
        else:
            producer = iterate_suspended(unwrap_owner(owner), with_context_info)
        return cls._make(producer)

    @classmethod
    def since(
        cls, outer_frame: Optional[types.FrameType], *, with_context_info: bool = True
    ) -> "Traceback":
        """Return a traceback reflecting the currently executing frames
        that were directly or indirectly called by *outer_frame*.

        If *outer_frame* is a frame on the current thread's stack, the resulting
        traceback will start with *outer_frame* and end with the immediate
        caller of :meth:`since`.

        If *outer_frame* is a frame on some other thread's stack, and it remains
        there throughout the traceback extraction process, the resulting
        traceback will start with *outer_frame* and end with some frame
        that was recently the innermost frame on that thread's stack.

        .. note:: If *other_frame* is not continuously on the same other thread's
           stack during the traceback extraction process, you're likely to get
           a one-frame traceback with an `error`. It's not possible to prevent
           thread switching from within Python code, so we can't do better than
           this without a C extension.

        If *outer_frame* is None, the resulting traceback contains all frames
        on the current thread's stack, starting with the outermost and ending
        with the immediate caller of :meth:`since`.

        In any other case -- if *outer_frame* belongs to a suspended coroutine,
        generator, greenlet, or if it starts or stops running on another thread
        while :meth:`since` is executing -- you will get a `Traceback` containing
        information only on *outer_frame* itself, whose `error` member describes
        the reason more information can't be provided.

        Produce an :ref:`enhanced traceback <enhanced-tb>` if *with_context_info*
        is True (the default), or a basic traceback if *with_context_info* is False.
        """
        if outer_frame is not None and not isinstance(outer_frame, types.FrameType):
            raise TypeError(f"outer_frame must be a frame, not {outer_frame!r}")
        return cls._make(iterate_running(outer_frame, None, with_context_info))

    @classmethod
    def until(
        cls,
        inner_frame: types.FrameType,
        *,
        limit: Union[int, types.FrameType, None] = None,
        with_context_info: bool = True,
    ) -> "Traceback":
        """Return a traceback reflecting the currently executing frames
        that are direct or indirect callers of *inner_frame*.

        If *inner_frame* belongs to a suspended coroutine or
        generator, or if it otherwise is not linked to other frames
        via its ``f_back`` attribute, then the returned traceback will
        contain only *inner_frame* and not any of its callers.

        If a *limit* is specified, only some of the callers of *inner_frame*
        will be returned. If the *limit* is a frame, then it must be an indirect
        caller of *inner_frame* and it will be the first frame in the returned
        traceback. Otherwise, the *limit* must be a positive integer, and the
        traceback will start with the *limit*'th parent of *inner_frame*.

        Produce an :ref:`enhanced traceback <enhanced-tb>` if *with_context_info*
        is True (the default), or a basic traceback if *with_context_info* is False.
        """
        if limit is None:
            outer_frame = None
        elif isinstance(limit, types.FrameType):
            outer_frame = inner_frame
            while outer_frame is not limit and outer_frame is not None:
                outer_frame = outer_frame.f_back
            if outer_frame is None:
                raise RuntimeError(
                    f"{limit} is not an indirect caller of {inner_frame}"
                )
        elif isinstance(limit, int):
            outer_frame = inner_frame
            while limit > 0 and outer_frame is not None:
                outer_frame = outer_frame.f_back
                limit -= 1
        else:
            raise TypeError(
                "'limit' argument must be a frame or integer, not "
                + type(limit).__name__
            )

        return cls._make(iterate_running(outer_frame, inner_frame, with_context_info))

    @classmethod
    def _make(cls, producer: FrameProducer) -> "Traceback":
        frames: List[FrameInfo] = []
        error: Optional[Exception] = None
        try:
            frames.extend(frames_from_producer(producer))
        except Exception as exc:
            error = exc
        return cls(tuple(frames), error)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[FrameInfo]:
        return iter(self.frames)

    def __str__(self) -> str:
        lines = []
        if self.frames:
            lines.append("Traceback (most recent call last):\n")
            lines.extend(self.as_stdlib_summary().format())
        if self.error:
            lines.append(
                "Error while extracting traceback: "
                + "".join(traceback.format_exception_only(type(self.error), self.error))
            )
            lines.extend(traceback.format_tb(self.error.__traceback__))
        return "".join(lines)

    def as_stdlib_summary(
        self, *, capture_locals: bool = False
    ) -> traceback.StackSummary:
        """Return a representation of this frame as a standard
        `traceback.StackSummary`. The result can be pickled and will
        not keep frames alive, at the expense of some loss of information
        (including the values of `FrameInfo.context_manager` and
        `FrameInfo.context_name`).

        If *capture_locals* is True, then the individual `~traceback.FrameSummary`
        objects in the returned `~traceback.StackSummary` will contain the
        stringified object representations of all local variables in each frame.
        """
        return traceback.StackSummary.from_list(
            frame.as_stdlib_summary(capture_locals=capture_locals)
            for frame in self.frames
        )


# -------- Everything below this point is implementation details --------


# Singleton object returned as the next_owner from frame_and_next() to indicate
# that a generator or coroutine is currently running and that we therefore
# need to switch to iterate_running().
RUNNING = object()


def frames_from_producer(producer: Optional[FrameProducer]) -> Iterator[FrameInfo]:
    # This is the "trampoline" that converts a FrameProducer into a
    # simple iterator over FrameInfos. See the comment on the FrameProducer
    # type definition for more details.
    while producer is not None:
        try:
            yield next(producer)
        except StopIteration as exc:
            producer = exc.value


def iterate_suspended(
    owner: object, with_context_info: bool, switch_count: int = 0
) -> FrameProducer:
    """Yield information about a series of frames representing the current
    stack of the suspended generator or coroutine object *owner*.

    If *with_context_info* is True, yield additional frames
    representing context managers, to produce an enhanced traceback.

    *switch_count* tracks the number of times we have switched modes
    from inward to outward or vice versa, as a guard against infinite
    loops.
    """

    while owner is not None:
        this_frame = frame_from_owner(owner)
        if this_frame is None:
            # Exhausted generator/coroutine has no traceback
            return

        next_owner = next_from_owner(owner)
        if next_owner is RUNNING:
            # If it's running, cr_await/ag_await/gi_yieldfrom aren't available,
            # so we need an alternate approach.
            return iterate_running(
                outer_frame=this_frame,
                inner_frame=None,
                with_context_info=with_context_info,
                first_owner=owner,
                switch_count=switch_count + 1,
            )

        if (
            isinstance(owner, types.GeneratorType)
            and owner.gi_code is _greenback_shim.__code__
        ):
            # Greenback shim. Is the child coroutine suspended in an await_()?
            child_greenlet = this_frame.f_locals.get("child_greenlet")
            orig_coro = this_frame.f_locals.get("orig_coro")
            gr_frame = getattr(child_greenlet, "gr_frame", None)
            if gr_frame is not None:
                # Yep; switch to walking the greenlet stack, since orig_coro
                # will look "running" but it's not on any thread's stack.
                return iterate_running(
                    outer_frame=None,
                    inner_frame=gr_frame,
                    with_context_info=with_context_info,
                    first_owner=orig_coro,
                    switch_count=switch_count + 1,
                )
            elif orig_coro is not None:
                # No greenlet, so child is suspended at a regular await.
                # Continue the traceback by walking the coroutine's frames.
                owner = orig_coro
                continue

            # This might happen if the coroutine gets resumed in a different
            # thread while we're looking at it. It's more challenging to recover
            # from that race than it is without greenback, so we don't bother
            # trying.
            yield from one_frame_traceback(owner, this_frame, None, with_context_info)
            raise RuntimeError(
                "Can't identify what's going on with the greenback shim in the above "
                "frame"
            )

        # Otherwise, the generator/coroutine is suspended. Yield info
        # about this_frame (and its context managers), then continue
        # down the stack to next_owner (the thing this owner is
        # awaiting).
        try:
            next_frame = frame_from_owner(next_owner)
        except Exception:
            next_frame = None
        yield from one_frame_traceback(
            owner, this_frame, next_frame, with_context_info,
        )
        owner = unwrap_owner(next_owner)


def iterate_running(
    outer_frame: Optional[types.FrameType],
    inner_frame: Optional[types.FrameType],
    with_context_info: bool,
    first_owner: object = None,
    switch_count: int = 0,
) -> FrameProducer:
    """Yield information about a series of frames linked via f_back
    members from inner_frame out to outer_frame. (They will be yielded
    in the opposite order, outer_frame first.)

    If outer_frame is None, continue until the outermost frame on inner_frame's
    stack is reached. If inner_frame is None, find a currently-executing frame
    on any thread that has outer_frame on its stack. If both are None, produce
    frames for the current thread's entire stack.

    The first yielded FrameInfo will have its `owner` attribute set to
    *first_owner*; this is used when producing the stack of a running
    coroutine or generator. All subsequent FrameInfos will have owner=None.

    If *with_context_info* is True, yield additional frames
    representing context managers, to produce an enhanced traceback.

    *switch_count* tracks the number of times we have switched modes
    from inward to outward or vice versa, as a guard against infinite
    loops.
    """

    # Running frames are linked "outward" (from the current frame
    # backwards via f_back members), not "inward" (even if the frames
    # belong to a coroutine/generator/etc, the cr_await / gi_yieldfrom
    # / etc members are None). Thus, in order to know what comes after
    # outer_frame in the traceback, we have to find a currently
    # executing frame that has outer_frame on its stack.

    def try_from(potential_inner_frame: types.FrameType) -> List[types.FrameType]:
        """If potential_inner_frame has outer_frame somewhere up its callstack,
        return the list of frames between them, starting with
        outer_frame and ending with potential_inner_frame. Otherwise, return [].
        If outer_frame is None, return all frames enclosing
        potential_inner_frame, including potential_inner_frame.
        """
        frames: List[types.FrameType] = []
        current: Optional[types.FrameType] = potential_inner_frame
        while current is not None:
            frames.append(current)
            if current is outer_frame:
                break
            current = current.f_back
        if current is outer_frame:
            return frames[::-1]
        return []

    if inner_frame is None:
        # Argument of 2 skips this frame and frames_from_producer(), so as not
        # to expose implementation details.
        frames = try_from(sys._getframe(2))
    else:
        frames = try_from(inner_frame)

    if not frames and inner_frame is None:
        # outer_frame isn't on *our* stack, but it might be on some
        # other thread's stack. It would be nice to avoid yielding
        # the GIL while we look, but that doesn't appear to be
        # possible -- sys.setswitchinterval() changes apply only
        # to those threads that have yielded the GIL since it was
        # called. We'll just have to accept the potential race
        # condition.
        for ident, inner_frame in sys._current_frames().items():
            if ident != threading.get_ident():
                frames = try_from(inner_frame)
                if frames:
                    break
        else:
            # We couldn't find any thread that had outer_frame
            # on its stack.
            if first_owner is not None and switch_count < 50:
                if next_from_owner(first_owner) is not RUNNING:
                    # It was running when we started looking, but now
                    # is suspended again. Switch to the logic for suspended
                    # frames. (There's no specific limit to how many times this
                    # can occur; our only argument against an infinite
                    # recursion here is probabilistic, but it's a pretty good
                    # one.)
                    return iterate_suspended(
                        owner=first_owner,
                        with_context_info=with_context_info,
                        switch_count=switch_count + 1,
                    )
                else:
                    # Possibly the owner became suspended before we
                    # extracted _current_frames() and then started running
                    # again before we called frame_and_next(). Or possibly
                    # we've become confused and found a frame that isn't
                    # running at all, e.g., because it's part of a
                    # suspended greenlet. Assume the former until we hit
                    # our mode-switching limit, at which point give up
                    # so as not to infinite-loop in the latter case.
                    return iterate_running(
                        outer_frame=outer_frame,
                        inner_frame=None,
                        with_context_info=with_context_info,
                        first_owner=first_owner,
                        switch_count=switch_count + 1,
                    )

    if not frames:
        # If outer_frame is None, try_from() always returns a non-empty
        # list, so we can only get here if outer_frame is not None.
        assert outer_frame is not None
        yield from one_frame_traceback(
            first_owner, outer_frame, None, with_context_info
        )
        raise RuntimeError(
            "Couldn't find where the above frame is running, so "
            "can't continue traceback"
        )

    for this_frame, next_frame in zip(frames, frames[1:] + [None]):
        if this_frame.f_code is greenback_await.__code__:
            # Greenback-mediated await of async function from sync land.
            # If we have a coroutine to descend into, do so;
            # otherwise the traceback will unhelpfully stop here.
            # This works whether the coro is running or not.
            # (The only way to get coro=None is if we're taking
            # the traceback in the early part of await_() before
            # coro is assigned.)
            coro = this_frame.f_locals.get("coro")
            if coro is not None:
                return iterate_suspended(
                    owner=coro,
                    with_context_info=with_context_info,
                    switch_count=switch_count + 1,
                )

        yield from one_frame_traceback(
            first_owner, this_frame, next_frame, with_context_info
        )
        first_owner = None


# Frames executing any of these code objects will not be included
# in tracebacks.
SKIP_CODE_OBJECTS = IdentitySet[types.CodeType]()


def one_frame_traceback(
    owner: object,
    this_frame: types.FrameType,
    next_frame: Optional[types.FrameType],
    with_context_info: bool,
) -> Iterator[FrameInfo]:
    """Yield a series of FrameInfos representing a single frame
    *this_frame* in the straight-line traceback.

    If *this_frame* is executing any of the code objects named in
    SKIP_CODE_OBJECTS, then nothing will be yielded. Otherwise,
    yields a series of zero or more FrameInfos describing the
    context managers active in *this_frame* (only if with_context_info is True)
    followed by one FrameInfo describing *this_frame* itself (always).

    *next_frame* should be the next inner frame in the traceback after
    *this_frame*, or None; it is used to determine the context manager
    object currently being exited, if any.
    """
    if this_frame.f_code in SKIP_CODE_OBJECTS:
        return
    if with_context_info:
        for context in contexts_active_in_frame(this_frame):
            if context.is_exiting and next_frame is not None:
                # Infer the context manager being exited based on the
                # self argument to its __exit__ or __aexit__ method in
                # the next frame
                args = inspect.getargvalues(next_frame)
                if args.args:
                    context = attr.evolve(context, manager=args.locals[args.args[0]])
            yield from crawl_context(owner, this_frame, context)
    yield FrameInfo(owner, this_frame)


class NotPresent:
    """Dummy type used to stub out a type with special support, if the actual
    type isn't available in this environment.
    """


def try_import(module: str, name: str) -> Any:
    """Return the result of 'from module import name', or NotPresent if the
    module or attribute doesn't exist.
    """
    try:
        ns = importlib.import_module(module)
    except ImportError:
        return NotPresent
    try:
        return getattr(ns, name)
    except AttributeError:  # pragma: no cover
        return NotPresent


# The type of the context manager returned by trio.open_nursery()
TrioNurseryManager = try_import("trio._core._run", "NurseryManager")

# The type of the context manager object returned by a function that is
# decorated with @async_generator.asynccontextmanager
AGCMBackport = try_import("async_generator._util", "_AsyncGeneratorContextManager")

# The type of the async generator iterator returned by a function that is
# decorated with @async_generator.async_generator
AsyncGeneratorBackport = try_import("async_generator._impl", "AsyncGenerator")

# The type of the internal awaitable used in @async_generator backport
# asend/athrow calls
AsyncGeneratorBackportNextIter = try_import("async_generator._impl", "ANextIter")

# The type of the context manager returned by greenback.async_context()
GreenbackAsyncContext = try_import("greenback", "async_context")


def _get_native_awaitable_types():
    async def some_asyncgen():
        yield

    aw = some_asyncgen().asend(None)
    asend_type = type(aw)
    aw.close()
    aw = some_asyncgen().athrow(ValueError)
    athrow_type = type(aw)
    aw.close()

    async def some_afn():
        pass

    aw = some_afn()
    coro_wrapper_type = type(aw.__await__())
    aw.close()

    return asend_type, athrow_type, coro_wrapper_type


# The types of the internal awaitables used in native async generator
# asend/athrow calls, and the coroutine_wrapper type returned by
# <coroutine object>.__await__()
(
    AsyncGeneratorASend, AsyncGeneratorAThrow, CoroutineWrapperType
) = _get_native_awaitable_types()


if sys.version_info >= (3, 7):
    # GCMBase: the common base type of context manager objects returned by
    # functions decorated with either @contextlib.contextmanager or
    # @contextlib.asynccontextmanager
    from contextlib import _GeneratorContextManagerBase as GCMBase  # type: ignore
    from contextlib import AsyncExitStack
else:
    from contextlib import _GeneratorContextManager as GCMBase

    # async_exit_stack.AsyncExitStack is an alias for
    # contextlib.AsyncExitStack on Pythons that have the latter, so
    # there's no need to consider both separately.
    AsyncExitStack = try_import("async_exit_stack", "AsyncExitStack")


def _determine_skip_code_objects() -> Iterator[types.CodeType]:
    yield Traceback.of.__func__.__code__
    yield Traceback.since.__func__.__code__
    yield Traceback.until.__func__.__code__
    yield Traceback._make.__func__.__code__

    # greenback_shim has special-case handling, but that handling gets skipped
    # if the coroutine is currently executing, and we still want to elide the
    # internal frames in that case.
    yield _greenback_shim.__code__

    # outcome.send() is quite noisy in tracebacks when using
    # greenback, and it's hard to think of any other case where you'd
    # care about it either -- if you see 'something.send(coro)' in one frame
    # and you're inside coro in the next, it's pretty obvious what's going on.
    try:
        from outcome import Value, Error
    except ImportError:
        pass
    else:
        yield Value.send.__code__
        yield Error.send.__code__

    if AsyncGeneratorBackport is not NotPresent:
        import async_generator

        @async_generator.async_generator
        async def noop_agen():
            pass  # pragma: no cover

        # Skip internal frames corresponding to asend() and athrow()
        # coroutines for the @async_generator backport. The native
        # versions are written in C, so won't show up regardless.
        agen_iter = noop_agen()
        asend_coro = agen_iter.asend(None)
        yield asend_coro.cr_code
        asend_coro.close()

        # Skip frames from async_generator.yield_() too.
        yield async_generator._impl.yield_.__code__
        yield async_generator._impl._yield_.__code__

    if TrioNurseryManager is not NotPresent:
        import trio

        # Skip frames corresponding to common lowest-level Trio traps,
        # so that the traceback ends where someone says 'await
        # wait_task_rescheduled(...)' or similar.  The guts are not
        # interesting and they otherwise show up *everywhere*.
        yield trio.lowlevel.cancel_shielded_checkpoint.__code__
        yield trio.lowlevel.wait_task_rescheduled.__code__
        yield trio._core._traps._async_yield.__code__


try:
    SKIP_CODE_OBJECTS.update(_determine_skip_code_objects())
except Exception as exc:  # pragma: no cover
    warnings.warn(
        "Couldn't initialize code objects to skip. Tracebacks may have some "
        "extraneous frames. Please file an issue at "
        "https://github.com/oremanj/asynctb/issues/new",
        RuntimeWarning,
    )


def frame_from_owner(owner: Any) -> Optional[types.FrameType]:
    """Return the outermost frame object associated with *owner*,
    a coroutine or generator iterator. Returns None if *owner*
    has already completed execution. Raises an exception if it's
    not a coroutine or generator iterator.
    """
    if isinstance(owner, AsyncGeneratorBackport):
        owner = owner._coroutine
    if isinstance(owner, types.CoroutineType):
        return owner.cr_frame
    if isinstance(owner, types.GeneratorType):
        return owner.gi_frame
    if isinstance(owner, types.AsyncGeneratorType):
        return owner.ag_frame
    raise RuntimeError(
        "Couldn't determine the frame associated with {}.{} {!r}".format(
            getattr(type(owner), "__module__", "??"),
            type(owner).__qualname__,
            owner,
        ),
    )


def next_from_owner(owner: Any) -> Any:
    """Given *owner*, a coroutine or generator iterator, return the other
    coroutine/generator/iterator/awaitable that *owner* is awaiting or
    yielding-from.
    """
    if isinstance(owner, AsyncGeneratorBackport):
        owner = owner._coroutine
    if isinstance(owner, types.CoroutineType):
        if owner.cr_running:
            return RUNNING
        return owner.cr_await
    elif isinstance(owner, types.GeneratorType):
        if owner.gi_running:
            return RUNNING
        return owner.gi_yieldfrom
    elif isinstance(owner, types.AsyncGeneratorType):
        # On 3.8+, ag_running is true even if the generator
        # is blocked on an event loop trap. Those will
        # have ag_await != None (because the only way to
        # block on an event loop trap is to await something),
        # and we want to treat them as suspended for
        # traceback extraction purposes.
        if owner.ag_running and owner.ag_await is None:
            return RUNNING
        return owner.ag_await
    else:
        return None


def unwrap_owner(owner: Any) -> Any:
    """Return the coroutine or generator iterator underlying
    the awaitable/iterator *owner*. Currently this can look
    inside async generator asend/athrow and coroutine_wrapper.
    """
    while True:
        if isinstance(owner, AsyncGeneratorBackportNextIter):
            owner = owner._it
            continue

        if isinstance(owner, (AsyncGeneratorASend, AsyncGeneratorAThrow)):
            # native async generator awaitable, which holds a
            # reference to its agen but doesn't expose it
            for referent in gc.get_referents(owner):
                if hasattr(referent, "ag_frame"):
                    owner = referent
                    break
            else:
                raise RuntimeError(
                    f"{owner!r} doesn't refer to anything with an ag_frame attribute"
                )
            continue

        if isinstance(owner, CoroutineWrapperType):
            # these refer to only one other object, the underlying coroutine
            for referent in gc.get_referents(owner):
                if hasattr(referent, "cr_frame"):
                    owner = referent
                    break
            else:
                raise RuntimeError(
                    f"{owner!r} doesn't refer to anything with a cr_frame attribute"
                )
            continue

        return owner


def crawl_context(
    owner: object,
    frame: types.FrameType,
    context: ContextInfo,
    override_line: Optional[str] = None,
) -> Iterator[FrameInfo]:
    """Yield a series of FrameInfos for the context manager described in *context*,
    which is a context manager active in *frame* which is associated with the
    awaitable/generator *owner* (if any).

    The first yielded FrameInfo introduces the context manager, by referring to the
    line in *frame* at which the 'with' or 'async with' block begins; its
    context_name and context_manager attributes allow inspection of the context
    manager and the name of the variable that stores its result.
    For example, in 'with trio.fail_after(1) as scope:', context_name is 'scope'
    and context_manager is a GeneratorContextManager object for fail_after().
    If *override_line* is specified, this first yielded FrameInfo will use
    that text rather than pulling the actual source line. This is used when
    representing ExitStack entries, since there's no way to determine the
    actual source line at which each ExitStack entry was pushed.

    If the context manager is generator-based, successive FrameInfos describe
    the environment in the suspended @contextmanager/@asynccontextmanager
    function, potentially including *its* context managers (recursively).

    If the context manager is an ExitStack or AsyncExitStack, successive
    FrameInfos describe each callback or context manager that remains on the stack
    to be run or exited when the ExitStack or AsyncExitStack is closed or exited.
    If these are generator-based context managers or other ExitStacks, their
    status is yielded recursively.
    """

    manager = context.manager
    if isinstance(manager, GreenbackAsyncContext):
        manager = manager._cm
    if isinstance(manager, TrioNurseryManager):
        manager = manager._nursery

    lineno = context.start_line or 0
    yield FrameInfo(
        owner=owner,
        frame=frame,
        lineno=lineno,
        context_manager=manager,
        context_name=context.varname,
        override_line=override_line,
    )

    if isinstance(manager, (AsyncExitStack, contextlib.ExitStack)):
        # ExitStack pops each callback right before running it, so if
        # it's exiting we'll crawl the still-pending callbacks here
        # and the running callback as we continue the top-level traceback.
        yield from crawl_exit_stack(manager, context.varname or "_", frame, lineno)
    elif not context.is_exiting:
        # Don't descend into @contextmanager frames if the context manager
        # is currently exiting, since we'll see them later in the traceback
        # anyway
        if isinstance(manager, GCMBase):
            yield from frames_from_producer(
                iterate_suspended(manager.gen, with_context_info=True)
            )
        elif isinstance(manager, AGCMBackport):
            yield from frames_from_producer(
                iterate_suspended(manager._agen, with_context_info=True)
            )


def format_funcname(func: object) -> str:
    try:
        if isinstance(func, types.MethodType):
            return f"{func.__self__!r}.{func.__name__}"
        else:
            return f"{func.__module__}.{func.__qualname__}"  # type: ignore
    except AttributeError:
        return repr(func)


def format_funcargs(args: Sequence[Any], kw: Mapping[str, Any]) -> str:
    argdescs = [repr(arg) for arg in args]
    kwdescs = [f"{k}={v!r}" for k, v in kw.items()]
    return argdescs + kwdescs


def format_funcall(func: object, args: Sequence[Any], kw: Mapping[str, Any]) -> str:
    return "{}({})".format(format_funcname(func), ", ".join(format_funcargs(args, kw)))


def crawl_exit_stack(
    stack: object, stackname: str, frame: types.FrameType, lineno: int
) -> Iterator[FrameInfo]:
    """Yield FrameInfos describing the individual entries in the ExitStack or
    AsyncExitStack described by *context*, which is active in *frame* and was
    entered at line *lineno* in that frame.

    The result is the concatenated output of crawl_context() for each
    ExitStack callback.
    """

    # List of (is_sync, callback) tuples, from outermost to innermost, where
    # each callback takes parameters following the signature of a __exit__ method
    callbacks: List[Tuple[bool, Callable[..., Any]]]

    raw_callbacks = stack._exit_callbacks  # type: ignore
    if sys.version_info >= (3, 7) or isinstance(stack, AsyncExitStack):
        callbacks = list(raw_callbacks)
    else:
        # Before 3.7, the native ExitStack didn't support async, so its callbacks
        # list didn't have the is_sync information.
        callbacks = [(True, cb) for cb in raw_callbacks]

    for idx, (is_sync, callback) in enumerate(callbacks):
        tag = ""
        manager = None
        if hasattr(callback, "__self__"):
            manager = callback.__self__  # type: ignore
            if (
                # 3.6 used a wrapper function with a __self__ attribute
                # for actual __exit__ invocations. Later versions use a method.
                not isinstance(callback, types.MethodType)
                or callback.__func__.__name__ in ("__exit__", "__aexit__")
            ):
                # stack.enter_context(some_cm) or stack.push(some_cm)
                tag = "" if is_sync else "await "
                method = "enter_context" if is_sync else "enter_async_context"
                if isinstance(manager, GCMBase):
                    if hasattr(manager, "func"):
                        arg = format_funcall(manager.func, manager.args, manager.kwds)
                    else:
                        # 3.7+ delete the func/args/etc attrs once entered
                        arg = f"{manager.gen.__qualname__}(...)"
                elif isinstance(manager, AGCMBackport):
                    arg = f"{manager._func_name}(...)"
                else:
                    arg = repr(manager)
            else:
                # stack.push(something.exit_ish_method)
                method = "push" if is_sync else "push_async_exit"
                arg = format_funcname(callback)
        elif (
            hasattr(callback, "__wrapped__")
            and getattr(callback, "__name__", None) == "_exit_wrapper"
            and isinstance(callback, types.FunctionType)
            and set(callback.__code__.co_freevars) >= {"args", "kwds"}
        ):
            # Normal callback wrapped in internal _exit_wrapper function
            # to adapt it to the __exit__ protocol
            args_idx = callback.__code__.co_freevars.index("args")
            kwds_idx = callback.__code__.co_freevars.index("kwds")
            assert callback.__closure__ is not None
            arg = ", ".join(
                [
                    format_funcname(callback.__wrapped__),
                    *format_funcargs(
                        callback.__closure__[args_idx].cell_contents,
                        callback.__closure__[kwds_idx].cell_contents,
                    ),
                ],
            )
            method = "callback" if is_sync else "push_async_callback"
        else:
            # stack.push(exit_ish_function)
            method = "push" if is_sync else "push_async_exit"
            arg = format_funcname(callback)

        yield from crawl_context(
            owner=None,
            frame=frame,
            context=ContextInfo(
                is_async=not is_sync,
                manager=manager,
                varname=f"{stackname}[{idx}]",
                start_line=lineno,
            ),
            override_line=f"# {tag}{stackname}.{method}({arg})",
        )
