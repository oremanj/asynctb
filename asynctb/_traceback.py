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
    AsyncGenerator,
    Callable,
    Coroutine,
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
from ._registry import (
    unwrap_awaitable,
    unwrap_context_manager,
    CodeHandling,
    HANDLING_FOR_CODE,
)

try:
    from greenlet import greenlet as GreenletType, getcurrent as greenlet_getcurrent
except ImportError:

    class GreenletType:
        parent: Optional["GreenletType"] = None
        gr_frame: Optional[types.FrameType]

    def greenlet_getcurrent() -> GreenletType:
        return GreenletType()


GeneratorLike = Union[
    Generator[Any, Any, Any], AsyncGenerator[Any, Any], Coroutine[Any, Any, Any]
]


@attr.s(auto_attribs=True, slots=True, eq=True, frozen=True)
class FrameInfo:
    """Information about one frame in a `Traceback`."""

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
# and iterate_running, each of which yields FrameInfos. Mode switching
# is implemented by having one yield from the other.


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
    def of(
        cls,
        stackref: Union[GeneratorLike, GreenletType],
        *,
        with_context_info: bool = True,
    ) -> "Traceback":
        """Return a traceback reflecting the current stack of *stackref*,
        which must be a coroutine object, greenlet object, or (sync or
        async) generator iterator. It may be either running or
        suspended. If it's running in a different thread, we'll still
        attempt to extract a traceback, but might not be able to. In
        particular, greenlets running in other threads can never be
        tracebacked, and generator-like objects might fail to extract
        a traceback depending on when thread switches occur.

        Produce an :ref:`enhanced traceback <enhanced-tb>` if *with_context_info*
        is True (the default), or a basic traceback if *with_context_info* is False.
        """
        if isinstance(stackref, GreenletType):
            inner_frame = stackref.gr_frame
            outer_frame = None
            if inner_frame is None:
                if not stackref:  # dead or not started
                    return Traceback(frames=())
                # otherwise a None frame means it's running
                if stackref is not greenlet_getcurrent():
                    return Traceback(
                        frames=(),
                        error=RuntimeError(
                            "Traceback.of(greenlet) can't handle a greenlet running "
                            "in another thread"
                        ),
                    )
                # since it's running in this thread, its stack is our own
                inner_frame = sys._getframe(1)
                if stackref.parent is not None:
                    outer_frame = stackref.parent.gr_frame
            producer = iterate_running(
                outer_frame, inner_frame, with_context_info, stackref
            )
        else:
            producer = iterate_suspended(stackref, with_context_info)
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
    def _make(cls, producer: Iterator[FrameInfo]) -> "Traceback":
        frames: List[FrameInfo] = []
        error: Optional[Exception] = None
        try:
            frames.extend(producer)
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


# Singleton object returned by next_from_genlike() to indicate
# that a generator or coroutine is currently running and that we therefore
# need to switch to iterate_running().
RUNNING = object()


_default_handling = CodeHandling()


def iterate_suspended(
    genlike: GeneratorLike, with_context_info: bool, switch_count: int = 0
) -> Iterator[FrameInfo]:
    """Yield information about a series of frames representing the current
    stack of the suspended generator iterator, async generator iterator,
    or coroutine object *genlike*.

    If *with_context_info* is True, yield additional frames
    representing context managers, to produce an enhanced traceback.

    *switch_count* tracks the number of times we have switched modes
    from inward to outward or vice versa, as a guard against infinite
    loops.
    """

    genlike = unwrap_awaitable(genlike)

    while genlike is not None:
        this_frame = frame_from_genlike(genlike)
        if this_frame is None:
            # Exhausted generator/coroutine has no traceback
            break

        next_genlike = next_from_genlike(genlike)
        if next_genlike is RUNNING:
            # If it's running, cr_await/ag_await/gi_yieldfrom aren't available,
            # so we need an alternate approach.
            yield from iterate_running(
                outer_frame=this_frame,
                inner_frame=None,
                with_context_info=with_context_info,
                parent=genlike,
                switch_count=switch_count + 1,
            )
            break

        # Otherwise, the generator/coroutine is suspended. Yield info
        # about this_frame (and its context managers), then continue
        # down the stack to next_genlike (the thing this genlike is
        # awaiting/yielding-from).

        try:
            next_frame = frame_from_genlike(next_genlike)
        except Exception:
            next_frame = None
        keep_going = yield from handle_frame(
            this_frame=this_frame,
            next_frame=next_frame,
            with_context_info=with_context_info,
            is_terminal=next_genlike is None,
            switch_count=switch_count,
        )
        if not keep_going:
            break

        genlike = unwrap_awaitable(next_genlike)


def iterate_running(
    outer_frame: Optional[types.FrameType],
    inner_frame: Optional[types.FrameType],
    with_context_info: bool,
    parent: Union[GeneratorLike, GreenletType, None] = None,
    switch_count: int = 0,
) -> Iterator[FrameInfo]:
    """Yield information about a series of frames linked via f_back
    members from inner_frame out to outer_frame. (They will be yielded
    in the opposite order, outer_frame first.)

    If outer_frame is None, continue until the outermost frame on inner_frame's
    stack is reached. If inner_frame is None, find a currently-executing frame
    on any thread that has outer_frame on its stack. If both are None, produce
    frames for the current thread's entire stack.

    If these frames represent the stack of a running generator or
    coroutine or greenlet, pass *parent* as that generator or
    coroutine or greenlet. This is used to switch back to
    iterate_suspended() if a generator or coroutine stops running (in
    another thread) while we're looking, and to

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

    # This function tries a bunch of different ways to fill out this 'frames'
    # list, as a list of the frames we want to traceback from outermost
    # to innermost.
    frames: List[types.FrameType] = []

    def get_true_caller() -> Optional[types.FrameType]:
        # Return the frame that called into the traceback-producing machinery.
        # That will be at least 3 frames up (iterate_running, Traceback._make,
        # Traceback.of/since/until) and might be more if we've done some mode
        # switching.
        caller = sys._getframe(3)
        while caller is not None and caller.f_globals is globals():
            caller = caller.f_back
        return caller

    if sys.implementation.name == "cpython" and greenlet_getcurrent().parent:
        # On CPython, each greenlet is its own universe traceback-wise:
        # if you're inside a non-main greenlet and you follow f_back links
        # outward from your current frame, you'll only find the outermost
        # frame in this greenlet, not in this thread. We augment that by
        # following the greenlet parent link (the same path an exception
        # would take) when we reach an f_back of None.
        #
        # This only works on the current thread, since there's no way
        # to call greenlet.getcurrent() for another thread. (There's a key
        # in the thread state dictionary, which we can't safely access because
        # any other thread state can disappear out from under us whenever we
        # yield the GIL, which we can't prevent from happening. Resolving
        # this would require an extension module.)
        #
        # PyPy uses a more sensible scheme where the f_back links in the
        # current callstack always work, so it doesn't need this trick.
        this_thread_frames: List[types.FrameType] = []
        greenlet = greenlet_getcurrent()
        current: Optional[types.FrameType] = get_true_caller()
        while greenlet is not None:
            while current is not None:
                this_thread_frames.append(current)
                current = current.f_back
            greenlet = greenlet.parent
            if greenlet is not None:
                current = greenlet.gr_frame
        try:
            from_idx: Optional[int]
            if inner_frame is None or this_thread_frames[0] is inner_frame:
                from_idx = None
            else:
                from_idx = this_thread_frames.index(inner_frame) - 1
            to_idx = (
                len(this_thread_frames)
                if outer_frame is None
                else this_thread_frames.index(outer_frame)
            )
        except ValueError:
            pass
        else:
            frames = this_thread_frames[to_idx:from_idx:-1]

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

    if not frames:
        frames = try_from(inner_frame or get_true_caller())

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
            if (
                parent is not None
                and not isinstance(parent, GreenletType)
                and switch_count < 50
            ):
                if next_from_genlike(parent) is not RUNNING:
                    # It was running when we started looking, but now
                    # is suspended again. Switch to the logic for suspended
                    # frames. (There's no specific limit to how many times this
                    # can occur; our only argument against an infinite
                    # recursion here is probabilistic, but it's a pretty good
                    # one.)
                    yield from iterate_suspended(
                        genlike=parent,
                        with_context_info=with_context_info,
                        switch_count=switch_count + 1,
                    )
                else:
                    # Possibly it became suspended before we extracted
                    # _current_frames() and then started running again
                    # before we called frame_and_next(). Or possibly
                    # we've become confused and found a frame that
                    # isn't running at all, e.g., because it's part of
                    # a suspended greenlet. Assume the former until we
                    # hit our mode-switching limit, at which point
                    # give up so as not to infinite-loop in the latter case.
                    yield from iterate_running(
                        outer_frame=outer_frame,
                        inner_frame=None,
                        with_context_info=with_context_info,
                        parent=parent,
                        switch_count=switch_count + 1,
                    )
                return

    if not frames:
        # If outer_frame is None, try_from() always returns a non-empty
        # list, so we can only get here if outer_frame is not None.
        assert outer_frame is not None
        yield from one_frame_traceback(outer_frame, None, False)
        raise RuntimeError(
            "Couldn't find where the above frame is running, so "
            "can't continue traceback"
        )

    for this_frame, next_frame in zip(frames, frames[1:] + [None]):
        keep_going = yield from handle_frame(
            this_frame=this_frame,
            next_frame=next_frame,
            with_context_info=with_context_info,
            is_terminal=next_frame is None,
            switch_count=switch_count,
        )
        if not keep_going:
            break


def handle_frame(
    this_frame: types.FrameType,
    next_frame: Optional[types.FrameType],
    with_context_info: bool,
    is_terminal: bool,
    switch_count: int,
) -> Generator[FrameInfo, None, bool]:
    """ XXX document """

    handling = HANDLING_FOR_CODE.get(this_frame.f_code, _default_handling)

    if not handling.skip_frame:
        yield from one_frame_traceback(this_frame, next_frame, with_context_info)

    if handling.skip_callees:
        return False

    if handling.get_target is not None:
        try:
            target = handling.get_target(this_frame, is_terminal)
            if isinstance(target, GreenletType):
                # This frame is the runner for a greenlet.
                # Grab gr_frame atomically in case it starts running in
                # another thread.
                gr_frame = target.gr_frame
                if gr_frame is not None:
                    yield from iterate_running(
                        outer_frame=None,
                        inner_frame=gr_frame,
                        with_context_info=with_context_info,
                        parent=target,
                        switch_count=switch_count + 1,
                    )
                elif target:
                    # not dead, so lack of gr_frame means running, but it's
                    # not running in this thread because it was reached from
                    # a suspended coroutine/etc -- so it must be running in
                    # another thread
                    raise RuntimeError(
                        "Can't trace into a greenlet running in another thread"
                    )
            elif target is not None:
                # This frame is the runner for a genlike
                yield from iterate_suspended(
                    genlike=target,
                    with_context_info=with_context_info,
                    switch_count=switch_count + 1,
                )
        except Exception:
            if handling.skip_frame:
                yield from one_frame_traceback(this_frame, None, False)
            raise

    return True


def one_frame_traceback(
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
    if with_context_info:
        for context in contexts_active_in_frame(this_frame):
            if context.is_exiting and next_frame is not None:
                # Infer the context manager being exited based on the
                # self argument to its __exit__ or __aexit__ method in
                # the next frame
                args = inspect.getargvalues(next_frame)
                if args.args:
                    context = attr.evolve(context, manager=args.locals[args.args[0]])
            yield from crawl_context(this_frame, context)
    yield FrameInfo(this_frame)


if sys.version_info >= (3, 7):
    # GCMBase: the common base type of context manager objects returned by
    # functions decorated with either @contextlib.contextmanager or
    # @contextlib.asynccontextmanager
    from contextlib import _GeneratorContextManagerBase as GCMBase  # type: ignore
    from contextlib import AsyncExitStack

    # async_exit_stack.AsyncExitStack is an alias for
    # contextlib.AsyncExitStack on Pythons that have the latter, so
    # there's no need to consider both separately.
else:
    from contextlib import _GeneratorContextManager as GCMBase

    try:
        from async_exit_stack import AsyncExitStack
    except ImportError:

        class AsyncExitStack:
            pass


# The type of the context manager object returned by a function that is
# decorated with @async_generator.asynccontextmanager
try:
    from async_generator._util import _AsyncGeneratorContextManager as AGCMBackport
except ImportError:

    class AGCMBackport:
        pass


def frame_from_genlike(genlike: GeneratorLike) -> Optional[types.FrameType]:
    """Return the outermost frame object associated with *genlike*,
    a coroutine or generator iterator. Returns None if *genlike*
    has already completed execution. Raises an exception if it's
    not a coroutine or generator iterator.
    """
    if isinstance(genlike, types.CoroutineType):
        return genlike.cr_frame
    if isinstance(genlike, types.GeneratorType):
        return genlike.gi_frame
    if isinstance(genlike, types.AsyncGeneratorType):
        return genlike.ag_frame
    raise RuntimeError(
        "Couldn't determine the frame associated with {}.{} {!r}".format(
            getattr(type(genlike), "__module__", "??"),
            type(genlike).__qualname__,
            genlike,
        ),
    )


def next_from_genlike(genlike: GeneratorLike) -> Any:
    """Given *genlike*, a coroutine or generator iterator, return the other
    coroutine/generator/iterator/awaitable that *genlike* is awaiting or
    yielding-from.
    """
    if isinstance(genlike, types.CoroutineType):
        if genlike.cr_running:
            return RUNNING
        return genlike.cr_await
    elif isinstance(genlike, types.GeneratorType):
        if genlike.gi_running:
            return RUNNING
        return genlike.gi_yieldfrom
    elif isinstance(genlike, types.AsyncGeneratorType):
        # On 3.8+, ag_running is true even if the generator
        # is blocked on an event loop trap. Those will
        # have ag_await != None (because the only way to
        # block on an event loop trap is to await something),
        # and we want to treat them as suspended for
        # traceback extraction purposes.
        if genlike.ag_running and genlike.ag_await is None:
            return RUNNING
        return genlike.ag_await
    else:
        return None


def crawl_context(
    frame: types.FrameType, context: ContextInfo, override_line: Optional[str] = None,
) -> Iterator[FrameInfo]:
    """Yield a series of FrameInfos for the context manager described in *context*,
    which is a context manager active in *frame*.

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

    manager = unwrap_context_manager(context.manager)
    lineno = context.start_line or 0
    yield FrameInfo(
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
            yield from iterate_suspended(manager.gen, with_context_info=True)
        elif isinstance(manager, AGCMBackport):
            yield from iterate_suspended(manager._agen, with_context_info=True)


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
            frame=frame,
            context=ContextInfo(
                is_async=not is_sync,
                manager=manager,
                varname=f"{stackname}[{idx}]",
                start_line=lineno,
            ),
            override_line=f"# {tag}{stackname}.{method}({arg})",
        )
