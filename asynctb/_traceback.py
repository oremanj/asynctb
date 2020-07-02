import attr
import contextlib
import gc
import importlib
import math
import sys
import traceback
import types
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from ._frames import ContextInfo, contexts_active_in_frame

__all__ = ("FrameInfo", "get_traceback")


@attr.s(auto_attribs=True, slots=True, eq=True, frozen=True)
class FrameInfo:
    owner: object
    frame: Optional[types.FrameType]
    lineno: Optional[int]
    override_line: Optional[str] = None
    context_manager: object = None
    context_name: Optional[str] = None

    def as_stdlib_summary(self, *, save_locals: bool = False) -> traceback.FrameSummary:
        return traceback.FrameSummary(
            self.frame.f_code.co_filename,
            self.lineno or 0,
            self.frame.f_code.co_name,
            locals=self.frame.f_locals if save_locals else None,
            line=self.override_line,
        )


RUNNING = object()


def get_traceback(owner: object) -> Iterator[FrameInfo]:
    while owner is not None:
        this_frame, next_owner = frame_and_next(owner)
        if this_frame is None:
            break
        if next_owner is RUNNING:
            def try_from(top_frame: types.FrameType) -> List[types.FrameType]:
                frames: List[types.FrameType] = [top_frame]
                while True:
                    prev = frames[-1].f_back
                    if prev is None:
                        return []
                    frames.append(prev)
                    if prev is this_frame:
                        return frames[::-1]

            frames = try_from(sys._getframe(0))
            if not frames:
                # See if it's running on a different thread
                for ident, top_frame in sys._current_frames().items():
                    if ident != threading.get_ident():
                        frames = try_from(top_frame)
                        if frames:
                            break
                else:
                    return

            yield from running_traceback(owner, frames)
            return

        yield from one_frame_traceback(
            owner, this_frame, lambda: frame_and_next(next_owner)[0]
        )
        owner = next_owner


def running_traceback(owner: object, frames: List[types.FrameType]) -> Iterator[FrameInfo]:
    for idx, frame in enumerate(frames):
        def get_next() -> Optional[types.FrameType]:
            try:
                return frames[idx + 1]
            except IndexError:
                return None

        yield from one_frame_traceback(owner, frame, get_next)
        owner = None


def one_frame_traceback(
    owner: object, frame: types.FrameType, get_next: Callable[[], Optional[types.FrameType]]
) -> Iterator[FrameInfo]:
    for context in contexts_active_in_frame(frame):
        if context.is_exiting:
            # Infer the context manager being exited based on the
            # self argument to its __exit__ or __aexit__ method in
            # the next frame
            next_frame = get_next()
            if next_frame is not None:
                args = inspect.getargvalues(next_frame)
                if args.args:
                    context = attr.evolve(context, manager=args.locals[args.args[0]])
        yield from crawl_context(owner, frame, context)
    yield FrameInfo(owner=owner, frame=frame, lineno=frame.f_lineno)


def frame_and_next(owner: Any) -> Tuple[Optional[types.FrameType], Any]:
    """Given an awaitable that is part of a suspended async call
    stack, return a tuple containing its active frame object and
    the other awaitable that it is currently awaiting.

    This supports coroutine objects, generator-based coroutines,
    async generator asend/athrow calls (both native and @async_generator),
    and coroutine wrappers. If the given awaitable isn't any of those,
    it returns (None, None).
    """

    if isinstance(owner, types.CoroutineType):
        if owner.cr_running:
            return owner.cr_frame, RUNNING
        return owner.cr_frame, owner.cr_await
    if isinstance(owner, types.GeneratorType):
        if owner.gi_running:
            return owner.gi_frame, RUNNING
        return owner.gi_frame, owner.gi_yieldfrom
    if isinstance(owner, types.AsyncGeneratorType):
        if owner.ag_running and owner.ag_await is None:
            return owner.ag_frame, RUNNING
        return owner.ag_frame, owner.ag_await
    if isinstance(owner, AsyncGeneratorBackport):
        return frame_and_next(owner._coroutine)

    typename = type(owner).__name__

    if typename == "ANextIter":  # @async_generator awaitable
        return frame_and_next(owner._it)
    if typename in ("async_generator_asend", "async_generator_athrow"):
        # native async generator awaitable, which holds a
        # reference to its agen but doesn't expose it
        for referent in gc.get_referents(owner):
            if hasattr(referent, "ag_frame"):
                return frame_and_next(referent)
    if typename == "coroutine_wrapper":
        # these refer to only one other object, the underlying coroutine
        for referent in gc.get_referents(owner):
            return frame_and_next(referent)
    return None, None


class NotPresent:
    pass


def try_import(module: str, name: str) -> Any:
    try:
        ns = importlib.import_module(module)
    except ImportError:
        return NotPresent
    try:
        return getattr(ns, name)
    except AttributeError:
        return NotPresent


TrioNurseryManager = try_import("trio._core._run", "NurseryManager")
AGCMBackport = try_import("async_generator._util", "_AsyncGeneratorContextManager")
AsyncGeneratorBackport = try_import("async_generator._impl", "AsyncGenerator")

try:
    from contextlib import AsyncExitStack
except ImportError:
    AsyncExitStack = try_import("async_exit_stack", "AsyncExitStack")

try:
    from contextlib import _GeneratorContextManagerBase as GCMBase
except ImportError:
    from contextlib import _GeneratorContextManager as GCMBase


def crawl_context(
    owner: object,
    frame: Optional[types.FrameType],
    context: ContextInfo,
    override_lineno: Optional[int] = None,
    override_line: Optional[str] = None,
) -> Iterator[FrameInfo]:

    manager = context.manager
    if isinstance(manager, TrioNurseryManager):
        manager = manager._nursery

    yield FrameInfo(
        owner=owner,
        frame=frame,
        lineno=override_lineno or context.start_line,
        context_manager=manager,
        context_name=context.varname,
    )

    if isinstance(context.manager, GCMBase):
        yield from get_traceback(context.manager.gen)
    elif isinstance(context.manager, AGCMBackport):
        yield from get_traceback(context.manager._agen)
    elif isinstance(context.manager, (AsyncExitStack, contextlib.ExitStack)):
        yield from crawl_exit_stack(context)


def crawl_exit_stack(context: ContextInfo):
    callbacks: List[Tuple[bool, Callable[..., Any]]]
    if sys.version_info >= (3, 7) or isinstance(context.manager, AsyncExitStack):
        callbacks = list(context.manager._exit_callbacks)
    else:
        callbacks = [(True, cb) for cb in context.manager._exit_callbacks]

    for idx, (is_sync, callback) in enumerate(callbacks):
        tag = ""
        manager = None
        if hasattr(callback, "__self__"):
            manager = callback.__self__
            if (
                # 3.6 used a wrapper function with a __self__ attribute
                # for actual __exit__ invocations. Later versions use a method.
                not isinstance(callback, types.MethodType)
                or callback.__func__.__name__ in ("__exit__", "__aexit__")
            ):
                tag = "" if is_sync else "await "
                method = "enter_context" if is_sync else "enter_async_context"
                if isinstance(manager, GCMBase):
                    arg = format_funcall(manager.func, manager.args, manager.kwds)
                elif isinstance(manager, AGCMBackport):
                    arg = f"{manager._func_name}(...)"
                else:
                    arg = repr(manager)
            else:
                method = "push" if is_sync else "push_async_context"
                arg = format_funcname(callback)
        elif (
            hasattr(callback, "__wrapped__")
            and getattr(callback, "__name__", None) == "_exit_wrapper"
            and isinstance(callback, types.FunctionType)
            and set(callback.__code__.co_freevars) >= {"args", "kwds"}
        ):
            args_idx = callback.__code__.co_freevars.index("args")
            kwds_idx = callback.__code__.co_freevars.index("kwds")
            arg = ", ".join(
                [
                    format_funcname(callback),
                    format_funcargs(
                        callback.__closure__[args_idx].cell_contents,
                        callback.__closure__[kwds_idx].cell_contents,
                    ),
                ],
            )
            method = "callback" if is_sync else "push_async_callback"

        stackname = context.varname or "..."
        yield from crawl_context(
            owner=None,
            frame=None,
            context=ContextInfo(
                is_async=not is_sync,
                manager=manager,
                varname=f"{stackname}[{idx}]",
            ),
            override_line=f"# {tag}{stackname}.{method}({arg})",
        )


def format_funcname(func: object) -> str:
    try:
        if isinstance(func, types.MethodType):
            return f"{func.__self__!r}.{func.__name__}"
        else:
            return f"{func.__module__}.{func.__qualname__}"
    except AttributeError:
        return repr(func)


def format_funcargs(args: Sequence[Any], kw: Mapping[str, Any]) -> str:
    argdescs = [repr(arg) for arg in args]
    kwdescs = [f"{k}={v!r}" for k, v in kw.items()]
    return ", ".join(argdescs + kwdescs)


def format_funcall(func: object, args: Sequence[Any], kw: Mapping[str, Any]) -> str:
    return "{}({})".format(format_funcname(func), format_funcargs(args, kw))
