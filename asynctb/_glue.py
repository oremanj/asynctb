import gc
import sys
import threading
import traceback
import types
import warnings
from typing import Any, AsyncGenerator, Optional
from . import _registry


def glue_native() -> None:
    async def some_asyncgen() -> AsyncGenerator[None, None]:
        yield  # pragma: no cover

    # Get the types of the internal awaitables used in native async
    # generator asend/athrow calls
    asend_type = type(some_asyncgen().asend(None))
    athrow_type = type(some_asyncgen().athrow(ValueError))

    async def some_afn() -> None:
        pass  # pragma: no cover

    # Get the coroutine_wrapper type returned by <coroutine object>.__await__()
    coro = some_afn()
    coro_wrapper_type = type(coro.__await__())
    coro.close()

    @_registry.register_unwrap_awaitable(asend_type)
    @_registry.register_unwrap_awaitable(athrow_type)
    def unwrap_async_generator_asend_athrow(aw: Any) -> Any:
        # native async generator awaitable, which holds a
        # reference to its agen but doesn't expose it
        for referent in gc.get_referents(aw):
            if hasattr(referent, "ag_frame"):  # pragma: no branch
                return referent
        raise RuntimeError(
            f"{aw!r} doesn't refer to anything with an ag_frame attribute"
        )

    @_registry.register_unwrap_awaitable(coro_wrapper_type)
    def unwrap_coroutine_wrapper(aw: Any) -> Any:
        # these refer to only one other object, the underlying coroutine
        for referent in gc.get_referents(aw):
            if hasattr(referent, "cr_frame"):  # pragma: no branch
                return referent
        raise RuntimeError(
            f"{aw!r} doesn't refer to anything with a cr_frame attribute"
        )

    # Don't show thread bootstrap gunk at the base of thread stacks
    _registry.customize(threading.Thread.run, skip_frame=True)
    for name in ("_bootstrap", "_bootstrap_inner"):
        if hasattr(threading.Thread, name):  # pragma: no branch
            _registry.customize(getattr(threading.Thread, name), skip_frame=True)


def glue_async_generator() -> None:
    try:
        import async_generator
    except ImportError:
        return
    _registry.customize(async_generator.yield_, skip_frame=True, skip_callees=True)

    @async_generator.async_generator
    async def noop_agen() -> None:
        pass  # pragma: no cover

    # Skip internal frames corresponding to asend() and athrow()
    # coroutines for the @async_generator backport. The native
    # versions are written in C, so won't show up regardless.
    agen_iter = noop_agen()
    asend_coro = agen_iter.asend(None)
    _registry.customize(asend_coro.cr_code, skip_frame=True)

    @_registry.register_unwrap_awaitable(type(agen_iter))
    def unwrap_async_generator_backport(agen: Any) -> Any:
        return agen._coroutine

    from async_generator._impl import ANextIter  # type: ignore

    @_registry.register_unwrap_awaitable(ANextIter)
    def unwrap_async_generator_backport_next_iter(aw: Any) -> Any:
        return aw._it

    asend_coro.close()


def glue_outcome() -> None:
    try:
        import outcome
    except ImportError:
        return

    # Don't give simple outcome functions their own traceback frame,
    # as they tend to add clutter without adding meaning. If you see
    # 'something.send(coro)' in one frame and you're inside coro in
    # the next, it's pretty obvious what's going on.
    _registry.customize(outcome.Value.send, skip_frame=True)
    _registry.customize(outcome.Error.send, skip_frame=True)
    _registry.customize(outcome.capture, skip_frame=True)
    _registry.customize(outcome.acapture, skip_frame=True)


def glue_trio() -> None:
    try:
        import trio
    except ImportError:
        return

    try:
        lowlevel = trio.lowlevel
    except ImportError:  # pragma: no cover
        # Support older Trio versions
        lowlevel = trio.hazmat  # type: ignore

    # Skip frames corresponding to common lowest-level Trio traps,
    # so that the traceback ends where someone says 'await
    # wait_task_rescheduled(...)' or similar.  The guts are not
    # interesting and they otherwise show up *everywhere*.
    for trap in (
        "cancel_shielded_checkpoint",
        "wait_task_rescheduled",
        "temporarily_detach_coroutine_object",
        "permanently_detach_coroutine_object",
    ):
        if hasattr(lowlevel, trap):  # pragma: no branch
            _registry.customize(
                getattr(lowlevel, trap), skip_frame=True, skip_callees=True
            )

    @_registry.register_unwrap_context_manager(type(trio.open_nursery()))
    def unwrap_nursery_manager(cm: Any) -> Any:
        return cm._nursery


def glue_greenlet_pypy() -> None:
    try:
        import greenlet  # type: ignore
    except ImportError:
        return
    if sys.implementation.name != "pypy":
        return

    # pypy greenlet is written in Python on top of the pypy-specific
    # module _continuation.  Hide traceback frames for its internals
    # for better consistency with CPython.
    _registry.customize(greenlet.greenlet.switch, skip_frame=True)
    _registry.customize(greenlet.greenlet._greenlet__switch, skip_frame=True)
    _registry.customize(greenlet._greenlet_start, skip_frame=True)
    _registry.customize(greenlet._greenlet_throw, skip_frame=True)


def glue_greenback() -> None:
    try:
        import greenback
    except ImportError:
        return

    _registry.customize(greenback._impl._greenback_shim, skip_frame=True)
    _registry.customize(greenback.await_, skip_frame=True)

    @_registry.register_get_target(greenback._impl._greenback_shim)
    def unwrap_greenback_shim(
        frame: types.FrameType, next_frame: Optional[types.FrameType]
    ) -> Any:
        if next_frame is not None:
            # Greenback shim that's not suspended at its yield point requires
            # no special handling -- just keep tracebacking.
            return None

        # Greenback shim. Is the child coroutine suspended in an await_()?
        child_greenlet = frame.f_locals.get("child_greenlet")
        orig_coro = frame.f_locals.get("orig_coro")
        gr_frame = getattr(child_greenlet, "gr_frame", None)
        if gr_frame is not None:
            # Yep; switch to walking the greenlet stack, since orig_coro
            # will look "running" but it's not on any thread's stack.
            return child_greenlet
        elif orig_coro is not None:
            # No greenlet, so child is suspended at a regular await.
            # Continue the traceback by walking the coroutine's frames.
            return orig_coro
        else:  # pragma: no cover
            raise RuntimeError(
                "Can't identify what's going on with the greenback shim in this "
                "frame"
            )

    @_registry.register_get_target(greenback.await_)
    def unwrap_greenback_await(
        frame: types.FrameType, next_frame: Optional[types.FrameType]
    ) -> Any:
        if next_frame is not None and next_frame.f_code.co_name != "switch":
            # await_ that's not suspended at greenlet.switch() requires
            # no special handling
            return None

        # Greenback-mediated await of async function from sync land.
        # If we have a coroutine to descend into, do so;
        # otherwise the traceback will unhelpfully stop here.
        # This works whether the coro is running or not.
        # (The only way to get coro=None is if we're taking
        # the traceback in the early part of await_() before
        # coro is assigned.)
        return frame.f_locals.get("coro")

    @_registry.register_unwrap_context_manager(greenback.async_context)
    def unwrap_greenback_async_context(cm: Any) -> Any:
        return cm._cm


INSTALLED: bool = False


def ensure_installed() -> None:
    global INSTALLED
    if INSTALLED:
        return

    globs = globals()
    for name in list(globs):
        # This function might be called concurrently or reentrantly
        # (imagine a signal handler that prints a backtrace).
        # Ensure that each glue_xyz() hook is only called once,
        # by removing it from the global namespace before we call it.
        # Different hooks can safely execute concurrently because
        # they modify different items in _registry.HANDLING_FOR_CODE,
        # and primitive operations on IdentityDict are thread-safe
        # since they're implemented in terms of primitive operations
        # on an underlying native dict.
        if name.startswith("glue_"):
            try:
                fn = globs.pop(name)
            except KeyError:
                continue
            try:
                fn()
            except Exception as exc:
                warnings.warn(
                    "Failed to initialize glue for {}: {}. Some tracebacks may be "
                    "presented less crisply or with missing information.".format(
                        name[5:],
                        "".join(
                            traceback.format_exception_only(type(exc), exc)
                        ).strip(),
                    ),
                    RuntimeWarning,
                )

    INSTALLED = True
