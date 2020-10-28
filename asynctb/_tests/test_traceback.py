import contextlib
import gc
import re
import sys
import threading
import types
from contextlib import ExitStack, contextmanager
from functools import partial
from typing import List, Callable, Any, cast

import attr
import pytest

from .. import FrameInfo, Traceback, customize, register_get_target


def remove_address_details(line):
    return re.sub(r"\b0x[0-9A-Fa-f]+\b", "(address)", line)


def clean_tb_line(line):
    return remove_address_details(line).partition("  #")[0]


def assert_tb_matches(tb, expected, error=None):
    # smoke test:
    str(tb)
    tb.as_stdlib_summary()
    tb.as_stdlib_summary(capture_locals=True)
    for frame in tb.frames:
        str(frame)

    try:
        if error is None and tb.error is not None:  # pragma: no cover
            raise tb.error

        assert type(tb.error) is type(error)
        assert remove_address_details(str(tb.error)) == remove_address_details(
            str(error)
        )
        assert len(tb) == len(expected)
        for (
            entry,
            (expect_fn, expect_line, expect_ctx_name, expect_ctx_typename),
        ) in zip(tb, expected):
            assert entry.funcname == expect_fn
            assert clean_tb_line(entry.linetext) == expect_line
            assert entry.context_name == expect_ctx_name
            if entry.context_manager is None:
                assert expect_ctx_typename is None
            else:
                assert type(entry.context_manager).__name__ == expect_ctx_typename
    except Exception:  # pragma: no cover
        print_assert_matches("tb")
        raise


def print_assert_matches(get_tb):  # pragma: no cover
    parent = sys._getframe(1)
    get_tb_code = compile(get_tb, "<eval>", "eval")
    tb = eval(get_tb_code, parent.f_globals, parent.f_locals)
    print("---")
    print(str(tb).rstrip())
    print("---")
    print("    assert_tb_matches(")
    print("        " + get_tb + ",")
    print("        [")
    for entry in tb:
        if entry.frame.f_code is get_tb_code:
            funcname = parent.f_code.co_name
            linetext = get_tb + ","
        else:
            funcname = entry.funcname
            linetext = clean_tb_line(entry.linetext)
        typename = type(entry.context_manager).__name__
        if typename == "NoneType":
            typename = None
        record = (funcname, linetext, entry.context_name, typename)
        print("            " + repr(record) + ",")
    print("        ],")
    if tb.error:
        print(f"        error={remove_address_details(repr(tb.error))},")
    print("    )")


def no_abort(_):  # pragma: no cover
    import trio

    return trio.lowlevel.Abort.FAILED


@contextmanager
def null_context():
    yield


@contextmanager
def outer_context():
    with inner_context() as inner:  # noqa: F841
        yield


def exit_cb(*exc):
    pass


def other_cb(*a, **kw):
    pass


@contextmanager
def inner_context():
    stack = ExitStack()
    with stack:
        stack.enter_context(null_context())
        stack.push(exit_cb)
        stack.callback(other_cb, 10, "hi", answer=42)
        yield


@types.coroutine
def async_yield(value):
    return (yield value)


null_mgr = null_context()
with null_mgr:
    if hasattr(null_mgr, "func"):
        null_context_repr = "asynctb._tests.test_traceback.null_context()"
    else:
        null_context_repr = "null_context(...)"
del null_mgr


# There's some logic in the traceback extraction of running code that
# behaves differently when it's run in a non-main greenlet on CPython,
# because we have to stitch together the traceback portions from
# different greenlets. To exercise it, we'll run some tests in a
# non-main greenlet as well as at top level.
try:
    import greenlet  # type: ignore
except ImportError:

    def try_in_other_greenlet_too(fn):
        return fn


else:

    def try_in_other_greenlet_too(fn):
        def try_both():
            fn()
            greenlet.greenlet(fn).switch()

        return try_both


def frames_from_inner_context(caller):
    return [
        (
            caller,
            "with inner_context() as inner:",
            "inner",
            "_GeneratorContextManager",
        ),
        ("inner_context", "with stack:", "stack", "ExitStack"),
        (
            "inner_context",
            f"# stack.enter_context({null_context_repr})",
            "stack[0]",
            "_GeneratorContextManager",
        ),
        ("null_context", "yield", None, None),
        (
            "inner_context",
            "# stack.push(asynctb._tests.test_traceback.exit_cb)",
            "stack[1]",
            None,
        ),
        (
            "inner_context",
            "# stack.callback(asynctb._tests.test_traceback.other_cb, 10, 'hi', answer=42)",
            "stack[2]",
            None,
        ),
        ("inner_context", "yield", None, None),
    ]


def frames_from_outer_context(caller):
    return [
        (caller, "with outer_context():", None, "_GeneratorContextManager"),
        *frames_from_inner_context("outer_context"),
        ("outer_context", "yield", None, None),
    ]


@try_in_other_greenlet_too
def test_running():
    # These two layers of indirection are mostly to test that skip_callees
    # works when using iterate_running.
    @customize(skip_frame=True, skip_callees=True)
    def call_call_traceback_since(root):
        return call_traceback_since(root)

    def call_traceback_since(root):
        return Traceback.since(root)

    def sync_example(root):
        with outer_context():
            if isinstance(root, types.FrameType):
                return call_call_traceback_since(root)
            else:
                return Traceback.of(root)

    # Currently running in this thread
    assert_tb_matches(
        sync_example(sys._getframe(0)),
        [
            ("test_running", "sync_example(sys._getframe(0)),", None, None),
            *frames_from_outer_context("sync_example"),
            ("sync_example", "return call_call_traceback_since(root)", None, None),
        ],
    )

    async def async_example():
        root = await async_yield(None)
        await async_yield(sync_example(root))

    def generator_example():
        root = yield
        yield sync_example(root)

    async def agen_example():
        root = yield
        yield sync_example(root)

    for which in (async_example, generator_example, agen_example):
        it = which()
        if which is agen_example:

            def send(val):
                with pytest.raises(StopIteration) as info:
                    it.asend(val).send(None)
                return info.value.value

        else:
            send = it.send
        send(None)
        if which is async_example:
            line = "await async_yield(sync_example(root))"
        else:
            line = "yield sync_example(root)"
        assert_tb_matches(
            send(it),
            [
                (which.__name__, line, None, None),
                *frames_from_outer_context("sync_example"),
                ("sync_example", "return Traceback.of(root)", None, None),
            ],
        )


def test_suspended():
    async def async_example(depth):
        if depth >= 1:
            return await async_example(depth - 1)
        with outer_context():
            return await async_yield(1)

    async def agen_example(depth):
        await async_example(depth)
        yield  # pragma: no cover

    agen_makers = [agen_example]

    try:
        import async_generator
    except ImportError:
        agen_backport_example = None
    else:

        @async_generator.async_generator
        async def agen_backport_example(depth):
            await async_example(depth)
            await yield_()  # pragma: no cover

        agen_makers.append(agen_backport_example)

    # Suspended coroutine
    coro = async_example(3)
    assert coro.send(None) == 1
    assert_tb_matches(
        Traceback.of(coro),
        [
            ("async_example", "return await async_example(depth - 1)", None, None),
            ("async_example", "return await async_example(depth - 1)", None, None),
            ("async_example", "return await async_example(depth - 1)", None, None),
            *frames_from_outer_context("async_example"),
            ("async_example", "return await async_yield(1)", None, None),
            ("async_yield", "return (yield value)", None, None),
        ],
    )
    assert_tb_matches(
        Traceback.of(coro, with_context_info=False),
        [
            ("async_example", "return await async_example(depth - 1)", None, None),
            ("async_example", "return await async_example(depth - 1)", None, None),
            ("async_example", "return await async_example(depth - 1)", None, None),
            ("async_example", "return await async_yield(1)", None, None),
            ("async_yield", "return (yield value)", None, None),
        ],
    )
    with pytest.raises(StopIteration, match="42"):
        coro.send(42)

    # Suspended async generator
    for thing in agen_makers:
        agi = thing(3)
        ags = agi.asend(None)
        assert ags.send(None) == 1
        for view in (agi, ags):
            assert_tb_matches(
                Traceback.of(view, with_context_info=False),
                [
                    (thing.__name__, "await async_example(depth)", None, None),
                    (
                        "async_example",
                        "return await async_example(depth - 1)",
                        None,
                        None,
                    ),
                    (
                        "async_example",
                        "return await async_example(depth - 1)",
                        None,
                        None,
                    ),
                    (
                        "async_example",
                        "return await async_example(depth - 1)",
                        None,
                        None,
                    ),
                    ("async_example", "return await async_yield(1)", None, None),
                    ("async_yield", "return (yield value)", None, None),
                ],
            )

    # Exhausted coro/generator has no traceback
    assert_tb_matches(Traceback.of(coro), [])


def test_greenlet():
    greenlet = pytest.importorskip("greenlet")

    tb_main = Traceback.of(greenlet.getcurrent())
    assert tb_main.error is None and tb_main.frames[-1].funcname == "test_greenlet"

    def outer():
        with outer_context():
            return inner()

    def inner():
        # Test getting the traceback of a greenlet from inside it
        assert_tb_matches(
            Traceback.of(gr),
            [
                *frames_from_outer_context("outer"),
                ("outer", "return inner()", None, None),
                ("inner", "Traceback.of(gr),", None, None),
            ],
        )
        return greenlet.getcurrent().parent.switch(1)

    gr = greenlet.greenlet(outer)
    assert_tb_matches(Traceback.of(gr), [])  # not started -> empty tb

    assert 1 == gr.switch()
    assert_tb_matches(
        Traceback.of(gr),
        [
            *frames_from_outer_context("outer"),
            ("outer", "return inner()", None, None),
            ("inner", "return greenlet.getcurrent().parent.switch(1)", None, None),
        ],
    )

    assert 2 == gr.switch(2)
    assert_tb_matches(Traceback.of(gr), [])  # dead -> empty tb

    # Test tracing into the runner for a dead greenlet

    def trivial_runner(gr):
        assert_tb_matches(
            Traceback.since(sys._getframe(0)),
            [("trivial_runner", "Traceback.since(sys._getframe(0)),", None, None)],
        )

    @register_get_target(trivial_runner)
    def get_target(frame, is_terminal):
        return frame.f_locals.get("gr")

    trivial_runner(gr)


def test_get_target_fails():
    outer_frame = sys._getframe(0)

    def inner():
        return Traceback.since(outer_frame)

    @customize(get_target=lambda *args: {}["wheee"])
    def example():
        return inner()

    # Frames that produce an error get mentioned in the traceback,
    # even if they'd otherwise be skipped
    @customize(skip_frame=True, get_target=lambda *args: {}["wheee"])
    def skippy_example():
        return inner()

    for fn in (example, skippy_example):
        assert_tb_matches(
            fn(),
            [
                ("test_get_target_fails", "fn(),", None, None),
                (fn.__name__, "return inner()", None, None),
            ],
            error=KeyError("wheee"),
        )


@pytest.mark.skipif(
    sys.implementation.name == "pypy",
    reason="https://foss.heptapod.net/pypy/pypy/-/blob/branch/py3.6/lib_pypy/greenlet.py#L124",
)
def test_greenlet_in_other_thread():
    greenlet = pytest.importorskip("greenlet")
    ready_evt = threading.Event()
    done_evt = threading.Event()
    gr = None

    def thread_fn():
        def target():
            ready_evt.set()
            done_evt.wait()

        nonlocal gr
        gr = greenlet.greenlet(target)
        gr.switch()

    threading.Thread(target=thread_fn).start()
    ready_evt.wait()
    assert_tb_matches(
        Traceback.of(gr),
        [],
        error=RuntimeError(
            "Traceback.of(greenlet) can't handle a greenlet running in another thread"
        ),
    )
    done_evt.set()


def test_exiting():
    # Test traceback when a synchronous context manager is currently exiting.
    result: Traceback

    @contextmanager
    def capture_tb_on_exit(coro):
        with inner_context() as inner:  # noqa: F841
            try:
                yield
            finally:
                nonlocal result
                result = Traceback.of(coro)

    async def async_capture_tb():
        coro = await async_yield(None)
        with capture_tb_on_exit(coro):
            pass
        await async_yield(result)

    coro = async_capture_tb()
    coro.send(None)
    assert_tb_matches(
        coro.send(coro),
        [
            (
                "async_capture_tb",
                "with capture_tb_on_exit(coro):",
                None,
                "_GeneratorContextManager",
            ),
            ("async_capture_tb", "pass", None, None),
            ("__exit__", "next(self.gen)", None, None),
            *frames_from_inner_context("capture_tb_on_exit"),
            ("capture_tb_on_exit", "result = Traceback.of(coro)", None, None),
        ],
    )

    # Test traceback when an async CM is suspended in __aexit__.  The
    # definition of __aexit__ as a staticmethod is to foil the logic
    # for figuring out which context manager is exiting.

    class SillyAsyncCM:
        async def __aenter__(self):
            pass

        @staticmethod
        async def __aexit__(*stuff):
            await async_yield(None)

    async def yield_when_async_cm_exiting():
        async with SillyAsyncCM():
            pass

    coro = yield_when_async_cm_exiting()
    coro.send(None)
    assert_tb_matches(
        Traceback.of(coro),
        [
            ("yield_when_async_cm_exiting", "async with SillyAsyncCM():", None, None),
            ("yield_when_async_cm_exiting", "pass", None, None),
            ("__aexit__", "await async_yield(None)", None, None),
            ("async_yield", "return (yield value)", None, None),
        ],
    )


def test_errors():
    with pytest.raises(TypeError, match="must be a frame"):
        Traceback.since(42)
    with pytest.raises(TypeError, match="must be a frame or integer"):
        Traceback.until(sys._getframe(0), limit=2.4)
    with pytest.raises(RuntimeError, match="is not an indirect caller of"):
        Traceback.until(sys._getframe(1), limit=sys._getframe(0))


@try_in_other_greenlet_too
def test_traceback_until():
    outer = sys._getframe(0)

    def example():
        inner = sys._getframe(0)

        def get_tb(limit):
            return Traceback.until(inner, limit=limit)

        tb1, tb2, tb3 = [get_tb(lim) for lim in (1, outer, None)]
        assert tb1 == tb2
        assert tb3.frames[-len(tb1) :] == tb1.frames
        assert_tb_matches(
            tb1,
            [
                ("test_traceback_until", "example()", None, None),
                (
                    "example",
                    "tb1, tb2, tb3 = [get_tb(lim) for lim in (1, outer, None)]",
                    None,
                    None,
                ),
            ],
        )

    example()


@try_in_other_greenlet_too
def test_running_in_thread():
    def thread_example(arrived_evt, depart_evt):
        with outer_context():
            arrived_evt.set()
            depart_evt.wait()

    def thread_caller(*args):
        thread_example(*args)

    # Currently running in other thread
    for cooked in (False, True):
        arrived_evt = threading.Event()
        depart_evt = threading.Event()
        thread = threading.Thread(target=thread_caller, args=(arrived_evt, depart_evt))
        thread.start()
        try:
            arrived_evt.wait()
            if cooked:
                tb = Traceback.of(thread)
            else:
                top_frame = sys._current_frames()[thread.ident]
                while (
                    top_frame.f_back is not None
                    and top_frame.f_code.co_name != "thread_caller"
                ):
                    top_frame = top_frame.f_back
                tb = Traceback.since(top_frame)

            # Exactly where we are inside Event.wait() is indeterminate, so
            # strip frames until we find Event.wait() and then remove it

            while (
                not tb.frames[-1].filename.endswith("threading.py")
                or tb.frames[-1].funcname != "wait"
            ):  # pragma: no cover
                tb = attr.evolve(tb, frames=tb.frames[:-1])
            while tb.frames[-1].filename.endswith("threading.py"):  # pragma: no cover
                tb = attr.evolve(tb, frames=tb.frames[:-1])

            assert_tb_matches(
                tb,
                [
                    ("thread_caller", "thread_example(*args)", None, None),
                    *frames_from_outer_context("thread_example"),
                    ("thread_example", "depart_evt.wait()", None, None),
                ],
            )
        finally:
            depart_evt.set()


def test_traceback_of_not_alive_thread(isolated_registry):
    thread = threading.Thread(target=lambda: None)
    assert_tb_matches(Traceback.of(thread), [])
    thread.start()
    thread.join()
    assert_tb_matches(Traceback.of(thread), [])

    @customize(get_target=lambda *_: thread)
    async def example():
        await async_yield(42)

    coro = example()
    coro.send(None)
    assert_tb_matches(
        Traceback.of(coro),
        [
            ("example", "await async_yield(42)", None, None),
            ("async_yield", "return (yield value)", None, None),
        ],
    )


def test_trace_into_thread(local_registry):
    trio = pytest.importorskip("trio")
    import outcome

    # Extremely simplified version of trio.to_thread.run_sync
    async def run_sync_in_thread(sync_fn):
        task = trio.lowlevel.current_task()
        trio_token = trio.lowlevel.current_trio_token()

        def run_it():
            result = outcome.capture(sync_fn)
            trio_token.run_sync_soon(trio.lowlevel.reschedule, task, result)

        thread = threading.Thread(target=run_it)
        thread.start()
        return await trio.lowlevel.wait_task_rescheduled(no_abort)

    @register_get_target(run_sync_in_thread)
    def get_target(this_frame, next_frame):
        return this_frame.f_locals["thread"]

    customize(run_sync_in_thread, "run_it", skip_frame=True)

    tb = None

    async def main():
        arrived_evt = trio.Event()
        depart_evt = threading.Event()
        trio_token = trio.lowlevel.current_trio_token()
        task = trio.lowlevel.current_task()

        def sync_fn():
            with inner_context() as inner:  # noqa: F841
                trio_token.run_sync_soon(arrived_evt.set)
                depart_evt.wait()

        def sync_wrapper():
            sync_fn()

        async def capture_tb():
            nonlocal tb
            try:
                await arrived_evt.wait()
                tb = Traceback.of(task.coro)
            finally:
                depart_evt.set()

        async with trio.open_nursery() as nursery:
            nursery.start_soon(capture_tb)
            await run_sync_in_thread(sync_wrapper)

    trio.run(main)

    # It's indeterminate where in sync_fn() the traceback was taken -- it could
    # be inside run_sync_soon() or inside threading.Event.wait() -- so trim
    # traceback frames until we get something reliable.
    while tb.frames[-1].filename != __file__:
        tb = attr.evolve(tb, frames=tb.frames[:-1])
    tb = attr.evolve(
        tb,
        frames=tb.frames[:-1]
        + (attr.evolve(tb.frames[-1], override_line="<indeterminate>"),),
    )
    assert_tb_matches(
        tb,
        [
            (
                "main",
                "async with trio.open_nursery() as nursery:",
                "nursery",
                "Nursery",
            ),
            ("main", "await run_sync_in_thread(sync_wrapper)", None, None),
            (
                "run_sync_in_thread",
                "return await trio.lowlevel.wait_task_rescheduled(no_abort)",
                None,
                None,
            ),
            ("sync_wrapper", "sync_fn()", None, None),
            *frames_from_inner_context("sync_fn"),
            ("sync_fn", "<indeterminate>", None, None),
        ],
    )


@pytest.mark.skipif(
    sys.implementation.name == "pypy",
    reason="profile function doesn't get called on Travis",
)
def test_threaded_race():
    # This tests the case where we're getting the traceback of a coroutine
    # running in a foreign thread, but it becomes suspended before we can
    # extract the foreign thread's stack.

    afn_running = threading.Event()
    suspend_afn = threading.Event()
    afn_suspended = threading.Event()
    resume_afn = threading.Event()

    async def async_fn():
        with outer_context():
            afn_running.set()
            suspend_afn.wait()
            await async_yield(1)

    coro = async_fn()

    def runner():
        coro.send(None)
        afn_suspended.set()
        resume_afn.wait()
        with pytest.raises(StopIteration):
            coro.send(None)

    def suspend_at_proper_place(frame, event, arg):  # pragma: no cover
        # (profile functions don't get traced)
        if (
            event == "call"
            and frame.f_globals is Traceback.of.__func__.__globals__
            and frame.f_code.co_name == "try_from"
        ):
            suspend_afn.set()
            afn_suspended.wait()

    old_profile = sys.getprofile()
    sys.setprofile(suspend_at_proper_place)
    try:
        thread = threading.Thread(target=runner)
        thread.start()
        assert_tb_matches(
            Traceback.of(coro),
            [
                *frames_from_outer_context("async_fn"),
                ("async_fn", "await async_yield(1)", None, None),
                ("async_yield", "return (yield value)", None, None),
            ],
        )
    finally:
        sys.setprofile(old_profile)
        resume_afn.set()


def test_unknown_awaitable():
    class WeirdObject:
        def __await__(self):
            return iter([42])

    async def example():
        await WeirdObject()

    coro = example()
    assert 42 == coro.send(None)
    name = "sequence" if sys.implementation.name == "pypy" else "list_"
    assert_tb_matches(
        Traceback.of(coro),
        [("example", "await WeirdObject()", None, None)],
        error=RuntimeError(
            f"Couldn't determine the frame associated with builtins.{name}iterator "
            f"<{name}iterator object at (address)>",
        ),
    )

    assert_tb_matches(
        Traceback.of(42),
        [],
        error=RuntimeError(
            "Couldn't determine the frame associated with builtins.int 42"
        ),
    )


def test_cant_get_referents(monkeypatch):
    async def agen():
        await async_yield(1)
        yield

    async def afn():
        await async_yield(1)

    class SomeAwaitable:
        def __await__(self):
            return wrapper

    ags = agen().asend(None)
    wrapper = afn().__await__()
    real_get_referents = gc.get_referents

    def patched_get_referents(obj):
        if obj is ags or obj is wrapper:
            return []
        return real_get_referents(obj)

    monkeypatch.setattr(gc, "get_referents", patched_get_referents)

    async def await_it(thing):
        await thing

    for thing, problem, attrib in (
        (ags, ags, "an ag_frame"),
        (SomeAwaitable(), wrapper, "a cr_frame"),
    ):
        coro = await_it(thing)
        assert 1 == coro.send(None)
        assert_tb_matches(
            Traceback.of(coro),
            [("await_it", "await thing", None, None)],
            error=RuntimeError(
                f"{problem!r} doesn't refer to anything with {attrib} attribute"
            ),
        )
        with pytest.raises(StopIteration):
            coro.send(None)


def test_cant_find_running_frame():
    greenlet = pytest.importorskip("greenlet")

    async def caller():
        await example()

    async def example():
        with outer_context():
            greenlet.getcurrent().parent.switch(42)

    coro = caller()
    gr = greenlet.greenlet(coro.send)
    assert gr.switch(None) == 42
    assert_tb_matches(
        Traceback.of(coro),
        [("caller", "await example()", None, None)],
        error=RuntimeError(
            "Couldn't find where the above frame is running, so can't continue "
            "traceback"
        ),
    )
    with pytest.raises(StopIteration):
        gr.switch(None)


def test_with_trickery_disabled(monkeypatch):
    import asynctb

    monkeypatch.setattr(asynctb._frames, "_can_use_trickery", False)

    def sync_example(root):
        with outer_context():
            return Traceback.since(root)

    # CPython GC doesn't crawl currently executing frames, so we get more
    # data without trickery on PyPy than on CPython
    only_on_pypy = [
        ("sync_example", "", None, "_GeneratorContextManager"),
        ("outer_context", "", None, "_GeneratorContextManager"),
        ("inner_context", "", None, "ExitStack"),
        (
            "inner_context",
            "# _.enter_context(asynctb._tests.test_traceback.null_context())",
            "_[0]",
            "_GeneratorContextManager",
        ),
        ("null_context", "yield", None, None),
        (
            "inner_context",
            "# _.push(asynctb._tests.test_traceback.exit_cb)",
            "_[1]",
            None,
        ),
        (
            "inner_context",
            "# _.callback(asynctb._tests.test_traceback.other_cb, 10, 'hi', answer=42)",
            "_[2]",
            None,
        ),
        ("inner_context", "yield", None, None),
        ("outer_context", "yield", None, None),
    ]
    assert_tb_matches(
        sync_example(sys._getframe(0)),
        [
            (
                "test_with_trickery_disabled",
                "sync_example(sys._getframe(0)),",
                None,
                None,
            ),
            *(only_on_pypy if sys.implementation.name == "pypy" else []),
            ("sync_example", "return Traceback.since(root)", None, None),
        ],
    )

    async def async_example():
        with outer_context():
            return await async_yield(42)

    coro = async_example()
    assert 42 == coro.send(None)
    assert_tb_matches(
        Traceback.of(coro),
        [
            ("async_example", "", None, "_GeneratorContextManager"),
            ("outer_context", "", None, "_GeneratorContextManager"),
            ("inner_context", "", None, "ExitStack"),
            (
                "inner_context",
                f"# _.enter_context({null_context_repr})",
                "_[0]",
                "_GeneratorContextManager",
            ),
            ("null_context", "yield", None, None),
            (
                "inner_context",
                "# _.push(asynctb._tests.test_traceback.exit_cb)",
                "_[1]",
                None,
            ),
            (
                "inner_context",
                "# _.callback(asynctb._tests.test_traceback.other_cb, 10, 'hi', answer=42)",
                "_[2]",
                None,
            ),
            ("inner_context", "yield", None, None),
            ("outer_context", "yield", None, None),
            ("async_example", "return await async_yield(42)", None, None),
            ("async_yield", "return (yield value)", None, None),
        ],
    )


def test_trio_nursery():
    trio = pytest.importorskip("trio")
    async_generator = pytest.importorskip("async_generator")

    @async_generator.asynccontextmanager
    @async_generator.async_generator
    async def uses_nursery():
        async with trio.open_nursery() as inner:  # noqa: F841
            await async_generator.yield_()

    async def main():
        result: Traceback
        task = trio.lowlevel.current_task()

        def report_back():
            nonlocal result
            result = Traceback.of(task.coro)
            trio.lowlevel.reschedule(task)

        async with trio.open_nursery() as outer, uses_nursery():  # noqa: F841
            trio.lowlevel.current_trio_token().run_sync_soon(report_back)
            await trio.lowlevel.wait_task_rescheduled(no_abort)

        return result

    assert_tb_matches(
        trio.run(main),
        [
            (
                "main",
                "async with trio.open_nursery() as outer, uses_nursery():",
                "outer",
                "Nursery",
            ),
            (
                "main",
                "async with trio.open_nursery() as outer, uses_nursery():",
                None,
                "_AsyncGeneratorContextManager",
            ),
            (
                "uses_nursery",
                "async with trio.open_nursery() as inner:",
                "inner",
                "Nursery",
            ),
            ("uses_nursery", "await async_generator.yield_()", None, None),
            ("main", "await trio.lowlevel.wait_task_rescheduled(no_abort)", None, None),
        ],
    )


def test_greenback():
    trio = pytest.importorskip("trio")
    greenback = pytest.importorskip("greenback")
    results: List[Traceback] = []

    async def outer():
        async with trio.open_nursery() as outer_nursery:  # noqa: F841
            middle()
            await inner()

    def middle():
        nursery_mgr = trio.open_nursery()
        with greenback.async_context(nursery_mgr) as middle_nursery:  # noqa: F841
            greenback.await_(inner())

            # This winds up traversing an await_ before it has a coroutine to use.
            class ExtractWhenAwaited:
                def __await__(self):
                    task = trio.lowlevel.current_task()
                    assert_tb_matches(
                        Traceback.of(task.coro),
                        [
                            (
                                "greenback_shim",
                                "return await _greenback_shim(orig_coro)",
                                None,
                                None,
                            ),
                            ("main", "return await outer()", None, None),
                            (
                                "outer",
                                "async with trio.open_nursery() as outer_nursery:",
                                "outer_nursery",
                                "Nursery",
                            ),
                            ("outer", "middle()", None, None),
                            (
                                "middle",
                                "with greenback.async_context(nursery_mgr) as middle_nursery:",
                                "middle_nursery",
                                "Nursery",
                            ),
                            (
                                "middle",
                                "greenback.await_(ExtractWhenAwaited())",
                                None,
                                None,
                            ),
                            ("adapt_awaitable", "return await aw", None, None),
                            ("__await__", "Traceback.of(task.coro),", None, None),
                        ],
                    )
                    yield from ()

            greenback.await_(ExtractWhenAwaited())  # pragma: no cover

    async def inner():
        with null_context():
            task = trio.lowlevel.current_task()

            def report_back():
                results.append(Traceback.of(task.coro))
                trio.lowlevel.reschedule(task)

            trio.lowlevel.current_trio_token().run_sync_soon(report_back)
            await trio.lowlevel.wait_task_rescheduled(no_abort)

    async def main():
        await greenback.ensure_portal()
        return await outer()

    trio.run(main)
    assert len(results) == 2
    assert_tb_matches(
        results[0],
        [
            ("greenback_shim", "return await _greenback_shim(orig_coro)", None, None,),
            ("main", "return await outer()", None, None),
            (
                "outer",
                "async with trio.open_nursery() as outer_nursery:",
                "outer_nursery",
                "Nursery",
            ),
            ("outer", "middle()", None, None),
            (
                "middle",
                "with greenback.async_context(nursery_mgr) as middle_nursery:",
                "middle_nursery",
                "Nursery",
            ),
            ("middle", "greenback.await_(inner())", None, None),
            ("inner", "with null_context():", None, "_GeneratorContextManager"),
            ("null_context", "yield", None, None),
            (
                "inner",
                "await trio.lowlevel.wait_task_rescheduled(no_abort)",
                None,
                None,
            ),
        ],
    )
    assert_tb_matches(
        results[1],
        [
            ("greenback_shim", "return await _greenback_shim(orig_coro)", None, None,),
            ("main", "return await outer()", None, None),
            (
                "outer",
                "async with trio.open_nursery() as outer_nursery:",
                "outer_nursery",
                "Nursery",
            ),
            ("outer", "await inner()", None, None),
            ("inner", "with null_context():", None, "_GeneratorContextManager"),
            ("null_context", "yield", None, None),
            (
                "inner",
                "await trio.lowlevel.wait_task_rescheduled(no_abort)",
                None,
                None,
            ),
        ],
    )


def test_exitstack_formatting():
    class A:
        def __repr__(self):
            return "A()"

        def method(self, *args):
            pass

    with ExitStack() as stack:
        stack.callback(A().method)
        stack.push(A().method)
        stack.callback(partial(lambda x: None, 42))
        tb = Traceback.since(sys._getframe(0))
        assert_tb_matches(
            tb,
            [
                (
                    "test_exitstack_formatting",
                    "with ExitStack() as stack:",
                    "stack",
                    "ExitStack",
                ),
                (
                    "test_exitstack_formatting",
                    "# stack.callback(A().method)",
                    "stack[0]",
                    None,
                ),
                (
                    "test_exitstack_formatting",
                    "# stack.push(A().method)",
                    "stack[1]",
                    "A",
                ),
                (
                    "test_exitstack_formatting",
                    "# stack.callback(functools.partial(<function test_exitstack_formatting.<locals>.<lambda> at (address)>, 42))",
                    "stack[2]",
                    None,
                ),
                (
                    "test_exitstack_formatting",
                    "tb = Traceback.since(sys._getframe(0))",
                    None,
                    None,
                ),
            ],
        )


ACM_IMPLS: List[Callable[..., Any]] = []
try:
    ACM_IMPLS.append(cast(Any, contextlib).asynccontextmanager)
except AttributeError:
    pass
try:
    import async_generator
except ImportError:
    pass
else:
    ACM_IMPLS.append(async_generator.asynccontextmanager)


@pytest.mark.parametrize("asynccontextmanager", ACM_IMPLS)
def test_asyncexitstack_formatting(asynccontextmanager):
    try:
        from contextlib import AsyncExitStack
    except ImportError:
        try:
            from async_exit_stack import AsyncExitStack  # type: ignore
        except ImportError:  # pragma: no cover
            pytest.skip("no AsyncExitStack")

    class A:
        def __repr__(self):
            return "<A>"

        async def __aenter__(self):
            pass

        async def __aexit__(self, *exc):
            pass

        async def aexit(self, *exc):
            pass

    async def aexit2(*exc):
        pass

    async def acallback(*args):
        pass

    @asynccontextmanager
    async def amgr():
        yield

    async def async_fn():
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(A())
            await stack.enter_async_context(amgr())
            stack.push_async_exit(A().aexit)
            stack.push_async_exit(aexit2)
            stack.push_async_callback(acallback, "hi")
            await async_yield(None)

    if asynccontextmanager.__module__.startswith("async_generator"):
        expect_name = "amgr(...)"
    else:
        expect_name = (
            "asynctb._tests.test_traceback.test_asyncexitstack_formatting."
            "<locals>.amgr()"
        )

    coro = async_fn()
    assert coro.send(None) is None
    assert_tb_matches(
        Traceback.of(coro),
        [
            (
                "async_fn",
                "async with AsyncExitStack() as stack:",
                "stack",
                "AsyncExitStack",
            ),
            ("async_fn", "# await stack.enter_async_context(<A>)", "stack[0]", "A"),
            (
                "async_fn",
                f"# await stack.enter_async_context({expect_name})",
                "stack[1]",
                "_AsyncGeneratorContextManager",
            ),
            ("amgr", "yield", None, None),
            ("async_fn", "# stack.push_async_exit(<A>.aexit)", "stack[2]", "A"),
            (
                "async_fn",
                "# stack.push_async_exit(asynctb._tests.test_traceback.test_asyncexitstack_formatting.<locals>.aexit2)",
                "stack[3]",
                None,
            ),
            (
                "async_fn",
                "# stack.push_async_callback(asynctb._tests.test_traceback.test_asyncexitstack_formatting.<locals>.acallback, 'hi')",
                "stack[4]",
                None,
            ),
            ("async_fn", "await async_yield(None)", None, None),
            ("async_yield", "return (yield value)", None, None),
        ],
    )
