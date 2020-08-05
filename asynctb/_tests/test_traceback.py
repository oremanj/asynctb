import contextlib
import gc
import pytest
import re
import sys
import threading
import types
from contextlib import ExitStack, contextmanager
from functools import partial
from .. import FrameInfo, Traceback


def remove_address_details(line):
    return re.sub(r"\b0x[0-9a-f]+\b", "(address)", line)


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
            assert remove_address_details(entry.linetext) == expect_line
            assert entry.context_name == expect_ctx_name
            if entry.context_manager is None:
                assert expect_ctx_typename is None
            else:
                assert type(entry.context_manager).__name__ == expect_ctx_typename
    except Exception:
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
            linetext = remove_address_details(entry.linetext)
        typename = type(entry.context_manager).__name__
        if typename == "NoneType":
            typename = None
        record = (funcname, linetext, entry.context_name, typename)
        print("            " + repr(record) + ",")
    print("        ],")
    if tb.error:
        print(f"        error={remove_address_details(repr(tb.error))},")
    print("    )")


@contextmanager
def null_context():
    yield


@contextmanager
def outer_context():
    with inner_context() as inner:
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


if sys.version_info >= (3, 7):
    null_context_repr = "null_context(...)"
else:
    null_context_repr = "asynctb._tests.test_traceback.null_context()"


def test_running():
    def sync_example(root):
        with outer_context():
            if isinstance(root, types.FrameType):
                return Traceback.since(root)
            else:
                return Traceback.of(root)

    sync_example_tbdata = [
        ("sync_example", "with outer_context():", None, "_GeneratorContextManager"),
        (
            "outer_context",
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
        ("outer_context", "yield", None, None),
    ]

    # Currently running in this thread
    assert_tb_matches(
        sync_example(sys._getframe(0)),
        [
            ("test_running", "sync_example(sys._getframe(0)),", None, None),
            *sync_example_tbdata,
            ("sync_example", "return Traceback.since(root)", None, None),
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
                *sync_example_tbdata,
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
            (
                "async_example",
                "with outer_context():",
                None,
                "_GeneratorContextManager",
            ),
            (
                "outer_context",
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
            ("outer_context", "yield", None, None),
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

    def outer():
        with outer_context():
            return inner()

    def inner():
        return greenlet.getcurrent().parent.switch(1)

    gr = greenlet.greenlet(outer)
    assert_tb_matches(Traceback.of(gr), [])  # not started -> empty tb

    assert 1 == gr.switch()
    assert_tb_matches(
        Traceback.of(gr),
        [
            ("outer", "with outer_context():", None, "_GeneratorContextManager"),
            (
                "outer_context",
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
            ("outer_context", "yield", None, None),
            ("outer", "return inner()", None, None),
            ("inner", "return greenlet.getcurrent().parent.switch(1)", None, None),
        ],
    )

    assert 2 == gr.switch(2)
    assert_tb_matches(Traceback.of(gr), [])  # dead -> empty tb


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
        with inner_context() as inner:
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
            (
                "capture_tb_on_exit",
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


def test_traceback_until():
    outer = sys._getframe(0)

    def example():
        inner = sys._getframe(0)
        tb1, tb2, tb3 = [Traceback.until(inner, limit=lim) for lim in (1, outer, None)]
        assert tb1 == tb2
        assert tb3.frames[-len(tb1) :] == tb1.frames
        assert_tb_matches(
            tb1,
            [
                ("test_traceback_until", "example()", None, None),
                (
                    "example",
                    "tb1, tb2, tb3 = [Traceback.until(inner, limit=lim) for lim in (1, outer, None)]",
                    None,
                    None,
                ),
            ],
        )

    example()


def test_running_in_thread():
    def thread_example(arrived_evt, depart_evt):
        with outer_context():
            arrived_evt.set()
            depart_evt.wait()

    def thread_caller(*args):
        thread_example(*args)

    # Currently running in other thread
    arrived_evt = threading.Event()
    depart_evt = threading.Event()
    thread = threading.Thread(target=thread_caller, args=(arrived_evt, depart_evt))
    thread.start()
    try:
        arrived_evt.wait()
        top_frame = sys._current_frames()[thread.ident]
        while (
            top_frame.f_back is not None and top_frame.f_code.co_name != "thread_caller"
        ):
            top_frame = top_frame.f_back
        assert_tb_matches(
            Traceback.since(top_frame),
            [
                ("thread_caller", "thread_example(*args)", None, None),
                (
                    "thread_example",
                    "with outer_context():",
                    None,
                    "_GeneratorContextManager",
                ),
                (
                    "outer_context",
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
                ("outer_context", "yield", None, None),
                ("thread_example", "depart_evt.wait()", None, None),
                ("wait", "with self._cond:", None, "Condition"),
                ("wait", "signaled = self._cond.wait(timeout)", None, None),
                ("wait", "waiter.acquire()", None, None),
            ],
        )
    finally:
        depart_evt.set()


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
                ("async_fn", "with outer_context():", None, "_GeneratorContextManager"),
                (
                    "outer_context",
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
                ("outer_context", "yield", None, None),
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
    assert_tb_matches(
        Traceback.of(coro),
        [("example", "await WeirdObject()", None, None)],
        error=RuntimeError(
            "Couldn't determine the frame associated with builtins.list_iterator "
            "<list_iterator object at (address)>",
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

    for thing, problem, attr in (
        (ags, ags, "an ag_frame"),
        (SomeAwaitable(), wrapper, "a cr_frame"),
    ):
        coro = await_it(thing)
        assert 1 == coro.send(None)
        assert_tb_matches(
            Traceback.of(coro),
            [("await_it", "await thing", None, None)],
            error=RuntimeError(
                f"{problem!r} doesn't refer to anything with {attr} attribute"
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

    assert_tb_matches(
        sync_example(sys._getframe(0)),
        [
            (
                "test_with_trickery_disabled",
                "sync_example(sys._getframe(0)),",
                None,
                None,
            ),
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


def no_abort(_):
    return trio.lowlevel.Abort.FAILED  # pragma: no cover


def test_trio_nursery():
    trio = pytest.importorskip("trio")
    async_generator = pytest.importorskip("async_generator")

    @async_generator.asynccontextmanager
    @async_generator.async_generator
    async def uses_nursery():
        async with trio.open_nursery() as inner:
            await async_generator.yield_()

    async def main():
        result: Traceback
        task = trio.lowlevel.current_task()

        def report_back():
            nonlocal result
            result = Traceback.of(task.coro)
            trio.lowlevel.reschedule(task)

        async with trio.open_nursery() as outer, uses_nursery():
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
        async with trio.open_nursery() as outer_nursery:
            middle()
            await inner()

    def middle():
        with greenback.async_context(trio.open_nursery()) as middle_nursery:
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
                                "return await _greenback_shim(orig_coro)  # type: ignore",
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
                                "with greenback.async_context(trio.open_nursery()) as middle_nursery:",
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

            greenback.await_(ExtractWhenAwaited())

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
            (
                "greenback_shim",
                "return await _greenback_shim(orig_coro)  # type: ignore",
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
                "with greenback.async_context(trio.open_nursery()) as middle_nursery:",
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
            (
                "greenback_shim",
                "return await _greenback_shim(orig_coro)  # type: ignore",
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
                    "# stack.callback(<asynctb._tests.test_traceback.test_exitstack_formatting.<locals>.A object at (address)>.method)",
                    "stack[0]",
                    None,
                ),
                (
                    "test_exitstack_formatting",
                    "# stack.push(<asynctb._tests.test_traceback.test_exitstack_formatting.<locals>.A object at (address)>.method)",
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


ACM_IMPLS = []
try:
    ACM_IMPLS.append(contextlib.asynccontextmanager)
except AttributeError:
    pass
try:
    import async_generator
except ImportError:
    pass
else:
    ACM_IMPLS.append(async_generator.asynccontextmanager)


@pytest.mark.parametrize("asynccontextmanager", ACM_IMPLS)
def test_asyncexitstack_foramtting(asynccontextmanager):
    try:
        from contextlib import AsyncExitStack
    except ImportError:
        try:
            from async_exit_stack import AsyncExitStack
        except ImportError:
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
            "asynctb._tests.test_traceback.test_asyncexitstack_foramtting."
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
                "# stack.push_async_exit(asynctb._tests.test_traceback.test_asyncexitstack_foramtting.<locals>.aexit2)",
                "stack[3]",
                None,
            ),
            (
                "async_fn",
                "# stack.push_async_callback(asynctb._tests.test_traceback.test_asyncexitstack_foramtting.<locals>.acallback, 'hi')",
                "stack[4]",
                None,
            ),
            ("async_fn", "await async_yield(None)", None, None),
            ("async_yield", "return (yield value)", None, None),
        ],
    )
