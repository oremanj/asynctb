import contextlib
import dis
import sys
import types

import attr
import pytest

from asynctb import _frames


original_inspect_frame = _frames.inspect_frame


@types.coroutine
def async_yield(val):
    return (yield val)


class YieldsDuringAexit:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        await async_yield(42)


def test_trickery_unavailable(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys.implementation, "name", "unsupported")
        m.setattr(_frames, "_can_use_trickery", None)
        with pytest.warns(_frames.InspectionWarning, match="trickery is not supported"):
            _frames.contexts_active_in_frame(sys._getframe(0))

    def boom(frame):
        raise ValueError("nope")

    with monkeypatch.context() as m:
        m.setattr(_frames, "_contexts_active_by_trickery", boom)
        m.setattr(_frames, "_can_use_trickery", None)
        with pytest.warns(_frames.InspectionWarning, match="trickery doesn't work"):
            _frames.contexts_active_in_frame(sys._getframe(0))

    _frames.contexts_active_in_frame(sys._getframe(0))

    with monkeypatch.context() as m:
        m.setattr(_frames, "_contexts_active_by_trickery", boom)
        with pytest.warns(_frames.InspectionWarning, match="trickery failed on frame"):
            _frames.contexts_active_in_frame(sys._getframe(0))

    with monkeypatch.context() as m:
        m.setattr(sys.implementation, "name", "unsupported")
        m.setattr(_frames, "inspect_frame", original_inspect_frame)

        with pytest.warns(_frames.InspectionWarning, match="details not supported"):
            _frames.contexts_active_in_frame(sys._getframe(0))


def test_contexts_by_referents(monkeypatch):
    monkeypatch.setattr(_frames, "_can_use_trickery", False)

    async def afn():
        with contextlib.ExitStack() as stack:  # noqa: F841
            async with YieldsDuringAexit():
                with contextlib.ExitStack() as stack2:  # noqa: F841
                    pass

    coro = afn()
    assert 42 == coro.send(None)
    assert _frames.contexts_active_in_frame(coro.cr_frame) == [
        _frames.ContextInfo(is_async=False, manager=coro.cr_frame.f_locals["stack"]),
        _frames.ContextInfo(is_async=True, is_exiting=True, manager=None),
    ]
    coro.close()


def test_frame_not_started():
    async def afn():
        with contextlib.ExitStack():  # pragma: no cover
            pass

    coro = afn()
    assert _frames.contexts_active_in_frame(coro.cr_frame) == []
    coro.close()


def test_non_async_yf():
    def subgen():
        yield 42

    def gen():
        with contextlib.ExitStack() as stack:  # noqa: F841
            yield from subgen()

    gi = gen()
    assert gi.send(None) == 42
    (info,) = _frames.contexts_active_in_frame(gi.gi_frame)
    assert not info.is_async
    assert info.manager is gi.gi_frame.f_locals["stack"]


def test_aexit_extended_arg():
    # Create a function that uses 256 constants before its first None,
    # so that when it does LOAD_CONST None in __aexit__ it needs an
    # EXTENDED_ARG.  Note that such a function requires a docstring,
    # because the first constant is the docstring and with no
    # docstring there's your None.
    lines = [
        "async def example():",
        "    '''Docstring.'''",
        "    s = 'A' * 1000",
        "    lst = [{}]".format(", ".join(f"s[{idx}]" for idx in range(1000))),
        "    async with YieldsDuringAexit():",
        "        pass",
    ]
    ns = {}
    exec("\n".join(lines), globals(), ns)
    coro = ns["example"]()
    assert 42 == coro.send(None)
    assert _frames.contexts_active_in_frame(coro.cr_frame) == [
        _frames.ContextInfo(is_async=True, is_exiting=True, manager=None, start_line=5),
    ]
    coro.close()


def test_jump_out_of_context():
    async def afn(arg):
        try:
            for _ in range(2):  # pragma: no branch
                async with YieldsDuringAexit() as mgr:  # noqa: F841
                    if arg == 1:
                        return arg * 10
                    if arg == 2:
                        return 20
                    if arg == 3:
                        break
                    if arg == 4:
                        raise KeyError(arg * 10)
                if arg == 5:  # pragma: no branch
                    return 50
        except KeyError as ex:
            return ex.args[0]
        return 30

    for arg in range(1, 6):
        coro = afn(arg)
        assert 42 == coro.send(None)
        (info,) = _frames.contexts_active_in_frame(coro.cr_frame)
        assert info.is_async and info.is_exiting and info.varname == "mgr"
        assert info.start_line == afn.__code__.co_firstlineno + 3
        with pytest.raises(StopIteration) as exc_info:
            coro.send(None)
        assert exc_info.value.value == arg * 10


def test_context_all_exits_are_jumps():
    async def afn(throw):
        async with YieldsDuringAexit() as mgr:  # noqa: F841
            if throw:
                raise ValueError
            return throw * 10

    for throw in (False, True):
        coro = afn(throw)
        assert 42 == coro.send(None)
        (info,) = _frames.contexts_active_in_frame(coro.cr_frame)
        assert info.is_async and info.is_exiting and info.varname == "mgr"
        assert info.start_line == afn.__code__.co_firstlineno + 1
        coro.close()


@pytest.mark.skipif(sys.version_info[:2] != (3, 8), reason="test is specific to 3.8")
def test_unreachable_pop_block():
    # We'll replace foo() here with the raising of an exception in a way that
    # doesn't match anything the Python compiler can actually generate, which
    # tickles a "can't happen" in _frames._currently_exiting_context.

    async def fn():
        async with YieldsDuringAexit():
            ValueError()

    op = dis.opmap
    # fmt: off
    search = [
        op["LOAD_GLOBAL"], 1,
        op["CALL_FUNCTION"], 0,
        op["POP_TOP"], 0,
    ]
    replace = [
        op["LOAD_GLOBAL"], 1,
        op["RAISE_VARARGS"], 1,
        op["NOP"], 0,
    ]
    # fmt: on
    fn.__code__ = fn.__code__.replace(
        co_code=fn.__code__.co_code.replace(bytes(search), bytes(replace))
    )
    coro = fn()
    coro.send(None)
    with pytest.warns(_frames.InspectionWarning, match="doesn't appear reachable"):
        _frames.contexts_active_in_frame(coro.cr_frame)
    coro.close()


@pytest.mark.skipif(sys.version_info < (3, 8), reason="uses CodeType.replace()")
def test_unexpected_aexit_sequence():
    # Interpose a no-op sequence in between LOAD_CONST None and YIELD_FROM,
    # which are normally inseparable, in order to tickle a "can't happen"
    # in _frames._currently_exiting_context.

    async def example():
        async with YieldsDuringAexit():
            pass

    op = dis.opmap
    if sys.version_info < (3, 9):
        before = [op["WITH_CLEANUP_START"], 0]
    else:
        before = [op["LOAD_CONST"], 0, op["DUP_TOP"], 0, op["DUP_TOP"], 0]
        before += [op["CALL_FUNCTION"], 3]
    before += [op["GET_AWAITABLE"], 0, op["LOAD_CONST"], 0]
    after = [op["YIELD_FROM"], 0]
    co = example.__code__
    example.__code__ = co.replace(
        co_code=co.co_code.replace(
            bytes(before + after),
            bytes(before + [op["LOAD_CONST"], 0, op["POP_TOP"], 0] + after),
        ),
    )
    coro = example()
    assert 42 == coro.send(None)
    with pytest.warns(_frames.InspectionWarning, match="not preceded by LOAD_CONST"):
        _frames.contexts_active_in_frame(coro.cr_frame)
    coro.close()


def test_assignment_targets():
    @attr.s
    class C:
        val = attr.ib()

        def __enter__(self):
            return self.val

        def __exit__(self, *exc):
            pass

    async def example():
        class NS(object):
            pass

        ns = NS()
        ns.sub = NS()
        dct = {"sub": {}}
        with C((1, 2)) as (a, b), C((3,)) as [c]:
            with C(4) as ns.foo, C(5) as ns.sub.foo:
                with C(6) as dct["one"], C(7) as dct["sub"]["two"]:
                    with C(8) as dct.__getitem__("sub")["three"]:
                        with C(9) as locals()["ns"].bar, C(10):
                            with C(11) as locals()[b"x".decode("ascii") + "y"]:
                                assert (a, b, c) == (1, 2, 3)
                                assert ns.foo == 4 and ns.sub.foo == 5
                                assert ns.bar == 9
                                assert dct["one"] == 6
                                assert dct["sub"] == {"two": 7, "three": 8}
                                await async_yield(42)

    coro = example()
    assert 42 == coro.send(None)
    contexts = _frames.contexts_active_in_frame(coro.cr_frame)
    assert all(not c.is_async and not c.is_exiting for c in contexts)
    assert [(c.manager.val, c.varname) for c in contexts] == [
        ((1, 2), "(a, b)"),
        ((3,), "(c,)"),
        (4, "ns.foo"),
        (5, "ns.sub.foo"),
        (6, "dct['one']"),
        (7, "dct['sub']['two']"),
        (8, "dct.__getitem__('sub')['three']"),
        (9, "locals()['ns'].bar"),
        (10, None),  # no target
        (11, None),  # unsupported target
    ]
    coro.close()

    if sys.version_info < (3, 9):
        # spurious SyntaxError on 3.9: https://bugs.python.org/issue41979
        ns = {}
        exec(
            "async def example():\n"
            "    with C(range(4)) as (first, *rest, last):\n"
            "        assert tuple(range(4)) == (first, *rest, last)\n"
            "        await async_yield(42)",
            {"C": C, **globals()},
            ns,
        )
        coro = ns["example"]()
        assert 42 == coro.send(None)
        assert _frames.contexts_active_in_frame(coro.cr_frame) == [
            _frames.ContextInfo(
                is_async=False,
                manager=C(range(4)),
                varname="(first, *rest, last)",
                start_line=2,
            ),
        ]
        coro.close()

    if sys.version_info >= (3, 8):
        ns = {}
        exec(
            "async def example():\n"
            "    with C(1) as locals()[(v := 'hi')]:\n"
            "        await async_yield(42)",
            {"C": C, **globals()},
            ns,
        )
        coro = ns["example"]()
        assert 42 == coro.send(None)
        assert _frames.contexts_active_in_frame(coro.cr_frame) == [
            _frames.ContextInfo(
                is_async=False, manager=C(1), varname=None, start_line=2
            ),
        ]
        coro.close()
