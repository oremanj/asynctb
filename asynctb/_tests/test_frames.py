import contextlib
import dis
import sys
import types

import attr
import pytest  # type: ignore

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
        with contextlib.ExitStack() as stack:
            async with YieldsDuringAexit():
                with contextlib.ExitStack() as stack2:
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
        with contextlib.ExitStack() as stack:  # pragma: no cover
            pass

    coro = afn()
    assert _frames.contexts_active_in_frame(coro.cr_frame) == []
    coro.close()


def test_aexit_extended_arg():
    # Create a function that uses 256 constants before its first None,
    # so that when it does LOAD_CONST None in __aexit__ it needs an
    # EXTENDED_ARG.  Note that such a function requires a docstring,
    # because the first constant is the docstring and with no
    # docstring there's your None.
    lines = [
        "async def example():",
        "    '''Docstring.'''",
        "    lst = " + repr(list(range(1000))),
        "    async with YieldsDuringAexit():",
        "        pass",
    ]
    ns = {}
    exec("\n".join(lines), globals(), ns)
    coro = ns["example"]()
    assert 42 == coro.send(None)
    assert _frames.contexts_active_in_frame(coro.cr_frame) == [
        _frames.ContextInfo(is_async=True, is_exiting=True, manager=None, start_line=4),
    ]
    coro.close()


@pytest.mark.skipif(sys.version_info < (3, 8), reason="uses CodeType.replace()")
def test_unexpected_aexit_sequence():
    async def example():
        async with YieldsDuringAexit():
            pass

    op = dis.opmap
    pattern = [op["WITH_CLEANUP_START"], 0, op["GET_AWAITABLE"], 0, op["LOAD_CONST"], 0]
    co = example.__code__
    example.__code__ = co.replace(
        co_code=co.co_code.replace(
            bytes(pattern),
            bytes(pattern[:2] + [op["LOAD_CONST"], 0, op["POP_TOP"], 0] + pattern[2:]),
        ),
    )
    coro = example()
    assert 42 == coro.send(None)
    assert _frames.contexts_active_in_frame(coro.cr_frame) == []
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
                    with C(range(8, 12)) as (first, *rest, last):
                        with C(12) as dct.__getitem__("sub")["three"]:
                            with C(13) as locals()["ns"].bar, C(14):
                                with C(15) as locals()[b"x".decode("ascii") + "y"]:
                                    assert (a, b, c) == (1, 2, 3)
                                    assert ns.foo == 4 and ns.sub.foo == 5
                                    assert ns.bar == 13
                                    assert dct["one"] == 6
                                    assert dct["sub"] == {"two": 7, "three": 12}
                                    assert (first, *rest, last) == (8, 9, 10, 11)
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
        (range(8, 12), "(first, *rest, last)"),
        (12, "dct.__getitem__('sub')['three']"),
        (13, "locals()['ns'].bar"),
        (14, None),  # no target
        (15, None),  # unsupported target
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
