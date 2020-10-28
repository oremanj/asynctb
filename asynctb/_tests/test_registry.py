import sys
import threading
from functools import partial, wraps

import attr
import pytest

import asynctb


def test_identity_dict():
    from asynctb._registry import IdentityDict

    @attr.s(eq=True)
    class Cell:
        value = attr.ib()

    c1 = Cell(10)
    c2 = Cell(10)
    idict = IdentityDict()
    idict.update([(c1, 1), (c2, 2)])
    rev = IdentityDict([(c2, 2), (c1, 1)])

    # equality, order preservation, repr
    assert idict == IdentityDict([(c1, 1), (c2, 2)])
    assert IdentityDict([(1, 2), (3, 4)]) == {1: 2, 3: 4}
    assert idict == rev
    assert idict != IdentityDict([(c2, 1), (c1, 2)])
    assert repr(idict) == "IdentityDict([(Cell(value=10), 1), (Cell(value=10), 2)])"
    assert len(idict) == 2
    assert list(map(id, iter(idict))) == [id(c1), id(c2)]

    # item lookup
    assert idict[c1] == rev[c1] == 1
    assert idict[c2] == rev[c2] == 2
    with pytest.raises(KeyError):
        idict[Cell(10)]
    c1.value = 100
    assert idict[c1] == 1

    # item assignment & deletion
    idict[c1] = 11
    assert idict != rev
    assert list(idict.values()) == [11, 2]
    del idict[c2]
    assert len(idict) == 1
    assert list(idict.items()) == [(Cell(100), 11)]
    assert next(iter(idict.items()))[0] is c1
    idict.clear()
    assert len(idict) == 0

    idict.setdefault(c2, 10)
    idict.setdefault(c2, 20)
    idict.setdefault(Cell(100))
    assert list(idict.items()) == [(c2, 10), (Cell(100), None)]
    assert idict.popitem() == (Cell(100), None)
    assert idict.pop("nope", 42) == 42
    assert idict.pop(c2, 99) == 10
    with pytest.raises(KeyError):
        idict.pop(c2)


# This get_target implementation pretends that every frame it's registered on
# that has a local or argument named "magic_arg" is actually the runner for
# a generator called "fake_target". The generator receives the value of magic_arg
# as its argument "arg".
def simple_get_target(frame, is_terminal):
    if "magic_arg" in frame.f_locals:  # pragma: no branch

        def fake_target(arg):
            yield

        gen = fake_target(frame.f_locals["magic_arg"])
        gen.send(None)
        return gen


def current_frame_uses_registered_get_target():
    tb = asynctb.Traceback.since(sys._getframe(1))
    return tb.frames[1].funcname == "fake_target"


def test_registration_through_functools_wraps_or_partial(isolated_registry):
    def example(magic_arg):
        return asynctb.Traceback.since(sys._getframe(0))

    @wraps(example)
    def wrapper(something):
        return example(something)

    bound_example = partial(wrapper, 10)
    asynctb.customize(bound_example, get_target=simple_get_target)

    tb = bound_example()
    assert len(tb.frames) == 2
    assert [f.funcname for f in tb.frames] == ["example", "fake_target"]
    assert tb.frames[-1].frame.f_locals["arg"] == 10


def test_registration_through_code_object(isolated_registry):
    def code_example(magic_arg):
        return asynctb.Traceback.since(sys._getframe(0))

    asynctb.customize(code_example.__code__, get_target=simple_get_target)
    tb = code_example(100)
    assert [f.funcname for f in tb.frames] == ["code_example", "fake_target"]


def test_registration_through_unsupported():
    with pytest.raises(TypeError, match="extract a code object"):
        asynctb.customize(42, skip_frame=True)


def test_registration_through_method(isolated_registry):
    class C:
        @asynctb.customize(get_target=simple_get_target)
        def instance(self, magic_arg):
            assert current_frame_uses_registered_get_target()
            return "instance", self

        @asynctb.customize(get_target=simple_get_target)
        @staticmethod
        def static(magic_arg):
            assert current_frame_uses_registered_get_target()
            return "static", None

        @asynctb.customize(get_target=simple_get_target)
        @classmethod
        def class_(cls, magic_arg):
            assert current_frame_uses_registered_get_target()
            return "class", cls

    c = C()
    assert c.instance(1) == C.instance(c, 2) == ("instance", c)
    assert c.class_(1) == C.class_(2) == ("class", C)
    assert c.static(1) == C.static(2) == ("static", None)


def test_registration_through_nested(isolated_registry):
    def outer_fn(magic_arg):
        def middle_fn():
            def inner_fn():
                assert magic_arg == 1
                return current_frame_uses_registered_get_target()

            assert magic_arg == 1
            assert current_frame_uses_registered_get_target()
            return inner_fn

        assert not current_frame_uses_registered_get_target()
        return middle_fn

    asynctb.customize(outer_fn, "middle_fn", get_target=simple_get_target)
    with pytest.raises(ValueError, match="function or class named"):
        asynctb.customize(outer_fn, "magic_arg")
    with pytest.raises(ValueError, match="function or class named"):
        asynctb.customize(outer_fn, "nope")

    middle_fn = outer_fn(1)
    inner_fn = middle_fn()
    assert not inner_fn()
    asynctb.customize(outer_fn, "middle_fn", "inner_fn", get_target=simple_get_target)
    assert inner_fn()


def test_install_concurrently(isolated_registry, monkeypatch):
    def one_fn():
        return asynctb.Traceback.since(sys._getframe(0)).frames

    def two_fn():
        return asynctb.Traceback.since(sys._getframe(0)).frames

    one_started = threading.Event()
    two_waiting = threading.Event()
    two_started = threading.Event()
    record = []

    def glue_one():
        record.append("one")
        one_started.set()
        two_started.wait()
        asynctb.customize(one_fn, skip_frame=True)

    def glue_two():
        record.append("two")
        two_waiting.set()
        one_started.wait()
        two_started.set()
        asynctb.customize(two_fn, skip_frame=True)

    assert one_fn()
    assert two_fn()

    monkeypatch.setattr(asynctb._glue, "INSTALLED", False)
    asynctb._glue.glue_two = glue_two
    asynctb._glue.glue_one = glue_one

    # Ordering here: t2 selects glue_two first (since it was added to the globals
    # dict first) and blocks there. Then t1 selects glue_one, and the two
    # race to register their rules simultaneously. t2 will later try to select
    # glue_one and find it already taken by another thread.

    t2 = threading.Thread(target=asynctb._glue.ensure_installed, daemon=True)
    t2.start()
    two_waiting.wait()
    t1 = threading.Thread(target=asynctb._glue.ensure_installed, daemon=True)
    t1.start()
    t1.join()
    t2.join()
    assert record == ["two", "one"]

    assert not one_fn()
    assert not two_fn()


def test_glue_error(isolated_registry, monkeypatch):
    def glue_whoops():
        raise ValueError("guess not")

    monkeypatch.setattr(asynctb._glue, "INSTALLED", False)
    asynctb._glue.glue_whoops = glue_whoops

    with pytest.warns(RuntimeWarning, match="Failed to initialize glue for whoops"):
        asynctb._glue.ensure_installed()
