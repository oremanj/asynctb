import ctypes
import dis
import gc
import sys
from types import FrameType
from typing import Dict, Iterator, List, Optional, Sequence, Type, TYPE_CHECKING
from ._frames import FrameDetails


# Every interpreter-level type in pypy has a description, index, and id.
# The description is something human-readable, possibly with spaces.
# The id is a unique half-word value (2 bytes on 32-bit, 4 bytes on 64-bit)
# stored in the bottom half of the first word in the instance representation;
# it indicates the offset of the type-info structure in a global array.
# The index is a denser unique value used in some other places.
#
# More details on pypy object representation:
# https://github.com/oremanj/asynctb/issues/1

_pypy_type_desc_from_index: List[str] = []
_pypy_type_index_from_id: Dict[int, int] = {}


if TYPE_CHECKING:
    # typeshed doesn't include the pypy-specific gc methods
    from . import _pypy_gc_stubs as pgc
else:
    pgc = gc


def _fill_pypy_typemaps() -> None:
    assert sys.implementation.name == "pypy"
    import zlib

    # The first few lines of get_typeids_z(), after decompression, look like:
    # member0    ?
    # member1    GcStruct JITFRAME { jf_frame_info, jf_descr, jf_force_descr, [...] }
    # member2    GcStruct rpy_string { hash, chars }
    # member3    GcStruct rpy_unicode { hash, chars }
    for line in zlib.decompress(pgc.get_typeids_z()).decode("ascii").splitlines():
        memberNNN, rest = line.split(None, 1)
        header, brace, fields = rest.partition(" { ")
        _pypy_type_desc_from_index.append(header)

    for idx, typeid in enumerate(pgc.get_typeids_list()):
        _pypy_type_index_from_id[typeid] = idx


if sys.implementation.name == "pypy":
    _fill_pypy_typemaps()


def _pypy_typename(obj: object) -> str:
    """Return the pypy interpreter-level type name of the type of *obj*.

    *obj* may be an ordinary Python object, or may be a gc.GcRef to inspect
    something that is not manipulatable at app-level.
    """
    return _pypy_type_desc_from_index[pgc.get_rpy_type_index(obj)]



def _pypy_typename_from_first_word(first_word: int) -> str:
    """Return the pypy interpreter-level type name of the type of the instance
    whose first word in memory has the value *first_word*.
    """
    if sys.maxsize > 2**32:
        mask = 0xffffffff
    else:
        mask = 0xffff
    return _pypy_type_desc_from_index[_pypy_type_index_from_id[first_word & mask]]


def inspect_frame(frame: FrameType) -> FrameDetails:
    assert sys.implementation.name == "pypy"

    # Somewhere in the list of immediate referents of the frame is its
    # code object.
    frame_refs = pgc.get_rpy_referents(frame)
    code_idx, = [idx for idx, ref in enumerate(frame_refs) if ref is frame.f_code]

    # The two referents immediately before the code object are
    # the last entry in the block list, followed by the value stack.
    # These are interp-level objects so we see them as opaque GcRefs.
    # We locate them by reference to the code object because the
    # earlier references might or might not be present (e.g., one depends
    # on whether the frame's f_locals have been accessed yet or not).
    assert code_idx >= 1
    valuestack_ref = frame_refs[code_idx - 1]
    assert isinstance(valuestack_ref, pgc.GcRef)

    lastblock_ref: Optional[pgc.GcRef] = None
    if code_idx >= 2:
        candidate = frame_refs[code_idx - 2]
        if "Block" not in _pypy_typename(candidate):
            # There are no blocks active in this frame. lastblock was
            # skipped when getting referents because it's null, so the
            # previous field (generator weakref or f_back) bled through.
            assert (
                _pypy_typename(candidate) == "GcStruct weakref"
                or "Frame" in _pypy_typename(candidate)
            )
        else:
            assert isinstance(candidate, pgc.GcRef)
            lastblock_ref = candidate

    # The value stack's referents are everything on the value stack.
    # Unfortunately we can't rely on the indices here because 'del x'
    # leaves a null (not None) that will be skipped. We'll fill them
    # in from ctypes later. Note that this includes locals/cellvars/
    # freevars (at the start, in that order).
    valuestack = pgc.get_rpy_referents(valuestack_ref)

    # The block list is a linked list in PyPy, unlike in CPython where
    # it's an array. The head of the list is the newest block.
    # Iterate through and unroll it into a list of GcRefs to blocks.
    blocks: List[pgc.GcRef] = []
    if lastblock_ref is not None:
        blocks.append(lastblock_ref)
        while True:
            assert len(blocks) < 100
            more = pgc.get_rpy_referents(blocks[-1])
            if not more:
                break
            for ref in more:
                assert isinstance(ref, pgc.GcRef)
                blocks.append(ref)
        assert all("Block" in _pypy_typename(blk) for blk in blocks)
        # Reverse so the oldest block is at the beginning
        blocks = blocks[::-1]
        # Remove those that aren't FinallyBlocks -- those are the
        # only ones we care about (used for context managers too)
        blocks = [blk for blk in blocks if "FinallyBlock" in _pypy_typename(blk)]

    # This seems to be necessary to reliably make the object representations
    # correct before we start peeking at them.
    gc.collect()

    def unwrap_gcref(ref: pgc.GcRef) -> "ctypes.pointer[ctypes.c_ulong]":
        ref_p = ctypes.pointer(ctypes.c_ulong.from_address(id(ref)))
        assert "W_GcRef" in _pypy_typename_from_first_word(ref_p[0].value)
        return ctypes.pointer(ctypes.c_ulong.from_address(ref_p[1].value))

    # Fill in nulls in the value stack. This requires inspecting the
    # memory that backs the list object. An RPython list is two words
    # (typeid, length) followed by one word per element.
    def build_full_stack(refs: Sequence[object]) -> Iterator[object]:
        assert isinstance(valuestack_ref, pgc.GcRef)
        stackdata_p = unwrap_gcref(valuestack_ref)
        assert _pypy_typename_from_first_word(stackdata_p[0].value) == (
            "GcArray of * GcStruct object"
        )
        ref_iter = iter(refs)
        for idx in range(stackdata_p[1].value):
            if stackdata_p[2 + idx].value == 0:
                yield None
            else:
                try:
                    yield next(ref_iter)
                except StopIteration:
                    break

    details = FrameDetails(stack=list(build_full_stack(valuestack)))
    for block_ref in blocks:
        block_p = unwrap_gcref(block_ref)
        assert _pypy_typename_from_first_word(block_p[0].value) == (
            "GcStruct pypy.interpreter.pyopcode.FinallyBlock"
        )
        details.blocks.append(
            FrameDetails.FinallyBlock(handler=block_p[1].value, level=block_p[3].value)
        )
    return details
