from typing import Sequence

class GcRef: ...

class GcCollectStepStats:
    count: int
    duration: float
    duration_min: float
    duration_max: float
    oldstate: int
    newstate: int
    major_is_done: bool

def get_typeids_z() -> bytes: ...
def get_typeids_list() -> Sequence[int]: ...
def get_rpy_type_index(obj: object) -> int: ...
def get_rpy_referents(obj: object) -> Sequence[object]: ...
def collect_step() -> GcCollectStepStats: ...
