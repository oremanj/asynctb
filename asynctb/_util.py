import attr
from typing import Dict, Iterable, Iterator, MutableSet, TypeVar

T = TypeVar("T")


@attr.s(eq=True, slots=True, auto_attribs=True)
class IdentitySet(MutableSet[T]):
    """A set that hashes objects by their identity, not their contents.

    We use this to track code objects, since they have an expensive-to-compute
    hash which is not cached. You can probably think of other uses too.
    """

    _data: Dict[int, T] = attr.ib(
        default=(), converter=lambda xs: {id(x): x for x in xs}
    )

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        return iter(self._data.values())

    def __contains__(self, val: object) -> bool:
        return id(val) in self._data

    def __le__(self, other: object) -> bool:
        if isinstance(other, IdentitySet):
            return self._data.keys() <= other._data.keys()
        return super().__le__(other)

    def __ge__(self, other: object) -> bool:
        if isinstance(other, IdentitySet):
            return self._data.keys() >= other._data.keys()
        return super().__ge__(other)

    def add(self, val: T) -> None:
        self._data.setdefault(id(val), val)

    def discard(self, val: T) -> None:
        self._data.pop(id(val), None)

    def remove(self, val: T) -> None:
        try:
            del self._data[id(val)]
        except KeyError:
            raise KeyError(val) from None

    def pop(self) -> T:
        return self._data.popitem()[1]

    def clear(self) -> None:
        self._data.clear()

    def update(self, vals: Iterable[T]) -> None:
        for val in vals:
            self.add(val)
