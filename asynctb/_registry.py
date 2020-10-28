import attr
import functools
import inspect
import types
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

GetTargetFn = Callable[[types.FrameType, Optional[types.FrameType]], Any]
RegisterFn = Callable[[type], Callable[[Callable[[Any], Any]], Callable[[Any], Any]]]

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
F_get_target = TypeVar("F_get_target", bound=GetTargetFn)
F = TypeVar("F", bound=Callable[..., Any])


def make_unwrapper(name: str) -> Tuple[Callable[[Any], Any], RegisterFn]:
    @functools.singledispatch
    def unwrap_once(thing: Any) -> Any:
        return None

    def unwrap_repeatedly(thing: Any) -> Any:
        while True:
            unwrapped = unwrap_once(thing)
            if unwrapped is None:
                return thing
            thing = unwrapped

    for attrib in ("__name__", "__qualname__"):
        setattr(unwrap_repeatedly, attrib, "unwrap_" + name)
        setattr(unwrap_once.register, attrib, "register_unwrap_" + name)

    return unwrap_repeatedly, cast(RegisterFn, unwrap_once.register)


unwrap_awaitable, register_unwrap_awaitable = make_unwrapper("awaitable")
unwrap_context_manager, register_unwrap_context_manager = make_unwrapper(
    "context_manager"
)


class IdentityDict(MutableMapping[K, V]):
    """A dict that hashes objects by their identity, not their contents.

    We use this to track code objects, since they have an expensive-to-compute
    hash which is not cached. You can probably think of other uses too.

    Single item lookup, assignment, deletion, and :meth:`setdefault` are
    thread-safe because they are each implented in terms of a single call to
    a method of an underlying native dictionary object.
    """

    __slots__ = ("_data",)

    def __init__(self, items: Iterable[Tuple[K, V]] = ()):
        self._data = {id(k): (k, v) for k, v in items}

    def __repr__(self) -> str:
        return "IdentityDict([{}])".format(
            ", ".join(f"({k!r}, {v!r})" for k, v in self._data.values())
        )

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[K]:
        return (k for k, v in self._data.values())

    def __getitem__(self, key: K) -> V:
        return self._data[id(key)][1]

    def __setitem__(self, key: K, value: V) -> None:
        self._data[id(key)] = key, value

    def __delitem__(self, key: K) -> None:
        del self._data[id(key)]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IdentityDict):
            return self._data == other._data
        return super().__eq__(other)

    _marker = object()

    @overload
    def pop(self, key: K) -> V:
        ...

    @overload
    def pop(self, key: K, default: Union[V, T] = ...) -> Union[V, T]:
        ...

    def pop(self, key: K, default: object = _marker) -> object:
        try:
            return self._data.pop(id(key))[1]
        except KeyError:
            if default is self._marker:
                raise KeyError(key) from None
            return default

    def popitem(self) -> Tuple[K, V]:
        return self._data.popitem()[1]

    def clear(self) -> None:
        self._data.clear()

    def setdefault(self, key: K, default: V = cast(V, None)) -> V:
        return self._data.setdefault(id(key), (key, default))[1]


@attr.s(auto_attribs=True, slots=True, eq=False)
class CodeHandling:
    skip_frame: bool = False
    skip_callees: bool = False
    get_target: Optional[GetTargetFn] = None


HANDLING_FOR_CODE = IdentityDict[types.CodeType, CodeHandling]()


def get_code(thing: object, *nested_names: str) -> types.CodeType:
    while True:
        if isinstance(thing, functools.partial):
            thing = thing.func
            continue
        if isinstance(thing, (types.MethodType, classmethod, staticmethod)):
            thing = thing.__func__
            continue
        if hasattr(thing, "__wrapped__"):
            thing = inspect.unwrap(cast(types.FunctionType, thing))
            continue
        break

    code: types.CodeType
    if isinstance(thing, types.FunctionType):
        code = thing.__code__
    elif isinstance(thing, types.CodeType):
        code = thing
    else:
        raise TypeError(f"Don't know how to extract a code object from {thing!r}")

    top_name = code.co_name
    for idx, name in enumerate(nested_names):
        for const in code.co_consts:
            if isinstance(const, types.CodeType) and const.co_name == name:
                code = const
                break
        else:
            raise ValueError(
                f"Couldn't find a function or class named {name!r} in "
                + ".".join([top_name, *nested_names[:idx]])
            )

    return code


@overload
def customize(
    *,
    skip_frame: Optional[bool] = None,
    skip_callees: Optional[bool] = None,
    get_target: Optional[GetTargetFn] = None,
) -> Callable[[F], F]:
    ...


@overload
def customize(
    thing: Union[Callable[..., Any], types.CodeType],
    *nested_names: str,
    skip_frame: Optional[bool] = None,
    skip_callees: Optional[bool] = None,
    get_target: Optional[GetTargetFn] = None,
) -> None:
    ...


def customize(
    thing: object = None,
    *nested_names: str,
    skip_frame: Optional[bool] = None,
    skip_callees: Optional[bool] = None,
    get_target: Optional[GetTargetFn] = None,
) -> object:
    if thing is None:

        def decorate(fn: F) -> F:
            customize(
                fn,
                skip_frame=skip_frame,
                skip_callees=skip_callees,
                get_target=get_target,
            )
            return fn

        return decorate

    code = get_code(thing, *nested_names)
    handling = HANDLING_FOR_CODE.setdefault(code, CodeHandling())
    if skip_frame is not None:
        handling.skip_frame = skip_frame
    if skip_callees is not None:
        handling.skip_callees = skip_callees
    if get_target is not None:
        handling.get_target = get_target
    return None


def register_get_target(
    fn: Callable[..., Any]
) -> Callable[[F_get_target], F_get_target]:
    def decorate(get_target: F_get_target) -> F_get_target:
        customize(fn, get_target=get_target)
        return get_target

    return decorate
