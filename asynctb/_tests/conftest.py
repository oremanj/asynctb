import pytest
from asynctb._registry import HANDLING_FOR_CODE
from asynctb._glue import ensure_installed


@pytest.fixture
def local_registry():
    ensure_installed()
    prev_contents = list(HANDLING_FOR_CODE.items())
    yield
    HANDLING_FOR_CODE.clear()
    HANDLING_FOR_CODE.update(prev_contents)


@pytest.fixture
def isolated_registry(local_registry):
    HANDLING_FOR_CODE.clear()
