import asynctb
import types

async def three():
  await two()

async def two():
  async with one():
    await async_yield("body")

from contextlib import asynccontextmanager
@asynccontextmanager
async def one():
  await async_yield("before")
  yield
  await async_yield("after")

@types.coroutine
def async_yield(x): yield x

coro = three()
coro.send(None)
print(list(asynctb.get_traceback(coro)))
