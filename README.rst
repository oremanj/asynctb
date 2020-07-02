asynctb: traceback introspection for Python async programming
=============================================================

This is a small library which helps you get tracebacks of your async
tasks.  It can handle a number of different types of awaitables, and
can traceback the running task as well as sleeping ones. Most notably,
its tracebacks include a frame for each active context manager, which
is useful if you want to draw a detailed Trio task tree, or even if
you just want to understand which resources a task will be cleaning up.

``asynctb`` itself is framework-agnostic: it operates on coroutine
objects, not tasks per se. Call
``asynctb.get_traceback(my_trio_task.coro)`` or
``asynctb.get_traceback(my_asyncio_task.get_coro())`` to get an
iterator that yields ``asynctb.FrameInfo`` objects from the outside
in. You can also pass ``get_traceback()`` a generator, async
generator, and a few other more obscure things.

This is still in development, and has undergone only light manual testing so far.
It will get real tests and docs before it's released.

License: Your choice of MIT or Apache License 2.0
