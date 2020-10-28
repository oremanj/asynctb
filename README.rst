asynctb: stack introspection for Python async programming
=========================================================

``asynctb`` is a library that helps you get tracebacks of parts of your
running Python program. It was originally designed with async tasks in
mind (thus the name), but also has some support for threads, greenlets,
and ordinary synchronous code. It is loosely affiliated with the `Trio
<https://trio.readthedocs.io/>`__ async framework, and shares Trio's
obsessive focus on usability and correctness. You don't have to use it
with Trio, though; its only package dependency is
`attrs <https://www.attrs.org/>`__.

This is mostly intended as a building block for other debugging and
introspection tools. You can use it directly, but there's only
rudimentary support for end-user-facing niceties such as
pretty-printed output. On the other hand, the core logic is
extremely flexible:

* Tracebacks can be extracted for generators (regular and async),
  coroutine objects, greenlets, threads, and ranges of frame objects.

* Other awaitables can be handled by defining an "unwrapper", which
  takes an awaitable of that type and returns a coroutine or generator.
  Several are supported out of the box, for async generator ``asend()``
  and ``athrow()`` methods and ``coroutine_wrapper`` objects.
  Third-party packages can add more.

* Tracebacks can include information about what context managers are
  active in each frame, including references to the context manager
  objects themselves. For example, this can be used to draw a
  detailed Trio task tree. The logic knows how to look inside
  ``@contextmanager``, ``ExitStack``, and their async equivalents
  (both those in the stdlib and their popular backports in
  ``async_generator`` and ``async_exit_stack``).
  Third-party packages can define context manager unwrappers too.

* There are a number of customization points allowing third-party
  packages to specify that some of their functions and/or those
  functions' callees should be excluded from ``asynctb`` tracebacks,
  or should incorporate the traceback of another object that
  ``asynctb`` knows how to handle. For example, the traceback of a
  Trio task blocked in ``trio.to_thread.run_sync()`` could cover the
  code that's running in the thread as well.

And of course, if you want to disable all of this and just have a
version of ``inspect.stack()`` that takes a coroutine object, you can.

``asynctb`` requires Python 3.6 or later. It is tested with CPython
(every minor version through 3.10-dev) and PyPy, on Linux, Windows,
and macOS. It will probably work on other operating systems.
Basic features will work on other interpreters, but the context
manager decoding will be less intelligent, and won't work at all
without a usable ``gc.get_referents()``.

This is still in development. It has full test coverage, but
will likely undergo some incompatible API changes before an initial
release. Documentation is also currently light. Watch this space!

License: Your choice of MIT or Apache License 2.0
