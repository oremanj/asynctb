from setuptools import setup, find_packages

exec(open("asynctb/_version.py", encoding="utf-8").read())

LONG_DESC = open("README.rst", encoding="utf-8").read()

setup(
    name="asynctb",
    version=__version__,
    description="Traceback utilities for async programming in Python",
    url="https://github.com/oremanj/asynctb",
    long_description=LONG_DESC,
    author="Joshua Oreman",
    author_email="oremanj@gmail.com",
    license="MIT -or- Apache License 2.0",
    packages=find_packages(),
    include_package_data=True,
    keywords=["async", "debugging", "trio", "asyncio"],
    python_requires=">=3.6",
    install_requires=["attrs >= 19.2"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
    ],
)
