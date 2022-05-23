from setuptools import setup


setup(
    name='canal',
    packages=[
        "canal"
    ],
    install_requires=[
        "magma-lang",
        "mantle",
        "hwtypes",
        "kratos",
        "ordered_set",
        "pyverilog"
    ],
)
