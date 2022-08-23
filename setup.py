from setuptools import find_packages
from distutils.core import setup

setup(
    name="gerry",
    version="0.0",
    package_dir={"gerry": "gerry"},
    packages=["gerry", "gerry.utils"],
    requires=["scipy", "numpy"],
)
