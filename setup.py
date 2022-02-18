import io
import unittest
from setuptools import find_packages, setup

# Package meta-data
NAME = "jigsaw_puzzle_solver"
DESCRIPTION = "fast jigsaw puzzle solver with unknown orientation"
URL = "https://github.com/hj2choi/fast_jigsaw_puzzle_solver"
EMAIL = "hongjoonchoi95@gmail.com"
AUTHOR = "Hong Joon CHOI"
VERSION = "1.0.0"


# What packages are required for this module to be executed?
REQUIRED = [
    "numpy"
    "opencv-python"
]


# Read in the README ofr the long description of PyPI
def long_description():
    with io.open("README.rst", "r", encoding="utf-8") as f:
        readme = f.read()
    return readme


def jigsaw_puzzle_solver_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("test", pattern="*test.py")


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description(),
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=REQUIRED,
    extras_require={"dev": ["coverage"]},
    test_suite="setup.jigsaw_puzzle_solver_test_suite",
    zip_safe=False,
)