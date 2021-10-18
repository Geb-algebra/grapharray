# Author: Geb
# Copyright (c) 2021 Geb
# License: MIT License

from setuptools import setup
import grapharray

DESCRIPTION = "GraphArray : Python package for treating arrays defined on a network, which allows for fast computation and easy visualization."
NAME = "grapharray"
AUTHOR = "Geb"
LICENSE = "MIT License"
PROJECT_URLS = {
    "Documentation": "https://geb-algebra.github.io/grapharray/",
    "Source Code": "https://github.com/Geb-algebra/grapharray",
}
VERSION = grapharray.__version__
PYTHON_REQUIRES = ">=3.8"

INSTALL_REQUIRES = ["networkx>=2.6.2", "numpy>=1.20.3", "scipy>=1.7.1"]

PACKAGES = ["grapharray"]

CLASSIFIERS = [
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
]

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name=NAME,
    author=AUTHOR,
    maintainer=AUTHOR,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=LICENSE,
    project_urls=PROJECT_URLS,
    version=VERSION,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    packages=PACKAGES,
    classifiers=CLASSIFIERS,
)
