#!/usr/bin/env python
import codecs
import os
import re

from setuptools import find_packages, setup

__version__ = '0.1'

# PROJECT SPECIFIC

NAME = "sedflow"
PACKAGES = find_packages(where="src")
META_PATH = os.path.join("src", "sedflow", "__init__.py")
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
]
INSTALL_REQUIRES = [
        "numpy>1.18",
]
# END PROJECT SPECIFIC


HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


with open("README.md", "r") as fh:
    long_description = fh.read()


if __name__ == "__main__":
    setup(
        name=NAME,
        version = __version__,
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        url=find_meta("uri"),
        license=find_meta("license"),
        description=find_meta("description"),
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=PACKAGES,
        package_dir={"": "src"},
        include_package_data=True,
        package_data={'sedflow': ['dat/*pt']},
        install_requires=INSTALL_REQUIRES,
        classifiers=CLASSIFIERS,
        zip_safe=True,
    )
