#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union

from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).parent.resolve()


def load_requirements(filename: Union[Path, str] = "requirements.txt"):
    with open(PROJECT_ROOT / filename) as f:
        return f.read().splitlines()


def load_readme(filename: Union[Path, str] = "README.md") -> str:
    with open(PROJECT_ROOT / filename, encoding="utf-8") as f:
        return f"\n{f.read()}"


def load_version(filename: Union[Path, str]) -> str:
    context = {}
    with open(PROJECT_ROOT / filename) as f:
        exec(f.read(), context)
    return context["__version__"]


setup(
    name="esrgan",
    version=load_version(Path("esrgan", "__version__.py")),
    description=(
        "Implementation of paper"
        " `ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks`"
    ),
    long_description=load_readme("README.md"),
    long_description_content_type="text/markdown",
    license="Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)",  # noqa: E501
    author="Yauheni Kachan",
    author_email="yauheni.kachan@leverx.com",
    python_requires=">=3.8.0",
    url="esrgan/experiment/config.yml",
    install_requires=load_requirements("requirements.txt"),
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "esrgan-process-images=esrgan.utils.scripts.process_images:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
    ],
)
