import os
import sys
import platform

from datetime import datetime
from setuptools import find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


def _fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]

def _fetch_readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()

def _fetch_package_name():
    return "charactertraining"

# Custom wheel class to modify the wheel name
class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abi_tag = f"{python_version}"

        if platform.system() == "Linux":
            platform_tag = "manylinux1_x86_64"
        else:
            platform_tag = platform.system().lower()

        return python_version, abi_tag, platform_tag

# Setup configuration
setup(
    author="sm2783",
    name=_fetch_package_name(),
    version="0.0.1",
    packages=find_packages(
        exclude=(
            "data",
            "docs",
            "examples",
        )
    ),
    description="The Claude Character Training framework on Open-Weights Models",
    long_description=_fetch_readme(),
    long_description_content_type="text/markdown",
    install_requires=_fetch_requirements("requirements.txt"),
    extras_require={
        "vllm": ["vllm==0.6.5"],
        "vllm_latest": ["vllm>0.6.5"],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    cmdclass={"bdist_wheel": bdist_wheel},
)
