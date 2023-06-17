from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def load_requirements(filename: str) -> List[str]:
    with open(filename, "r") as f:
        reqs = f.read().splitlines()
    return reqs


basic_reqs = load_requirements("requirements/basic.txt")
gpu_extra_reqs = load_requirements("requirements/gpu.txt")


setup(
    # technical things
    version="0.1.0",
    packages=find_packages(exclude=['data', 'docs', 'legacy', 'first_breaks._pytorch', 'tests', "requirements"]),
    python_requires=">=3.7,<4.0",
    install_requires=basic_reqs,
    extras_require={'gpu': gpu_extra_reqs},
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    entry_points={
            "console_scripts": [
                "first-breaks-picking=first_breaks.cli:cli_commands"
            ],
        },
    # general information
    name="first-breaks-picking",
    description="Tool for picking first breaks in seismic gather",
    author="Aleksei Tarasov",
    author_email="aleksei.v.tarasov@gmail.com",
    url="https://github.com/DaloroAT/first_breaks_picking",
    keywords=[
        "seismic",
        "first-breaks",
        "computer-vision",
        "deep-learning",
        "segmentation",
        "data-science"
    ],
    classifiers=[
            "Environment :: Console",
            "Environment :: X11 Applications :: Qt",
            "Operating System :: OS Independent",
            "License :: OSI Approved :: Apache Software License",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Image Recognition",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
    project_urls={"Homepage": "https://github.com/DaloroAT/first_breaks_picking"},
    license="Apache License 2.0",
)
