from pathlib import Path

from setuptools import find_packages, setup


basic_requirements = [
    "requests==2.28.2",
    "numpy==1.24.2",
    "pandas==2.0.0",
    "PyQt5==5.15.9",
    "onnxruntime==1.14.1",
    "pyqtgraph==0.13.3",
    "tqdm==4.65.0",
    "click==8.1.3"
]

torch_requirements = [
    "torch==2.0.1",
    "torchvision==0.15.2"
]


setup(
    # technical things
    version="0.1.0",
    packages=find_packages(exclude=['data', 'docs', 'legacy']),
    python_requires=">=3.7,<4.0",
    install_requires=basic_requirements,
    extras_require={'torch': torch_requirements},
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
