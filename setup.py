from setuptools import find_packages, setup

setup(
    # technical things
    version="0.0.1",
    packages=find_packages(exclude=['data', 'docs', 'legacy']),
    python_requires=">=3.7,<4.0",
    # general information
    name="first-breaks-picking",
    description="Tool for picking first breaks in seismic gather",
    author="Aleksei Tarasov",
    author_email="aleksei.v.tarasov@gmail.com",
    url="https://github.com/DaloroAT/first_breaks_picking",
    license="Apache License 2.0",
)
