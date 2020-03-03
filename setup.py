# coding=utf-8

import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

reqs = [
    "torch==1.4.0",
    "torchvision==0.5.0",
    "scikit-learn==0.22.1",
    "pandas==0.25.3",
    "numpy==1.18.1",
    "scipy==1.4.1",
    "attrs==19.3.0",
    "flatten_dict==0.2.0",
    "tensorboardX==2.0.0",
    "gcsfs==0.6.0"
]

setup(
    name="allRank",
    version="1.2.1",
    description="allRank is a framework for training learning-to-rank neural models",
    long_description=README,
    long_description_content_type="text/markdown",
    license="Apache 2",
    url="https://github.com/allegro/allRank",
    install_requires=reqs,
    author_email="allrank@allegro.pl",
    packages=find_packages(exclude=["tests"]),
    package_data={"allrank": ["config.json"]},
    entry_points={"console_scripts": ['allRank = allrank.main:run']},
    zip_safe=False,
)
