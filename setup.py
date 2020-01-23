# coding=utf-8

from setuptools import setup, find_packages

reqs = [
    "tensorflow==1.15.0",
    "torch==1.4.0",
    "torchvision==0.5.0",
    "scikit-learn==0.22.1",
    "pandas==0.25.3",
    "numpy==1.18.1",
    "scipy==1.4.1",
    "attrs==19.3.0",
    "flatten_dict==0.2.0"
]

setup(
    name="allRank",
    version="1.2.0",
    license="Apache 2",
    url="https://github.com/allegro/allRank",
    install_requires=reqs,
    author_email="allrank@allegro.pl",
    description="allRank is a framework for training learning-to-rank neural models",
    packages=find_packages(exclude=["tests"]),
    package_data={"allrank": ["config.json"]},
    entry_points={"console_scripts": ['allRank = allrank.main:run']},
    zip_safe=False,
)
