import os
from setuptools import setup


setup(
    name = "ddpg",
    version = "0.0.1",
    author = "Hamza Merzic",
    author_email = "hamzamerzic@gmail.com",
    description = ("Deep Deterministic Policy Gradient algorithm."),
    license = "MIT",
    url = "https://github.com/hamzamerzic/ddpg",
    packages=['ddpg'],
    classifiers=[
        "Environment :: Console",
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License"
    ],
)
