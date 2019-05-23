from setuptools import setup, find_packages

setup(
    name="YASET",
    version="0.1",
    author="Julien Tourille",
    author_email="julien.tourille@cea.fr",
    description="Yet Another SEquence Tagger",
    url="https://github.com/jtourille/yaset",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=['bin/yaset']
)
