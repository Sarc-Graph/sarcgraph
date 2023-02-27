from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sarcgraph",
    version="0.1.2",
    author="Saeed Mohammadzadeh",
    author_email="saeedmhz@bu.edu",
    description="A software for sarcomere detection and tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sarc-Graph/sarcgraph",
    packages=["sarcgraph"],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
