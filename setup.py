from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sarcgraph",
    version="0.2.1",
    author="Saeed Mohammadzadeh",
    author_email="saeedmhz@bu.edu",
    description="A software for sarcomere detection and tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/sarcgraph/",
    project_urls={
        "Source": "https://github.com/Sarc-Graph/sarcgraph",
        "Documentation": "https://sarc-graph.readthedocs.io/en/latest/",
    },
    maintainer=[
        ("Saeed Mohammadzadeh", "saeedmhz@bu.edu"),
        ("Emma Lejeune", "elejeune@bu.edu"),
    ],
    packages=["sarcgraph"],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "matplotlib==3.5.2",
        "networkx==2.8.4",
        "numpy==1.23.5",
        "pandas==1.5.2",
        "scikit-image==0.19.3",
        "scikit-learn==1.2.1",
        "scipy==1.10.0",
        "sk-video==1.1.10",
        "trackpy==0.6.1",
    ],
    zip_safe=False,
)
