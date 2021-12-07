from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantumqubo",
    version="1.0",
    author="ORCA Computing",
    author_email="hello@orcacomputing.com",
    description="A package for running a QUBO solver on a simulation of ORCA's PT-Series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=[package for package in find_packages()
                if package.startswith('quantumqubo')],
    python_requires=">=3.7",
    install_requires=[
        'torch',
        'numpy<1.21,>=1.17',
        'numba',
        'matplotlib'
    ],
    extras_require={
        'portfolio-optimization': ['pandas'],
        'max-cut': ['networkx'],
        'tests':['pytest'],
        'notebooks':['jupyter']
    }
)