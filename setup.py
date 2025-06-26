from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="mbnpy",
    version="0.1.5",  # Update version when making changes
    packages=find_packages(),
    author="The DUCO team",
    author_email="ji-eun.byun@glasgow.ac.uk",
    description="MBNpy is a Python package for Bayesian network applications for large-scale system events (i.e. high-dimensional data).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jieunbyun/MBNpy",  # GitHub URL
    install_requires=read_requirements(),
    python_requires=">=3.12",  # Enforces Python 3.12 or later
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)