
from setuptools import setup, find_packages

setup(
    name="mllake",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "scikit-learn", "xgboost", "pandas"],
)

