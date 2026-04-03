"""
jnu-timefs-for-finance setup configuration.
"""
from setuptools import setup, find_packages

setup(
    name="jnu-timefs-for-finance",
    version="0.1.0",
    description="A financial data analysis project based on TimeFS",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
)
