from setuptools import setup, find_packages

setup(
    name="pymsis",
    version="2.0",
    package_dir={"": "pymsis"},  # tells setuptools that packages are found in the "pymsis" directory
    packages=find_packages(where="pymsis"),
    description="Local installation of NRLMSIS2.0 for pymsis",
)
