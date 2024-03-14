from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as reqs_file:
    requirements = reqs_file.read().split("\n")

setup(
    name="lullaby",
    description="",
    long_description=readme,
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
)
