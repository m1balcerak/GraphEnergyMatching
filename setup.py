from setuptools import find_packages, setup

setup(
    name="graph-energy-matching",
    version="0.7.0",
    url="https://github.com/m1balcerak/GraphEnergyMatching",
    description="Graph Energy Matching for molecular graph generation",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
