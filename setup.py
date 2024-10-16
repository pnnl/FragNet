from distutils.core import setup

setup(
    # Application name:
    name="fragnet",

    # Version number (initial):
    version="0.1.0",

    # Application author details:
    author="Gihan Panapitiya",
    author_email="gihan.panapitiya@pnnl.gov",

    # Packages
    #packages=["mpet","mpet.utils", "mpet.models", "mpet.doa"],

    # Include additional files into the package
    include_package_data=True,

    # Details
    url="",

    #
    # license="LICENSE.txt",
    description="FragNet: A GNN with Four Layers of Interpretability",

    # long_description=open("README.txt").read(),

    # Dependent packages (distributions)
    # install_requires=["openbabel"],
)

