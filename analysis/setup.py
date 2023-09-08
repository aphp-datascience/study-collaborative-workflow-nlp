from distutils.core import setup

setup(
    name="analysis",
    version="1.0",
    description="Analysis module for eCCI",
    author="Thomas PETIT-JEAN",
    author_email="thomas.petitjean@aphp.fr",
    packages=["analysis"],
    install_requires=[
        "matplotlib",
        "seaborn",
    ],
)
