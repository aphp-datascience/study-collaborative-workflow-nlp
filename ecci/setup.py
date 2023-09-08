from setuptools import find_packages, setup


def get_lines(relative_path):
    with open(relative_path) as f:
        return f.readlines()


INSTALL_REQUIRES = get_lines("requirements.txt")

setup(
    name="ecci",
    version="2.0.2",
    author="Data Science - DSI APHP",
    author_email="thomas.petitjean-ext@aphp.fr",
    description="clinic ai loop",
    python_requires=">=3.6",
    packages=find_packages(),
    package_data={"": ["icd10/*"]},
    install_requires=INSTALL_REQUIRES,
)
