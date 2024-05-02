from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(requirements_file:str)->List[str]:
    """Get requirements from a given requirements file.

    Args:
        requirements_file (str): The path to the requirements

    Returns: 
        List[str]: A list of requirements
    """

    requirements = []
    with open(requirements_file, 'r') as f:
        for line in f:
            requirements.append(line.strip())

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(name="e2e-mlops", 
      version="0.0.1", 
      author="beingmechon", 
      author_email="bharanishraj@gmail.com",
      packages=find_packages(),
      install_requires=get_requirements('./requirements.txt')
      )
