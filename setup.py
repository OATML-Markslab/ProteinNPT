from setuptools import setup,find_packages

with open("README.md") as f:
    readme = f.read()

setup(
    name="proteinnpt",
    description="ProteinNPT: Improving Protein Property Prediction and Design with Non-Parametric Transformers",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Pascal Notin and Ruben Weitzman",
    version="1.4",
    license="MIT",
    url="https://github.com/OATML-Markslab/ProteinNPT",    
    packages=find_packages()
)