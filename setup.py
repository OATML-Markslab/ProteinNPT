from setuptools import setup,find_packages

with open("README.md") as f:
    readme = f.read()

setup(
    name="proteinnpt",
    description="ProteinNPT: Improving Protein Property Prediction and Design with Non-Parametric Transformers",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={
        'proteinnpt': ['proteinnpt/utils/tranception/utils/tokenizers/Basic_tokenizer']
    },
    author="Pascal Notin and Ruben Weitzman",
    version="1.5.2",
    license="MIT",
    url="https://github.com/OATML-Markslab/ProteinNPT",    
    packages=find_packages()
)