from setuptools import setup, find_packages

setup(
    name="amoe",
    version="0.1.0",
    description="AMOE: Agglomeration Mixture of Experts Vision Foundation Model",
    author="Sofian Chaybouti",
    packages=find_packages(include=["amoe", "amoe.*"]),
    python_requires=">=3.10",
    homepage="https://github.com/sofianchaybouti/amoe",
)