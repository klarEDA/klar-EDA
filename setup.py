from setuptools import setup, Extension, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="klar_eda",
    author="Sayali, Rishabh, Ishaan, Ashish",
    author_email="contact.klareda@gmail.com",
    version="0.0.1",
    description="A python library for automated data visualization and pre-processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/klarEDA/klar-EDA",
    packages=find_packages(),
    namespace_packages=find_packages(include=['klar_eda', 'klar_eda.visualize','klar_eda.preprocess']),
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'tensorflow',
        'sklearn',
        'opencv-python',
        'sphinx'
    ],
    python_requires='>=3.6',
)