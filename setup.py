from setuptools import setup, Extension, find_packages

setup(
    name="klar_eda",
    version="0.0.1",
    description="A python library for automated data visualization and pre-processing",
    url="https://github.com/psy2d/klar-eda",
    packages=['klar_eda'],
    namespace_packages=find_packages(include=['klar_eda', 'klar_eda.*']),
    license='MIT',
    install_requires==[
        'numpy',
        'pandas',
        'seaborn',
        'tensorflow',
        'sklearn',
        'opencv-python'
    ],
)
