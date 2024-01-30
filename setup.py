from setuptools import setup, find_packages

setup(
    name='pygl',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.3',
        'scikit-learn==1.4.0',
        'matplotlib==3.8.2',
        'scipy==1.12.0',
        'networkx==3.2.1'
    ],
    author='Masaki Takahashi',
    url='https://github.com/matyaki-matyaki/PyGraphLearning'
)