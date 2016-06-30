from setuptools import setup

setup(
    name='data_analyses_tools',
    version='0.1.0',
    description='Functions for data analyses and processing.',
    long_description=open('README.md').read(),
    author='Roald Ruiter',
    author_email='roaldruiter@gmail.com',
    url='https://github.com/Roald87/data_analyses_tools',
    setup_requires=[
        'os',
        'numpy',
        'pandas',
    ],
)
