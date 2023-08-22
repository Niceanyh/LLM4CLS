from setuptools import setup, find_packages

setup(
    name='llm4cls',
    version='0.1',
    description='Text Classification with Large Language Models',
    author='niceanyh',
    author_email='niceanyh@gmail.com',
    url='https://github.com/Niceanyh/llm4cls',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18',
        'requests>=2.22',
    ],
)