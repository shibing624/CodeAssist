# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

__version__ = "0.1.0"

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='codeassist',
    version=__version__,
    description='Code AutoComplete',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/autocoder',
    license='Apache License 2.0',
    zip_safe=False,
    python_requires='>=3.5',
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='CodeGenie,autocomplete,code-autocomplete',
    install_requires=[
        "loguru",
        "transformers",
        "pandas",
    ],
    packages=find_packages(exclude=['tests']),
    package_dir={'autocoder': 'autocoder'},
    package_data={'autocoder': ['*.*']}
)
