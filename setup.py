# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

__version__ = "0.0.3"

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    reqs = f.read()

setup(
    name='code-autocomplete',
    version=__version__,
    description='Code AutoComplete',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/code-autocomplete',
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
    keywords='autocomplete,code-autocomplete',
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(exclude=['tests']),
    package_dir={'autocomplete': 'autocomplete'},
    package_data={'autocomplete': ['*.*', '../LICENSE', '../README.*', '../*.txt', 'utils/*',
                                   'data/*', ]}
)
