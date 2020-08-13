# -*- coding: utf-8 -*-
# Initial setup instructions from: https://queirozf.com/entries/package-a-python-project-and-make-it-available-via-pip-install-simple-example

import setuptools

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name='YourModuleNameGoesHere',  # TODO: HIGH: change module name
    version='0.0.1',  # TODO: HIGH: change initial version as necessary
    url='https://github.com/username/repo',  # TODO: HIGH
    author='',  # TODO: HIGH
    author_email='',  # TODO: HIGH
    description='description goes here',  # TODO: HIGH
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='',  # TODO: HIGH
    packages=['YourModuleNameHere. Restructuring will probably need to occur first.', ],  # TODO: HIGH
    install_requires=[  # TODO: HIGH: re-evaluate necessary minimum versions of packages in requirements.txt
        'numpy>=1.1',
        'matplotlib>=1.5',
        # etc...
    ],
    classifiers=(                                  # Classifiers help people find your  # TODO: HIGH
        "Programming Language :: Python :: 3",     # projects. See all possible classifiers
        "License :: OSI Approved :: MIT License",  # in https://pypi.org/classifiers/
        "Operating System :: OS Independent",
    ),
)
