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
    classifiers=(  # https://pypi.org/classifiers/  #TODO: HIGH
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        # 'Intended Audience :: Education',
        # 'Intended Audience :: Science/Research',
        # 'License :: OSI Approved :: GNU General Public License (GPL)',
        # 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        # 'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        # 'Topic :: Scientific/Engineering :: Image Recognition',
        # 'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ),
)
