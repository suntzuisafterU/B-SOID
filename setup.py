# -*- coding: utf-8 -*-
# Initial setup instructions from: https://queirozf.com/entries/package-a-python-project-and-make-it-available-via-pip-install-simple-example

import setuptools

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name='bsoid', # This is likely the name that will be called on `pip install _`, so make it count # TODO: HIGH: change module name
    version='0.0.1',  # TODO: HIGH: change initial version as necessary
    url='https://github.com/username/repo',  # TODO: HIGH
    author='Example Author',  # TODO: HIGH
    author_email='example@example.com',  # TODO: HIGH
    description='description goes here',  # TODO: HIGH
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='',  # TODO: HIGH
    packages=setuptools.find_packages(include=['bsoid_py']),
    install_requires=[  # TODO: HIGH: re-evaluate necessary minimum versions of packages in requirements.txt
        'Cython',
        'numpy>=1.1',
        'matplotlib>=1.5',
        'bhtsne',
        'ffmpeg',
        'hdbscan',
        'joblib',
        'matplotlib',
        'networkx',
        'numpy',
        'pandas',
        'psutil',
        'opencv-python',
        'seaborn',
        'scikit-learn',
        'streamlit',
        'tables',
        'tqdm',
        'umap-learn',
        # etc...
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
    classifiers=[ # https://pypi.org/classifiers/  #TODO: HIGH
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
        ],
)
