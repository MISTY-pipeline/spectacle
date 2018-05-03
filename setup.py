#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'astropy',
    'peakutils',
    'emcee'
]

setup_requirements = [
    'pytest-runner',
    # TODO(nmearl): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: put package test requirements here
]

setup(
    name='spectacle',
    version='0.2.0',
    description="Spectroscopic analysis package for simulated or observed spectra",
    long_description=readme + '\n\n' + history,
    author="Nicholas Earl",
    author_email='nearl@stsci.edu',
    url='https://github.com/MISTY-pipeline/spectacle',
    packages=find_packages(),
    use_2to3=True,
    entry_points={
        'console_scripts': []
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='spectacle',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
