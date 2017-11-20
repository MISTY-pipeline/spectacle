import setuptools


setuptools.setup(
    name="spectacle",
    version="0.1.0",
    url="https://github.com/misty-pipeline/spectacle",

    author="Nicholas Earl",
    author_email="nearl@stsci.edu",

    description="Spectral generator and analysis toolkit",
    long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
