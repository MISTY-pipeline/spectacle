# Spectacle

This is the repository for the spectral generator and analysis toolkit for the MISTY project.

Features:
- Generate spectra from models
- Compose spectra from observational data
- LSF kernel, simulated noise, and resampling support
- Automatic line identification
- Custom line list support
- Simultaneous line fitting of multiple line features
- Built-in multi-component modeling of blended features
- Basic and higher-order analysis and correlation functionality
- Integrated analysis techniques for easy information gathering


## Installation

The package can be installed either by cloning the repository and installing via python

```bash
$ git clone https://github.com/MISTY-pipeline/spectacle.git
$ cd spectacle
$ python setup.py install
```

or, alternatively, you may install the package directly from GitHub without having to clone

```bash
$ pip install git+https://github.com/MISTY-pipeline/spectacle.git
```