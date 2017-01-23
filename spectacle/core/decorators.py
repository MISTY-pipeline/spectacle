from astropy.io import registry as io_registry
from functools import wraps
from .spectra import Spectrum1D

import logging


def data_loader(label, identifier=None):
    def decorator(func):
        logging.info("Added {} to custom loaders.".format(label))
        io_registry.register_reader(label, Spectrum1D, func)
        io_registry.register_identifier(label, Spectrum1D, identifier)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def custom_writer(label):
    def decorator(func):
        logging.info("Added {} to custom writers.".format(label))
        io_registry.register_writer(label, Spectrum1D, func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
