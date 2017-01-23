from astropy.io import registry as io_registry
import os
import importlib.machinery as mach
import importlib.util as util
import logging

# Load the default IO functions
from .io import *

logging.basicConfig(level=logging.INFO)


def load_user():
    # Get the path relative to the user's home directory
    path = os.path.expanduser("~/.spectacle")

    # If the directory doesn't exist, create it
    if not os.path.exists(path):
        os.mkdir(path)

    # Import all python files from the directory
    for file in os.listdir(path):
        if not file.endswith("py"):
            continue
        # loader = mach.SourceFileLoader('custom', os.path.join(path, file))
        # mod = util.spec_from_loader(loader.name, loader)
        # loader.exec_module(mod)
        spec = util.spec_from_file_location('custom', os.path.join(path, file))
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        logging.info("Loaded module {}.".format(mod))


load_user()
