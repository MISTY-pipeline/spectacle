import six
import os

from astropy.io import registry as io_registry
from astropy.io import fits
from astropy.units import Unit

from ..core.spectra import Spectrum1D


def trident_reader(filename):
    # Open the fits file
    hdulist = fits.open(filename)

    # The primary header contains general information; the extensions contain
    # information on individual absorptions features
    meta = dict()

    meta['primary'] = dict(hdulist[0].header)

    for i in range(1, len(hdulist)):
        ext_hdr = hdulist[i].header
        meta[hdulist[0].header['LINE_{}'.format(i)]] = dict(ext_hdr)

    # Attempt to parse the unit for the data value
    unit = Unit("")

    hdulist.close()

    return Spectrum1D()


def identify_trident_fits(origin, *args, **kwargs):
    with fits.open(args[0]) as hdulist:
        is_sim_file = dict(hdulist[0].header).get('HIERARCH SIMULATION_NAME')

    return (isinstance(args[0], six.string_types) and is_sim_file and
            os.path.splitext(args[0].lower())[1] == '.fits')


io_registry.register_reader('trident', Spectrum1D, trident_reader)
io_registry.register_identifier('trident', Spectrum1D, identify_trident_fits)
