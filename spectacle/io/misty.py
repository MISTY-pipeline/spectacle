import six
import os

from astropy.io import fits
from astropy.units import Unit

from ..core.spectra import Spectrum1D
from ..core.decorators import data_loader


def identify_misty_fits(origin, *args, **kwargs):
    with fits.open(args[0]) as hdulist:
        is_sim_file = dict(hdulist[0].header).get('HIERARCH SIMULATION_NAME')

    return (isinstance(args[0], six.string_types) and is_sim_file and
            os.path.splitext(args[0].lower())[1] == '.fits')


@data_loader("misty", identifier=identify_misty_fits)
def misty_reader(file_path):
    """
    Custom loader specifically designed to parse MISTY data files.

    .. note:: The MISTY FITS data file is a **collection** of spectra. Thus,
              the returned object is a *list* of `Spectrum1D` objects.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    spec_list : list
        List of `Spectrum1D` objects.

    """
    # Open the fits file
    hdulist = fits.open(file_path)

    # The primary header contains general information; the extensions contain
    # information on individual absorptions features
    meta = dict()

    meta['primary'] = dict(hdulist[0].header)
    meta['parameters'] = dict(hdulist[1].data)

    spec_list = []

    for i in range(2, len(hdulist)):
        ext_hdr = hdulist[i].header
        meta[hdulist[0].header['LINE_{}'.format(i)]] = dict(ext_hdr)

        spec = Spectrum1D(data=hdulist[i].data['flux'],
                          dispersion=hdulist[i].data['wavelength'],
                          tau=hdulist[i].data['tau'])

        spec_list.append(spec)

    hdulist.close()

    return spec_list

