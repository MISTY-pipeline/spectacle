
Reading and Writing Spectra
===========================

The spectrum object leverages the Astropy IO infrastructure to allow custom
loading of spectral data. By default, the loader understands ASCII and FITS
files that follow simple rules:

    * ASCII files have to have three columns. If the file is not an `.ecsv`,
      then the loader assumes that the first column is data (e.g. flux), the
      second uncertainty, and the third dispersion.
    * FITS files are expected to have a single table extension with three
      columns in the same order as above.

When spectral data is read in, the format can be specified explicitly.
Otherwise, the system will attempt to automatically discern the type of data::

    >>> spectrum = Spectrum1D.read("my_data.fits")
    >>> spectrum = Spectrum1D.read("my_data.fits", format="fits")

Likewise, there are three default writers to export spectral data. FITS, basic
ASCII, and ECSV. As with reading a custom spectrum data file, writing one
should also specify the format::

    >>> spectrum.write("my_new_data.fits", format="fits")

Custom Readers and Writers
--------------------------

Custom readers and writers are used to help Spectacle parse unique data sets.
This is done by defining a function that returns a `Spectrum1D` object, and
adding that function to the IO registry via a simple decorator.

All custom functions should be placed in a python file in the directory
`~/.spectacle`.

::

    from spectacle.core.decorators import data_loader
    import numpy as np


    @data_loader(label="my-data")
    def my_data_loader(file_path, *args, **kwargs):
        # A simple three-column file
        flux, disp, uncert = np.loadtxt(file_path, unpack=True)

        spec = Spectrum1D(flux, dispersion=disp, uncertainty=uncert,
                          disp_unit="Angstrom")

        return spec

This custom loader would be called from called like so::

    >>> spectrum = Spectrum1D.read("path/to/my_data.txt", format="my-data")

Likewise, a custom writer can be added your `.py` file in the `~/.spectacle`
directory in a similar fashion::

    from spectacle.core.decorators import data_wrtier


    @data_writer(label="my-data")
    def my_data_writer(spectrum, file_path, clobber=False):
        flux = spectrum.data
        wave = spectrum.dispersion
        uncert = spectrum.uncertainty

        np.savetxt(file_path, np.transpose([flux, uncert, wave])

And to call the custom writer::

    >>> spectrum.write("path/to/save/my_new_data.txt", format="my-data")