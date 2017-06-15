import matplotlib.pyplot as plt
from astropy.io import fits

from spectacle.core.spectra import Spectrum1D
from spectacle.modeling.models import Absorption1D, Line
from spectacle.core.registries import line_registry
from spectacle.modeling.fitting import DynamicLevMarFitter, LevMarLSQFitter
from spectacle.modeling.optimizers import PosteriorFitter
from spectacle.process.lsf import LSF, COSLSF

from uncertainties import unumpy as unp

hdulist = fits.open("/Users/nearl/Dropbox/misty-foggie-example/"
                    "hlsp_misty_foggie_halo008508_rd0042_i013.0-a6.01_v1_los.fits")
print(hdulist.info())

ind = 2
name = hdulist[ind].header['LINENAME']
lambda_0 = hdulist[ind].header['RESTWAVE']
gamma = hdulist[ind].header['GAMMA']
f_value = hdulist[ind].header['F_VALUE']
wavelength = hdulist[ind].data['wavelength']
flux = hdulist[ind].data['flux']

# print(hdulist[2].header)

spectrum = Spectrum1D(flux, dispersion=wavelength)
line = Line(name=name, lambda_0=lambda_0,
            v_doppler=1e7,
            f_value=f_value, gamma=gamma, fixed={'lambda_0': False,
                                                 'f_value': True,
                                                 'gamma': True,
                                                 'v_doppler': False,
                                                 'column_density': False,
                                                 'delta_v': True,
                                                 'delta_lambda': True
                                                 })

spec_mod = Absorption1D(lines=[line])#, line2, line3])
y = spec_mod(spectrum.dispersion)

# Plot
f, (ax1) = plt.subplots()
ax1.plot(y.dispersion, y.data)
plt.savefig("output.png")

# Create a fitter. The default fitting routine is a LevMarLSQ.
fitter = PosteriorFitter()

fit_spec_mod = fitter(spec_mod, spectrum.dispersion, spectrum.data)

# Print some info
# print(fitter.fit_info['message'])

for mod in fit_spec_mod:
    print(list(zip(mod.param_names, mod.parameters)))

fit_y = fit_spec_mod(spectrum.dispersion)


# Plot
f, (ax1) = plt.subplots()
ax1.plot(spectrum.dispersion, spectrum.data)
ax1.plot(y.dispersion, y.data)
ax1.plot(fit_y.dispersion, fit_y.data)
plt.savefig("output.png")