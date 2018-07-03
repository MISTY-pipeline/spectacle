import logging

import astropy.units as u
import numpy as np
from astropy.constants import c, m_e
from astropy.modeling import Fittable1DModel, Fittable2DModel
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Const1D, Gaussian1D, RedshiftScaleFactor

from ..core.region_finder import find_regions
from ..core.spectrum import Spectrum1DModel
from ..io.registries import line_registry
from ..modeling.converters import FluxConvert, FluxDecrementConvert
from ..modeling.profiles import OpticalDepth1DModel
from ..utils import find_nearest
from ..utils.peak_detect import detect_peaks, region_bounds

PROTON_CHARGE = u.Quantity(4.8032056e-10, 'esu')
TAU_FACTOR = ((np.sqrt(np.pi) * PROTON_CHARGE ** 2 /
               (m_e.cgs * c.cgs))).cgs

dop_rel_equiv = u.equivalencies.doppler_relativistic

class LineFinder:
    @u.quantity_input(rest_wavelength=u.Unit('Angstrom'))
    def __init__(self, ion_name=None, rest_wavelength=None, redshift=0,
                 data_type='optical_depth', continuum=None, threshold=0.1,
                 min_distance=2, width=15, max_iter=4000, defaults=None):
        # Discern the rest wavelength for the spectrum. If an ion name is given
        # instead, use that to determine the rest wavelength
        self._rest_wavelength = u.Quantity(rest_wavelength or 0, 'Angstrom')

        if ion_name is not None:
            ion_name = line_registry.with_name(ion_name)
            self._rest_wavelength = ion_name['wave'] * line_registry['wave'].unit

        # Store the redshift value inside a redshift scalar model
        self._redshift_model = RedshiftScaleFactor(z=redshift)

        # Determine the data type. This effects the operations that get
        # performed in the rest of the line finding routine.
        if data_type not in ('optical_depth', 'flux', 'flux_decrement'):
            logging.error("No available data type named '%s'. Defaulting to"
                          "'optical_depth'.", data_type)
            self._data_type = 'optical_depth'
        else:
            self._data_type = data_type

        # If a continuum is provided, use that, otherwise create a basic one
        # dependent on the type of data.
        if continuum is not None and isinstance(continuum, Fittable1DModel):
            self._continuum_model = continuum
        elif self._data_type == 'flux':
            self._continuum_model = Const1D(1, fixed={'amplitude': True})
        else:
            self._continuum_model = None

        self._threshold = threshold
        self._min_distance = min_distance
        self._width = width
        self._max_iter = max_iter
        self._defaults = defaults or {}

    @u.quantity_input(x=['length', 'speed'])
    def __call__(self, x, y):
        # Convert the dispersion to velocity space
        with u.set_enabled_equivalencies(dop_rel_equiv(self.rest_wavelength)):
            vel = self._redshift_model.inverse(x).to('km/s')
            wav = self._redshift_model.inverse(x).to('Angstrom')

        import matplotlib.pyplot as plt
        f, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(x, y)
        ax2.plot(wav, y)
        ax3.plot(vel, y)

        ax2.set_xlabel("Wavelength [Angstrom]")
        ax3.set_xlabel("Velocity [km/s]")

        f.tight_layout()

        # Convert the min_distance from dispersion units to data elements
        min_ind = (np.abs(vel - (vel[0] + self._min_distance))).argmin()
        logging.info("Min distance: %i elements.", min_ind)

        spec_mod = Spectrum1DModel(rest_wavelength=self.rest_wavelength,
                                   redshift=self._redshift_model.z,
                                   continuum=self._continuum_model)

        # Calculate the bounds on each absorption feature
        if (np.max(y) - np.min(y)) > self._threshold:
            spec_mod.bounds = region_bounds(y,
                                            height=self._threshold,
                                            distance=min_ind,
                                            smooth=True)
            logging.info("Found %i minima.", len(spec_mod.bounds))

        # For each set of bounds, estimate the initial values for that line
        for mn_bnd, mx_bnd in spec_mod.bounds:
            # Calculate the centroid of this region
            centroid = vel[mn_bnd] + (vel[mx_bnd] - vel[mn_bnd]) * 0.5
            centroid = vel[find_nearest(vel, centroid)]

            logging.info("Found centroid at %s (%s)", centroid,
                centroid.to('Angstrom', dop_rel_equiv(self.rest_wavelength)))

            # Estimate the doppler b and column densities for this line
            v_dop, col_dens = parameter_estimator(
                (mn_bnd, mx_bnd), vel, y, self.rest_wavelength,
                self._continuum_model)

            # Create an optical depth profile for the line
            vel_mn_bnd = vel[mn_bnd].value #self._redshift_model.inverse(vel[mn_bnd].value)
            vel_mx_bnd = vel[mx_bnd].value #self._redshift_model.inverse(vel[mx_bnd].value)

            line_params = dict(
                lambda_0=self.rest_wavelength,
                v_doppler=v_dop,
                column_density=col_dens,
                delta_v=centroid,
                bounds={
                    'delta_v': (vel_mn_bnd + centroid.value * 0.5, vel_mx_bnd - centroid.value * 0.5),
                    # 'v_doppler': (v_dop.value * 0.1, v_dop.value * 10),
                    # 'column_density': (col_dens - 1, col_dens + 1)
                })

            line_params.update(self._defaults)

            # Add line to the spectrum model
            spec_mod.add_line(model=OpticalDepth1DModel(**line_params))

        # Begin fitting the spectrum. Get the data-type-appropriate model.
        data_mod = getattr(spec_mod, self._data_type)
        fitter = LevMarLSQFitter()

        # The fitter will strip away any unit information, so we must be sure
        # to provide the dispersion as the redshifted velocity values.
        data_mod = fitter(data_mod, self._redshift_model(vel), y,
                          maxiter=self._max_iter)

        # Update spectrum line model parameters with fitted results
        fit_line_mods = [smod for smod in data_mod
                         if hasattr(smod, 'lambda_0')]

        if len(fit_line_mods) > 0:
            spec_mod._line_model = np.sum(fit_line_mods)

        # Calculate the regions in the raw data
        spec_mod.regions = {(reg[0], reg[1]): []
                            for reg in find_regions(y, rel_tol=1e-2, abs_tol=1e-4,
                                continuum=spec_mod._continuum_model(vel))}
        logging.info("Found %i absorption regions.", len(spec_mod.regions))

        return spec_mod

    @property
    def rest_wavelength(self):
        return self._rest_wavelength


def parameter_estimator(bounds, x, y, rest_wavelength, continuum):
    bound_low, bound_up = bounds
    mid_diff = int((bound_up - bound_low) * 0.5)
    mask = ((x >= x[bound_low - mid_diff]) & (x <= x[bound_up + mid_diff]))

    if continuum is not None:
        y = continuum(x) - y

    mx, my = x[mask], y[mask]

    # Width can be estimated by the weighted 2nd moment of the x coordinate
    dx = mx - np.mean(mx)
    fwhm = 2 * np.sqrt(np.sum((dx * dx) * my) / np.sum(my))
    center = np.sum(mx * my) / np.sum(my)
    sigma = fwhm / 2.355

    # Amplitude is derived from area
    delta_x = mx[1:] - mx[:-1]
    sum_y = np.sum(my[1:] * delta_x)

    height = sum_y / (sigma * np.sqrt(2 * np.pi))

    g = Gaussian1D(amplitude=height,
                   mean=center,
                   stddev=sigma,
                   bounds={'mean': (mx[0].value, mx[-1].value),
                           'stddev': (None, 4 * sigma.value)})

    g = continuum + (g | FluxDecrementConvert()) if continuum is not None else g

    g_fit = LevMarLSQFitter()(g, mx, my)

    new_delta_x = x[1:] - x[:-1]
    new_y = g_fit(x)
    new_fwhm = np.abs(g_fit[1].fwhm)
    new_sum_y = np.sum(new_y[1:] * new_delta_x)

    # Estimate the doppler b parameter
    v_dop = (new_fwhm / 2.355).to('km/s')

    # Estimate the column density
    f_value = line_registry.with_lambda(rest_wavelength)['osc_str']
    col_dens = (new_sum_y.value * (v_dop / rest_wavelength).to('Hz') / (
        TAU_FACTOR * f_value)).to('1/cm2')
    col_dens = np.log10(col_dens.value)

    logging.info("""Estimated intial values:
    Center: {:g}
    Column density: {:g}
    Doppler width: {:g}""".format(center, col_dens, v_dop))

    return v_dop, col_dens
