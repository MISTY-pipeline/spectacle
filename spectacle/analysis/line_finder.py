import peakutils
from spectacle.modeling import *
from spectacle.core.spectrum import Spectrum1D

from astropy.modeling import Fittable2DModel
from astropy.modeling.models import Voigt1D

from .initializers import Voigt1DInitializer


class LineFinder(Fittable2DModel):
    inputs = ('x', 'y')
    outputs = ('y',)

    input_units_strict = True
    input_units_allow_dimensionless = True

    center = Parameter(default=1216, min=0, unit=u.Unit('Angstrom'), fixed=True)
    z = Parameter(default=0, min=0, fixed=True)
    threshold = Parameter(default=0.5)
    min_distance = Parameter(default=30, min=0)

    input_units = {'x': u.Unit('km/s')}

    @property
    def input_units_equivalencies(self):
        return {'x': [
            (u.Unit('km/s'), u.Unit('Angstrom'),
             lambda x: WavelengthConvert(self.center)(x * u.Unit('km/s')),
             lambda x: VelocityConvert(self.center)(x * u.Unit('Angstrom')))
        ]}

    def evaluate(self, x, y, center, z, threshold, min_distance):
        # Astropy fitters strip modeling of their unit information. However, the
        # first iterate of a fitter includes the quantity arrays passed to the
        # call method. If the input array is a quantity, immediately store the
        # quantity unit as a reference for future iterations.
        if isinstance(x, u.Quantity):
            self.input_units = {'x': x.unit}

        print(center[0])

        x = u.Quantity(x, self.input_units['x'])
        wavelength = x.to('Angstrom',
                          equivalencies=self.input_units_equivalencies['x'])
        # min_distance = u.Quantity(min_distance, self.input_units['x'])

        ion = line_registry.with_name('HI1216')
        ion = ion['wave'] * line_registry['wave'].unit

        indexes = peakutils.indexes(y, thres=threshold, min_dist=min_distance)

        print("Found {} lines.".format(len(indexes)))

        spectrum = Spectrum1D(center=center[0], redshift=z)

        # Get the interesting regions within the data provided
        reg = find_regions(y, rel_tol=1e-2, abs_tol=1e-4)

        filt_reg = [(rl, rr) for rl, rr in reg
                    if wavelength[rl] <= self.center <= wavelength[rr]]

        # Create the mask that can be applied to the original data to produce
        # data only with the particular region we're interested in
        mask = np.logical_or.reduce(
            [(wavelength > wavelength[rl]) &
             (wavelength <= wavelength[rr]) for rl, rr in filt_reg])

        for ind in indexes:
            peak = u.Quantity(wavelength[ind], wavelength.unit)
            dlambda = peak - ion
            dv = dlambda.to('km/s', equivalencies=self.input_units_equivalencies['x'])

            print("Found line at {} ({} with shift of {}).".format(peak, ion, dlambda))

            # Estimate the voigt parameters
            voigt = Voigt1D()
            initializer = Voigt1DInitializer()
            init_voigt = initializer.initialize(voigt,
                                                wavelength[mask].value,
                                                y[mask])

            # Fit the spectral line to the voigt line we initialized above
            fitter = LevMarLSQFitter()

            fit_line = fitter(init_voigt, wavelength[mask].value, y[mask],
                              maxiter=200)

            print(list(zip(fit_line.param_names, fit_line.parameters)))
            print(10 ** (init_voigt.amplitude_L.value * 2), (fit_line.amplitude_L.value * 2))

            # Add the line to the spectrum object
            line = TauProfile(lambda_0=center[0],
                              delta_lambda=dlambda,
                              v_doppler=1e7 * u.Unit('cm/s'),
                              column_density=10 ** (fit_line.amplitude_L.value *
                                                    1.4142135623730951) * u.Unit('1/cm2')
                              )

            spectrum.add_line(model=line)

        self._result_model = spectrum

        import matplotlib.pyplot as plt

        f, ax = plt.subplots()

        ax.plot(x, y)
        ax.plot(x, spectrum.optical_depth(x))

        plt.show()

        return spectrum.optical_depth(x)

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict()

    @property
    def result_model(self):
        return self._result_model
