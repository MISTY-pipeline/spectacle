from astropy.modeling import Fittable1DModel


class SingleIonSpectral1D(Fittable1DModel):
    @property
    def input_units(self):
        pass

    def __init__(self, ion, *args, **kwargs):
        if isinstance(ion, str):


        super().__init__(*args, **kwargs)