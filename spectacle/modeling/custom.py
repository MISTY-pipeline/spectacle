from astropy.modeling.models import RedshiftScaleFactor as _RedshiftScaleFactor, Scale as _Scale
import astropy.units as u


class RedshiftScaleFactor(_RedshiftScaleFactor):
    def _parameter_units_for_data_units(self, input_units, output_units):
        return dict()


class Scale(_Scale):
    def _parameter_units_for_data_units(self, input_units, output_units):
        return dict()