import logging
import os

from astropy.table import Table
import astropy.units as u

from ..utils.spell import SpellCorrector
from ..utils import find_nearest


class LineRegistry(Table):
    def __init__(self, *args, **kwargs):
        super(LineRegistry, self).__init__(*args, **kwargs)

    def with_name(self, name):
        ion = next((row for row in self if row['name'] == name), None)

        if ion is None:
            name = self.correct(name)
            ion = next((row for row in self if row['name'] == name), None)

            if ion is None:
                raise LookupError("No such line with name '{}' in ion database.".format(name))

        return ion

    def with_lambda(self, lambda_0):
        lambda_0 = u.Quantity(lambda_0, u.Unit('Angstrom'))
        ind = find_nearest(self['wave'], lambda_0.value)

        return self[ind]

    def correct(self, name):
        _corrector = SpellCorrector(list(self['name']))
        correct_name = _corrector.correction(name)

        if correct_name != name:
            logging.info(
                "Found line with name '{}' from given name '{}'.".format(
                    correct_name, name))

        return correct_name


# Import any available line information databases
cur_path = os.path.realpath(__file__).split(os.sep)
cur_path = os.sep.join(cur_path[:-1])

line_registry = LineRegistry.read(
    os.path.abspath(
        os.path.join(cur_path, "..", "data", "atoms.ecsv")),
    format="ascii.ecsv")