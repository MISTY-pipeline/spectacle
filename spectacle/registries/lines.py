import logging
import os

from astropy.table import QTable
import astropy.units as u

from ..utils.spelling_corrector import SpellingCorrector
from ..utils.misc import find_nearest
import numpy as np

__all__ = ['LineRegistry']


DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "atoms.ecsv"))


class LineRegistry(QTable):
    def __init__(self, *args, **kwargs):
        super(LineRegistry, self).__init__(*args, **kwargs)

    def subset(self, ions):
        inds = []

        for ion in ions:
            if isinstance(ion, str):
                name = self.correct(ion)
            elif isinstance(ion, u.Quantity) or isinstance(ion, float):
                lambda_0 = u.Quantity(lambda_0, u.Unit('Angstrom'))
                ind = find_nearest(self['wave'], lambda_0)
                name = self[ind]['name']
            else:
                logging.error("No ion could be found for {}.".format(ion))
                continue

            # TODO: use intersect1d when we update requirements to numpy 1.15
            index = np.where(self['name'] == name)[0]

            if len(index) > 0:
                inds.append(index[0])

        return self[np.array(inds)]

        # return line_registry[np.intersect1d(
        #     line_registry['name'], names, return_indices=True)[1]]

    def with_name(self, name):
        ion = next((row for row in self if row['name'] == name), None)

        if ion is None:
            name = self.correct(name)
            ion = next((row for row in self if row['name'] == name), None)

            if ion is None:
                raise LookupError("No such line with name '{}' in ion "
                                  "database.".format(name))

        return ion

    def with_lambda(self, lambda_0):
        lambda_0 = u.Quantity(lambda_0, u.Unit('Angstrom'))
        ind = find_nearest(self['wave'], lambda_0)

        return self[ind]

    def correct(self, name):
        _corrector = SpellingCorrector(list(self['name']))
        correct_name = _corrector.correction(name)

        if correct_name != name:
            logging.info(
                "Found line with name '{}' from given name '{}'.".format(
                    correct_name, name))

        return correct_name

    def load(self, path):
        self.remove_rows(slice(0, len(self)))

        new_table = LineRegistry.read(path, format='ascii.ecsv')

        for row in new_table:
            self.add_row(row)

    def load_default(self):
        self.load(DB_PATH)


line_registry = LineRegistry.read(DB_PATH, format='ascii.ecsv')
