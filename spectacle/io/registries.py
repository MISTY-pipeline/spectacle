import logging
import os

from astropy.table import Table

from ..utils.spell import SpellCorrector


class LineRegistry(Table):
    def __init__(self, *args, **kwargs):
        super(LineRegistry, self).__init__(*args, **kwargs)

    def with_name(self, name):
        name = self.correct(name)

        return next((row for row in self if row['name'] == name), None)

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