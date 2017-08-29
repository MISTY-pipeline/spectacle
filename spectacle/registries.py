import os

from astropy.table import Table, join
from astropy.io import registry as io_registry
from .spell import SpellCorrector


class LineRegistry(Table):
    def __init__(self, *args, **kwargs):
        super(LineRegistry, self).__init__(*args, **kwargs)

        self._corrector = SpellCorrector(list(self['name']))

    def with_name(self, name):
        name = self.correct(name)

        return next((row for row in self if row['name'] == name), None)

    def correct(self, name):
        correct_name = self._corrector.correction(name)

        print("Correct name is {} given {}.".format(correct_name, name))
        return correct_name


# Import any available line information databases
cur_path = os.path.realpath(__file__).split(os.sep)
cur_path = os.sep.join(cur_path[:-1])

line_registry = LineRegistry.read(os.path.join(cur_path, "data", "atoms.ecsv"),
                                  format="ascii.ecsv")