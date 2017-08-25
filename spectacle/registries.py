import os

from astropy.table import Table, join
from astropy.io import registry as io_registry


class LineRegistry(Table):
    def __init__(self, *args, **kwargs):
        super(LineRegistry, self).__init__(*args, **kwargs)

    def with_name(self, name):
        name = name.lower().replace(" ", "")

        return next((row for row in self if name == row['name']), None)


# Import any available line information databases
cur_path = os.path.realpath(__file__).split(os.sep)
cur_path = os.sep.join(cur_path[:-1])

line_registry = LineRegistry.read(os.path.join(cur_path, "data", "atoms.ecsv"),
                                  format="ascii.ecsv")