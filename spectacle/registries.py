import os

from astropy.table import Table, join

class LineRegistry(Table):
    def __init__(self, *args, **kwargs):
        super(LineRegistry, self).__init__(*args, **kwargs)

        # Import any available line information databases
        cur_path = os.path.realpath(__file__).split(os.sep)
        cur_path = os.sep.join(cur_path[:-1])

        for root, dirs, files in os.walk(os.path.join(cur_path, "data")):
            for name in files:
                self.read(os.path.join(root, name), format="ascii.ecsv")


line_registry = LineRegistry()