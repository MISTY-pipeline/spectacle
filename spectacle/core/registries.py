import os
import logging

import numpy as np
from astropy.table import Table


class LineRegistry:
    def __init__(self, path=None):
        if path is None:
            path = os.path.abspath(
                os.path.join(__file__, '..', '..', 'data', 'line_list',
                             'atoms.ecsv'))

        self._line_list = self.parse(path, format='ascii.ecsv')

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self._line_list[i]
                    for i in range(key.start, key.stop, key.step)]

        return self._line_list[key]

    def __repr__(self):
        return self._line_list.__repr__()

    @staticmethod
    def parse(path, format):
        line_list = Table.read(path, format=format)

        return line_list

    def read(self, path, format='ascii.ecsv'):
        self._line_list = self.parse(path, format=format)

    def add_ion(self, name, wave, osc_str, gamma):
        self._line_list.add_row([name, wave, osc_str, gamma])

    def remove_ion(self, name=None, wave=None, osc_str=None, gamma=None):
        indices = []

        if name is not None:
            indices += list(np.where(self._line_list['name'] == name)[0])

        if wave is not None:
            indices += list(np.where(self._line_list['wave'] == wave)[0])

        if osc_str is not None:
            indices += list(np.where(self._line_list['osc_str'] == osc_str)[0])

        if gamma is not None:
            indices += list(np.where(self._line_list['gamma'] == gamma)[0])

        # Get all the unique indices
        indices = list(set(indices))

        if len(indices) > 0:
            logging.warn("More than one ion with name {} found, proceeding to "
                         "remove only {}".format(
                name, self._line_list['wave'][indices[0]]))

        self._line_list.remove_row(indices[0])


line_registry = LineRegistry()
