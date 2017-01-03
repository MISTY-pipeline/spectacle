from spectacle.core.spectra import Spectrum1D
import numpy as np


def main():
    spec = Spectrum1D()
    spec.add_line(lambda_0=1215, f_value=0.4, v_doppler=1e6, column_density=10**13)
    spec.add_line(lambda_0=999, f_value=0.3, v_doppler=1e7, column_density=10**14)
    print(spec.model)
    print(spec.model.tied)


if __name__ == '__main__':
    main()
