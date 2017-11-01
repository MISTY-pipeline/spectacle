from astropy.modeling import models, Fittable1DModel, Parameter
from astropy.modeling import fitting
import astropy.units as u
import astropy.constants as const

from spectacle.core.spectrum import Spectrum1D
from spectacle.modeling.custom import Masker
from spectacle.analysis.metrics import Epsilon, CrossCorrelate, \
    CorrMatrixCoeff, KendallsTau, KolmogorovSmirnov, AndersonDarling

import numpy as np

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{siunitx}"]

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import seaborn as sns
sns.set_style("whitegrid", {'grid.color': '.95', 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

plt.rcParams["figure.figsize"] = [6, 8]

wavelength = np.linspace(1000, 2000, 1001) * u.Angstrom
velocity = np.linspace(-300, 600, 500) * u.Unit('km/s')

# Define the parameter ranges
v_doppler_range = np.linspace(1e6, 1e7, 3) * u.Unit('cm/s')
column_density_range = np.linspace(1e13, 1e14, 3) * u.Unit('1/cm2')
delta_lambda_range = np.linspace(0, 2, 100) * u.Angstrom
delta_v_range = np.linspace(0, 300, 100) * u.Unit('km/s')


def generate(name, corr_func, use_mask=False):
    # Define the control spectrum. It does not change.
    line1 = dict(lambda_0=1.21567010E+03 * u.Angstrom,
                 v_doppler=v_doppler_range[0],
                 column_density=column_density_range[1])
    
    line1_2 = dict(lambda_0=1.21567010E+03 * u.Angstrom,
                   v_doppler=v_doppler_range[0],
                   column_density=column_density_range[0],
                   delta_v=25 * u.Unit('km/s'))

    spectrum1 = Spectrum1D(center=line1['lambda_0']).add_line(**line1).add_line(**line1_2)
    
    if use_mask:
        x1_f, y1_f = Masker(continuum=np.ones(velocity.shape))(velocity, spectrum1.flux(velocity))
        x1_t, y1_t = Masker()(velocity, spectrum1.optical_depth(velocity))
        x1_d, y1_d = Masker()(velocity, spectrum1.flux_decrement(velocity))
    else:
        x1_f, y1_f = velocity, spectrum1.flux(velocity)
        x1_t, y1_t = velocity, spectrum1.optical_depth(velocity)
        x1_d, y1_d = velocity, spectrum1.flux_decrement(velocity)

    pline1_t = [{'x': x1_t, 'y': y1_t}]
    pline1_f = [{'x': x1_f, 'y': y1_f}]
    pline1_d = [{'x': x1_d, 'y': y1_d}]
    pline2_t = []
    pline2_f = []
    pline2_d = []
    
    def perform_variations(v_dop, col_dens):
        corr, corr_tau, corr_dec = [], [], []

        for dv in delta_v_range:
            line2 = dict(lambda_0=1.21567010E+03 * u.Angstrom,
                         v_doppler=v_dop,
                         column_density=col_dens,
                         delta_v=dv)
            
            spectrum2 = Spectrum1D(center=line2['lambda_0']).add_line(**line2)
            
            if use_mask:
                x2_f, y2_f = Masker(continuum=np.ones(velocity.shape))(velocity, spectrum2.flux(velocity))
            else:
                x2_f, y2_f = velocity, spectrum2.flux(velocity)

            corr.append(corr_func(x1_f, x2_f, y1_f, y2_f))
            
            pline2_f.append({'x': x2_f, 'y': y2_f})
            
            if use_mask:
                x2_t, y2_t = Masker()(velocity, spectrum2.optical_depth(velocity))
            else:
                x2_t, y2_t = velocity, spectrum2.optical_depth(velocity)

            corr_tau.append(corr_func(x1_t, x2_t, y1_t, y2_t))
            
            pline2_t.append({'x': x2_t, 'y': y2_t})
            
            if use_mask:
                x2_d, y2_d = Masker()(velocity, spectrum2.flux_decrement(velocity))
            else:
                x2_d, y2_d = velocity, spectrum2.flux_decrement(velocity)

            corr_dec.append(corr_func(x1_d, x2_d, y1_d, y2_d))
            
            pline2_d.append({'x': x2_d, 'y': y2_d})
        
        return corr, corr_tau, corr_dec
    
    corr_mat_tau = np.zeros((v_doppler_range.size, column_density_range.size, delta_v_range.size))
    corr_mat_flux = np.zeros((v_doppler_range.size, column_density_range.size, delta_v_range.size))
    corr_mat_flux_dec = np.zeros((v_doppler_range.size, column_density_range.size, delta_v_range.size))
        
    for i, v_dop in enumerate(v_doppler_range):
        for j, col_dens in enumerate(column_density_range):
            corr_flux, corr_tau, corr_dec = perform_variations(v_dop, col_dens)
            
            corr_mat_tau[i][j] = corr_tau
            corr_mat_flux[i][j] = corr_flux
            corr_mat_flux_dec[i][j] = corr_dec
    
#     for i, v_dop in enumerate(v_doppler_range):
#     corr_flux, corr_tau, corr_dec = perform_variations(v_doppler_range[1], column_density_range[2])
    
    return {
        'name': name,
        'tau': {
            'line1': pline1_t,
            'line2': pline2_t,
            'corr': corr_mat_tau
        },
        'flux': {
            'line1': pline1_f,
            'line2': pline2_f,
            'corr': corr_mat_flux
        },
        'flux_dec': {
            'line1': pline1_d,
            'line2': pline2_d,
            'corr': corr_mat_flux_dec
        },
    }


def plot_corr(res):
    corr_flux = res['flux']['corr']
    corr_tau = res['tau']['corr']
    corr_flux_dec = res['flux_dec']['corr']
    
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    f.subplots_adjust(hspace=0)
    
    ax1.set_ylabel("Correlation in Tau")
    
    ax2.set_ylabel("Correlation in Flux")
    
    ax3.set_ylabel("Correlation in Flux Decrement")
    ax3.set_xlabel("$\Delta v$ [$\si{km.s^{-1}}$]")
    
    colors = sns.color_palette()
    line_styles = ['--', '-.', ':']

    # Variations in vdop
    for i in range(len(v_doppler_range)):
        for j in range(len(column_density_range)):
            ax1.plot(delta_v_range, corr_tau[i, j],
                     color=colors[i], 
                     linestyle=line_styles[j]
                    )
            ax2.plot(delta_v_range, corr_flux[i, j],
                     color=colors[i], 
                     linestyle=line_styles[j]
                    )
            ax3.plot(delta_v_range, corr_flux_dec[i, j],
                     color=colors[i], 
                     linestyle=line_styles[j]
                    )

    #Create legend from custom artist/label lists
    ax1.legend([plt.Line2D((0,1),(0,0), linestyle='', marker='o', color=colors[0]),
                plt.Line2D((0,1),(0,0), linestyle='', marker='o', color=colors[1]),
                plt.Line2D((0,1),(0,0), linestyle='', marker='o', color=colors[2]),
                plt.Line2D((0,1),(0,0), color='k', linestyle='--'),
                plt.Line2D((0,1),(0,0), color='k', linestyle='-.'),
                plt.Line2D((0,1),(0,0), color='k', linestyle=':'),],
              ["$\\num{{ {0.value:0.02e} }}$ {0.unit:latex}".format(x) for x in v_doppler_range] + 
              ["$\\num{{ {0.value:0.02e} }}$ {0.unit:latex}".format(x) for x in column_density_range],
              ncol=2)

    f.savefig(res['name'] + '.png')

            
def plot_shift(res):
    f, axes = plt.subplots(3, 1, sharex=True)
    f.set_size_inches(8, 6)
    f.subplots_adjust(hspace=0)
    
    for k, ax in zip(['tau', 'flux', 'flux_dec'], axes):
        if k == 'tau':
            y_label = "Tau"
            ax.legend(loc=0)
        elif k == 'flux':
            y_label = 'Normalized Flux'
        elif k == 'flux_dec':
            y_label = 'Normalized Flux Decrement'
            ax.set_xlabel("$\Delta v$ [km/s]")
            
        ax.set_ylabel(y_label)
        
        line1 = res[k]['line1']
        line2 = res[k]['line2']
    
        for i, d in enumerate(line2[::int(len(delta_v_range)/5)]):
            ax.plot(d['x'], d['y'], color=sns.color_palette()[1], alpha=0.35, label="Comparative Profile")

        for d in line1:
            ax.plot(d['x'], d['y'], label="Fiducial Profile")

    f.savefig("basic_shift.png")


import itertools

# f, ax1 = plt.subplots()
# f.set_size_inches(6, 6)
#
# ax1.set_xlabel("Column Density $[\si{cm^{-2}}]$")
# ax1.set_ylabel("Doppler $b$ $[\si{cm/s}]$")
#
# x, y = list(zip(*list(itertools.product(column_density_range, v_doppler_range))))
#
# ax1.scatter([column_density_range[1].value], [v_doppler_range[1].value], s=100, label="Fiducial Profile")
# ax1.scatter([i.value for i in x], [i.value for i in y], label="Comparative Profile")
#
# ax1.legend(loc=0)

# Multiprocessing can't take lambdas, so create some dynamic class objects
comp_corr = {
    'epsilon': lambda x1, x2, y1, y2: Epsilon()(y1, y2),
    'cross_correlation': lambda x1, x2, y1, y2: CrossCorrelate()(y1, y2),
    'corr_matrix': lambda x1, x2, y1, y2: CorrMatrixCoeff()(y1, y2),
    'kendalls_tau': lambda x1, x2, y1, y2: KendallsTau()(y1, y2),
    'ks': lambda x1, x2, y1, y2: KolmogorovSmirnov()(y1, y2),
    'ad': lambda x1, x2, y1, y2: AndersonDarling()(y1, y2)
}

from multiprocess import Pool

# Run all the correlation metrics in parallel
pool = Pool()
results = pool.map(lambda x: generate(x[0], x[1]), comp_corr.items())

for res in results:
    plot_corr(res)


# Plot a basic example of the shifts in velocity space the calculations will
# undergo
res = generate("basic_plot_shift.png",
               lambda x1, x2, y1, y2: 0)
plot_shift(res)


from spectacle.analysis.statistics import delta_v_90

res = generate("delta_v_90",
               lambda x1, x2, y1, y2:
               (delta_v_90(x1, y1)[1].value - delta_v_90(x1, y1)[0].value) -
               (delta_v_90(x2, y2)[1].value - delta_v_90(x2, y2)[0].value))

plot_corr(res)


# In[15]:


# from spectacle.analysis.statistics import delta_v_90


# velocity = np.linspace(-400, 400, 201) * u.Unit('km/s')

# f, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, sharex=True)
# f.subplots_adjust(hspace=0)

# def get_spec(v_dop, col_dens, masked=True):
#     line1 = dict(lambda_0=1.21567010E+03 * u.Angstrom,
#                  v_doppler=v_dop,
#                  column_density=col_dens)

#     spectrum1 = Spectrum1D(center=line1['lambda_0']).add_line(**line1)

#     if masked:
#         x1_t, y1_t = Masker(abs_tol=1e-5)(velocity, spectrum1.tau(velocity))
#     else:
#         x1_t, y1_t = velocity, spectrum1.tau(velocity)

#     return x1_t, y1_t


# def plot_dv90(x, y, ax, with_lines=True, title=None, x_label=None, y_label=None, *args, **kwargs):
#     dv = delta_v_90(x, y)

#     ax.step(x, y, *args, **kwargs)
    
#     if title is not None:
#         ax.set_title(title)
        
#     if x_label is not None:
#         ax.set_xlabel(x_label)
        
#     if y_label is not None:
#         ax.set_ylabel(y_label)
    
#     if with_lines:
#         ax.axvline(dv[0].value, linestyle=':', color=sns.color_palette()[2], alpha=0.4)
#         ax.axvline(dv[1].value, linestyle=':', color=sns.color_palette()[2], alpha=0.4)
        
#     ax.legend(loc=0)

# # plot_dv90(x1_f, y1_f, ax1)
# # plot_dv90(x2_f, y2_f, ax2)

# plot_dv90(*get_spec(v_doppler_range[0], column_density_range[0], False), 
#           ax1, alpha=0.4, with_lines=False, 
#           label="Full", title="Variations in $N$")
# plot_dv90(*get_spec(v_doppler_range[0], column_density_range[1], False), 
#           ax2, alpha=0.4, with_lines=False, y_label="Tau")
# plot_dv90(*get_spec(v_doppler_range[0], column_density_range[2], False), 
#           ax3, alpha=0.4, with_lines=False,
#           x_label="$v$ [\si{km/s}]")

# plot_dv90(*get_spec(v_doppler_range[0], column_density_range[0]), ax1, 
#           label="Line\nRegion")
# plot_dv90(*get_spec(v_doppler_range[0], column_density_range[1]), ax2)
# plot_dv90(*get_spec(v_doppler_range[0], column_density_range[2]), ax3)

# plot_dv90(*get_spec(v_doppler_range[0], column_density_range[0], False), 
#           ax4, alpha=0.4, with_lines=False, 
#           label="Full", title="Variations in $b$")
# plot_dv90(*get_spec(v_doppler_range[1], column_density_range[0], False), 
#           ax5, alpha=0.4, with_lines=False)
# plot_dv90(*get_spec(v_doppler_range[2], column_density_range[0], False), 
#           ax6, alpha=0.4, with_lines=False,
#           x_label="$v$ [\si{km/s}]")

# plot_dv90(*get_spec(v_doppler_range[0], column_density_range[0]), ax4,
#          label="Line\nRegion")
# plot_dv90(*get_spec(v_doppler_range[1], column_density_range[0]), ax5)
# plot_dv90(*get_spec(v_doppler_range[2], column_density_range[0]), ax6)

# # plot_dv90(x1_d, y1_d, ax5)
# # plot_dv90(x2_d, y2_d, ax6)


# In[ ]:




