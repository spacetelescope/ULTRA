"""
This module contains various handy plotting functions used by launcher scripts, or notebooks.
"""

from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os

import hcipy
from pastis.util import sort_1d_mus_per_actuator

from ultra.config import CONFIG_ULTRA


def plot_multimode_surface_maps(tel, mus, num_modes, mirror, cmin, cmax, data_dir=None, fname=None):
    """Creates surface deformation maps (not WFE) for localized wavefront aberrations.

    The input mode coefficients 'mus' are in units of *WFE* and need to be grouped by segment, meaning the array holds
    the mode coefficients as:
        mode1 on seg1, mode2 on seg1, ..., mode'nmodes' on seg1, mode1 on seg2, mode2 on seg2 and so on.

    Parameters
    ----------
    tel : class instance of internal simulator
        the simulator to plot the surface maps for
    mus : 1d array
        1d array of standard deviations for all modes on each segment, in nm WFE
    num_modes : int
        number of local modes used to poke each segment
    mirror : str
        'harris_seg_mirror' or 'seg_mirror', segmented mirror of simulator 'tel' to use for plotting
    cmin : float
        minimum value for colorbar
    cmax : float
        maximum value for colorbar
    data_dir : str, default None
        path to save the plots; if None, then not saved to disk
    fname : str, default None
        file name for surface maps saved to disk
    """
    if fname is None:
        fname = f'surface_on_{mirror}'

    mus_per_actuator = sort_1d_mus_per_actuator(mus, num_modes, tel.nseg)  # in nm

    mu_maps = []
    for mode in range(num_modes):
        coeffs = mus_per_actuator[mode]
        if mirror == 'harris_seg_mirror':
            tel.harris_sm.actuators = coeffs * 1e-9 / 2  # in meters of surface
            mu_maps.append(tel.harris_sm.surface)  # in m
        if mirror == 'seg_mirror':
            tel.sm.actuators = coeffs * 1e-9 / 2  # in meters of surface
            mu_maps.append(tel.sm.surface)  # in m

    plot_norm = TwoSlopeNorm(vcenter=0, vmin=cmin, vmax=cmax)
    for i in range(num_modes):
        plt.figure(figsize=(7, 5))
        hcipy.imshow_field((mu_maps[i]) * 1e12, norm=plot_norm, cmap='RdBu')
        plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Surface (pm)", fontsize=10)
        plt.tight_layout()

        if data_dir is not None:
            os.makedirs(os.path.join(data_dir, 'mu_maps'), exist_ok=True)
            image_name = fname + f'_mode_{i}.pdf'
            plt.savefig(os.path.join(data_dir, 'mu_maps', image_name))


def plot_iter_wf(variance_q, contrasts, contrast_floor, data_dir):
    """Plots 'triangular' contrast curves for different wfs exposure time.

    This function generates and saves contrast vs texp,
    and calculates the optimal wfs time scale and delta wavefront error scale.

    Parameters
    ----------
    variance_q : numpy 2d array
        a square diagonal matrix, with static tolerances as diagonal elements.
    contrasts : list
        list of contrasts for different wfs exposure time.
    contrast_floor : float
        static coronagraphic contrast without any external aberration.
    data_dir :  str
        path to save the plot.

    Returns
    -------
    contrast_minimum : float
        minimum contrast corresponding to
    t_wfs_optimal : float
        optimal wavefront sensing time (in secs)
    wavescale_optimal : int
        optimal scaling factor to be later mulitplied to variance_q, fractional scale to give the total
        wavefront error drift in pm/s.
    """
    wavescale_min = CONFIG_ULTRA.getint('close_loop', 'wavescale_min')
    wavescale_max = CONFIG_ULTRA.getint('close_loop', 'wavescale_max')
    wavescale_step = CONFIG_ULTRA.getint('close_loop', 'wavescale_step')
    fractional_scale = CONFIG_ULTRA.getfloat('close_loop', 'fractional_scale')

    TimeMinus = CONFIG_ULTRA.getfloat('close_loop', 'TimeMinus')
    TimePlus = CONFIG_ULTRA.getfloat('close_loop', 'TimePlus')
    Ntimes = CONFIG_ULTRA.getint('close_loop', 'Ntimes')

    Vmag = CONFIG_ULTRA.getfloat('target', 'Vmag')
    C_TARGET = CONFIG_ULTRA.getfloat('target', 'contrast')

    delta_wf = []
    for wavescale in range(wavescale_min, wavescale_max, wavescale_step):
        wf = np.sqrt(np.mean(np.diag(fractional_scale * wavescale ** 2 * variance_q))) * 1e3
        delta_wf.append(wf)

    wavescale_vec = range(wavescale_min, wavescale_max, wavescale_step)

    texp = np.logspace(TimeMinus, TimePlus, Ntimes)
    font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 20}

    contrasts = np.asarray(contrasts)
    plt.figure(figsize=(15, 10))

    index_minima = []
    contrasts_minima = []
    t_minima = []
    for pp in range(0, len(wavescale_vec)):

        contrasts_subarray = contrasts[pp * Ntimes:(pp + 1) * Ntimes] - contrast_floor
        index_min = np.unravel_index(contrasts_subarray.argmin(), contrasts_subarray.shape)
        contrast_min = contrasts_subarray[index_min]
        t_min = texp[index_min]

        contrasts_minima.append(contrast_min)
        index_minima.append(index_min)
        t_minima.append(t_min)

        plt.title('Target contrast = %s, Vmag= %s' % (C_TARGET, Vmag), fontdict=font)
        plt.plot(texp, contrasts_subarray, label=r'$\Delta_{wf}= % .2f\ pm/s$' % (delta_wf[pp]))
        plt.xlabel("$t_{WFS}$ in secs", fontsize=20)
        plt.ylabel(r"$ \Delta $ contrast", fontsize=20)
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='upper center', fontsize=20)
        plt.tick_params(top=True, bottom=True, left=True, right=True, labelleft=True, labelbottom=True, labelsize=20)
        plt.tick_params(axis='both', which='major', length=10, width=2)
        plt.tick_params(axis='both', which='minor', length=6, width=2)
        plt.grid()

    delta_contrast_minima = abs((np.array(contrasts_minima) - C_TARGET))
    index_minimum = (np.unravel_index(delta_contrast_minima.argmin(), delta_contrast_minima.shape))[0]
    contrast_minimum = contrasts_minima[index_minimum]
    t_wfs_optimal = t_minima[index_minimum]
    wavescale_optimal = wavescale_vec[index_minimum]

    plt.savefig(os.path.join(data_dir, 'contrast_wf_%s_%d_%d_%d.png' % (C_TARGET, wavescale_min, wavescale_max, wavescale_step)))

    return contrast_minimum, t_wfs_optimal, wavescale_optimal


def plot_iter_mv(contrasts, contrast_floor, data_dir):
    """Plots 'triangular' contrast curves for different wfs exposures, iterating over stellar magnitudes.

    Parameters
    ----------
    contrasts : list
        list of contrasts for different wfs exposure time.
    contrast_floor : float
        static coronagraphic contrast without any external aberration.
    data_dir : str
        path to save the plot.
    """
    mv_min = CONFIG_ULTRA.getint('close_loop', 'mv_min')
    mv_max = CONFIG_ULTRA.getint('close_loop', 'mv_max')
    mv_step = CONFIG_ULTRA.getint('close_loop', 'mv_step')

    TimeMinus = CONFIG_ULTRA.getfloat('close_loop', 'TimeMinus')
    TimePlus = CONFIG_ULTRA.getfloat('close_loop', 'TimePlus')
    Ntimes = CONFIG_ULTRA.getint('close_loop', 'Ntimes')

    C_TARGET = CONFIG_ULTRA.getfloat('target', 'contrast')

    mv_list = []
    for mv in range(mv_min, mv_max, mv_step):
        mv_list.append(mv)

    texp = np.logspace(TimeMinus, TimePlus, Ntimes)
    contrasts = np.asarray(contrasts)
    plt.figure(figsize=(15, 10))

    for pp in range(0, len(mv_list)):
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 20}
        plt.title('Target contrast = %s' % (C_TARGET), fontdict=font)
        plt.plot(texp, contrasts[pp * Ntimes:(pp + 1) * Ntimes] - contrast_floor, label=r'$ m_{v}= %d $' % (mv_list[pp]))
        plt.xlabel("$t_{WFS}$ in secs", fontsize=20)
        plt.ylabel(r"$ \Delta $ contrast", fontsize=20)
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='upper center', fontsize=20)
        plt.tick_params(top=True, bottom=True, left=True, right=True, labelleft=True, labelbottom=True, labelsize=20)
        plt.tick_params(axis='both', which='major', length=10, width=2)
        plt.tick_params(axis='both', which='minor', length=6, width=2)
        plt.grid()

    plt.savefig(os.path.join(data_dir, 'contrast_iter_mv.png'))


def plot_pastis_matrix(pastis_matrix, data_dir, vcenter, vmin, vmax):
    """Plots PASTIS matrix.

    Parameters
    ----------
    pastis_matrix : numpy 2d array
        a square PASTIS matrix.
    data_dir : str
        path to save the plot
    vcenter : float
        TwoSlopeNorm's vcenter
    vmin : float
        TwoSlopeNorm's vmin
    vmax : float
        TwoSlopeNorm's vmax
    """
    clist = [(0.1, 0.6, 1.0), (0.05, 0.05, 0.05), (0.8, 0.5, 0.1)]
    blue_orange_divergent = LinearSegmentedColormap.from_list("custom_blue_orange", clist)

    plt.figure(figsize=(10, 8))
    norm_mat = TwoSlopeNorm(vcenter, vmin, vmax)
    plt.imshow(pastis_matrix, cmap=blue_orange_divergent, norm=norm_mat)
    plt.title(r"PASTIS matrix $M$", fontsize=20)
    plt.xlabel("Mode Index", fontsize=20)
    plt.ylabel("Mode Index", fontsize=20)
    plt.tick_params(labelsize=15)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label(r"in units of ${contrast\ per\ {nm}^2}$", fontsize=15)
    plt.tight_layout()

    plt.savefig(os.path.join(data_dir, 'pastis_matrix.png'))
