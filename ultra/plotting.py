import hcipy
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from pastis.util import sort_1d_mus_per_actuator


def plot_multimode_surface_maps(tel, mus, num_modes, mirror, cmin, cmax, data_dir=None, fname=None):
    """Creates surface deformation maps (not WFE) for localized wavefront aberrations.
    The input mode coefficients 'mus' are in units of *WFE* and need to be grouped by segment, meaning the array holds
    the mode coefficients as:
        mode1 on seg1, mode2 on seg1, ..., mode'nmodes' on seg1, mode1 on seg2, mode2 on seg2 and so on.
    Parameters:
    -----------
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
            fname += f'_mode_{i}.png'
            plt.savefig(os.path.join(data_dir, fname))


def plot_iter_wf(Qharris, wavescale_min, wavescale_max, wavescale_step,
                 TimeMinus, TimePlus, Ntimes, result_wf_test, contrast_floor, C_TARGET, Vmag, data_dir):
    delta_wf = []
    for wavescale in range(wavescale_min, wavescale_max, wavescale_step):
        wf = np.sqrt(np.mean(np.diag(0.0001 * wavescale ** 2 * Qharris))) * 1e3
        delta_wf.append(wf)

    texp = np.logspace(TimeMinus, TimePlus, Ntimes)
    font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 20}

    result_wf_test = np.asarray(result_wf_test)
    plt.figure(figsize=(15, 10))
    plt.title('Target contrast = %s, Vmag= %s' % (C_TARGET, Vmag), fontdict=font)
    plt.plot(texp, result_wf_test[0:20] - contrast_floor, label=r'$\Delta_{wf}= % .2f\ pm/s$' % (delta_wf[0]))
    plt.plot(texp, result_wf_test[20:40] - contrast_floor, label=r'$\Delta_{wf}= % .2f\ pm/s$' % (delta_wf[1]))
    plt.plot(texp, result_wf_test[40:60] - contrast_floor, label=r'$\Delta_{wf}= % .2f\ pm/s$' % (delta_wf[2]))
    plt.plot(texp, result_wf_test[60:80] - contrast_floor, label=r'$\Delta_{wf}= % .2f\ pm/s$' % (delta_wf[3]))
    plt.plot(texp, result_wf_test[80:100] - contrast_floor, label=r'$\Delta_{wf}= % .2f\ pm/s$' % (delta_wf[4]))
    plt.plot(texp, result_wf_test[100:120] - contrast_floor, label=r'$\Delta_{wf}= %.2f\ pm/s$' % (delta_wf[5]))
    plt.plot(texp, result_wf_test[120:140] - contrast_floor, label=r'$\Delta_{wf}= % .2f\ pm/s$' % (delta_wf[6]))
    plt.xlabel("$t_{WFS}$ in secs", fontsize=20)
    plt.ylabel(r"$ \Delta $ contrast", fontsize=20)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='upper center', fontsize=20)
    plt.tick_params(top=True, bottom=True, left=True,
                    right=True, labelleft=True, labelbottom=True,
                    labelsize=20)
    plt.tick_params(axis='both', which='major', length=10, width=2)
    plt.tick_params(axis='both', which='minor', length=6, width=2)
    plt.grid()
    plt.savefig(os.path.join(data_dir, 'contrast_wf_%s_%d_%d_%d.png' % (C_TARGET, wavescale_min, wavescale_max, wavescale_step)))


def plot_pastis_matrix(pastis_matrix, data_dir, vcenter, vmin, vmax):
    """
    Parameters
    ----------
    pastis_matrix : 2d array
        PASTIS matrix, where each element is stored in units of contrast/nm**2.
    data_dir : str
        path to save the image
    vcenter : float
        center of matplotlib colorbar
    vmin : float
        minimum extent of matplotlib colorbar
    vmax : float
        maximum extent of matplotlib colorbar
    """
    clist = [(0.1, 0.6, 1.0), (0.05, 0.05, 0.05), (0.8, 0.5, 0.1)]
    blue_orange_divergent = LinearSegmentedColormap.from_list("custom_blue_orange", clist)

    list_of_ticks = np.arange(0, pastis_matrix.shape[0], 20)

    plt.figure(figsize=(6, 4), dpi=150)
    norm_mat = TwoSlopeNorm(vcenter, vmin, vmax)
    plt.imshow(pastis_matrix, origin='lower', cmap=blue_orange_divergent, norm=norm_mat)
    plt.title(r"PASTIS matrix $M$", fontsize=15)
    plt.xlabel("Mode Index", fontsize=15)
    plt.ylabel("Mode Index", fontsize=15)
    plt.tick_params(labelsize=10)
    plt.xticks(list_of_ticks)
    plt.yticks(list_of_ticks)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label(r"in units of ${contrast\ per\ {nm}^2}$", fontsize=10)
    plt.tight_layout()

    plt.savefig(os.path.join(data_dir, 'pastis_matrix.png'))
