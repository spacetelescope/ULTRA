import os
import hcipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from pastis.config import CONFIG_PASTIS
import pastis.util as util
from pastis.simulators.scda_telescopes import HexRingAPLC
from pastis.util import sort_1d_mus_per_actuator


def plot_multimode_surface_maps2(tel, mus, mirror, data_dir=None):
    """Creates surface deformation maps (not WFE) for only 5 kinds of localized wavefront aberrations.
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
    data_dir : str, default None
        path to save the plots; if None, then not saved to disk
    """

    mus_per_actuator = sort_1d_mus_per_actuator(mus, 5, tel.nseg)  # in nm

    mu_maps = []
    for mode in range(5):
        coeffs = mus_per_actuator[mode]
        if mirror == 'harris_seg_mirror':
            tel.harris_sm.actuators = coeffs * 1e-9 / 2  # in meters of surface
            mu_maps.append(tel.harris_sm.surface)  # in m
        if mirror == 'seg_mirror':
            tel.sm.actuators = coeffs * 1e-9 / 2  # in meters of surface
            mu_maps.append(tel.sm.surface)  # in m

    plt.figure(figsize=(32, 5))
    plt.subplot(1, 5, 1)
    plot_norm = TwoSlopeNorm(vcenter=0, vmin=-12, vmax=12)
    hcipy.imshow_field((mu_maps[0]) * 1e12 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='RdBu')
    plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True, labelsize=15)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("Surface (pm/s)", fontsize=15)
    plt.tight_layout()

    plt.subplot(1, 5, 2)
    plot_norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
    hcipy.imshow_field((mu_maps[1]) * 1e12 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='RdBu')
    plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True, labelsize=15)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("Surface (pm/s)", fontsize=15)
    plt.tight_layout()

    plt.subplot(1, 5, 3)
    plot_norm = TwoSlopeNorm(vcenter=0, vmin=-8, vmax=8)
    hcipy.imshow_field((mu_maps[2]) * 1e12 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='RdBu')
    plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True, labelsize=15)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("Surface (pm/s)", fontsize=15)
    plt.tight_layout()

    plt.subplot(1, 5, 4)
    plot_norm = TwoSlopeNorm(vcenter=0, vmin=-7, vmax=7)
    hcipy.imshow_field((mu_maps[3]) * 1e12 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='RdBu')
    plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True, labelsize=15)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("Surface (pm/s)", fontsize=15)
    plt.tight_layout()

    plt.subplot(1, 5, 5)
    plot_norm = TwoSlopeNorm(vcenter=0, vmin=-12, vmax=12)
    hcipy.imshow_field((mu_maps[4]) * 1e12 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='RdBu')
    plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True, labelsize=15)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("Surface (pm/s)", fontsize=15)
    plt.tight_layout()

    plt.savefig(os.path.join(data_dir, 'temporal_pm_%d_hex.pdf' % tel.num_rings))


if __name__ == '__main__':

    # Set number of rings
    NUM_RINGS = 1

    # Define the type of WFE.
    WHICH_DM = 'harris_seg_mirror'

    optics_dir = os.path.join(util.find_repo_location(), 'data', 'SCDA')
    sampling = 4

    tel = HexRingAPLC(optics_dir, NUM_RINGS, sampling)
    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)
        NUM_MODES = 5
    tel.create_segmented_harris_mirror(fpath, pad_orientations, thermal=True, mechanical=False, other=False)

    data_dir = '/Users/asahoo/Desktop/data_repos/paper2/temporal_pm_harris'

    mus1 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/paper2/mus_Hex_1_1e-11.csv', delimiter=',')
    # mus2 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/paper2/mus_Hex_2_1e-11.csv', delimiter=',')
    # mus3 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/paper2/mus_Hex_3_1e-11.csv', delimiter=',')
    # mus4 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/paper2/mus_Hex_4_1e-11.csv', delimiter=',')
    # mus5 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/paper2/mus_Hex_5_1e-11.csv', delimiter=',')

    wavescale = 60
    plot_multimode_surface_maps2(tel, mus1, mirror='harris_seg_mirror', data_dir=data_dir)
