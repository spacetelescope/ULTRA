import numpy as np
import os
from astropy.io import fits
from pastis.config import CONFIG_PASTIS
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldHex
from pastis.pastis_analysis import calculate_segment_constraints
from pastis.util import dh_mean
from pastis.util import sort_1d_mus_per_actuator
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import hcipy
from pastis.simulators.scda_telescopes import HexRingAPLC
import pastis.util as util

from ultra.config import CONFIG_ULTRA
from pastis.plotting import plot_multimode_surface_maps
from ultra.plotting import plot_multimode_surface_maps

if __name__ == '__main__':

    # Set number of rings
    NUM_RINGS = 1

    # Define the type of WFE.
    WHICH_DM = 'harris_seg_mirror'

    # Define target contrast
    C_TARGET = 1e-11

    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)
        NUM_MODES = 5 # TODO: works only for thermal modes currently

    run_matrix = MatrixEfieldHex(which_dm=WHICH_DM, dm_spec=DM_SPEC, num_rings=NUM_RINGS,
                                 calc_science=True, calc_wfs=False,
                                 initial_path=CONFIG_PASTIS.get('local', 'local_data_path'), norm_one_photon=True)

    run_matrix.calc()
    data_dir = run_matrix.overall_dir
    print(f'All saved to {data_dir}.')

    tel = run_matrix.simulator

    # unaber_psf = fits.getdata(os.path.join(data_dir, 'unaberrated_coro_psf.fits'))  # already normalized to max of direct pdf
    # dh_mask_shaped = tel.dh_mask.shaped
    # contrast_floor = dh_mean(unaber_psf, dh_mask_shaped)

    # Calculate static tolerances.
    # pastis_matrix = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'pastis_matrix.fits'))
    # mus = calculate_segment_constraints(pastis_matrix, c_target=C_TARGET, coronagraph_floor=0)
    # np.savetxt(os.path.join(data_dir, 'mus_Hex_%d_%s.csv' % (NUM_RINGS, C_TARGET)), mus, delimiter=',')

    # plot static tolerance maps
    # plot_multimode_surface_maps('dynamic', tel, mus, num_modes=NUM_MODES, mirror=WHICH_DM, cmin=-10, cmax=10, wavescale=80, data_dir=data_dir)

    # plot harris mode

    harris_mode_basis = tel.harris_sm

    plt.figure(figsize=(40, 20))
    plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    plt.title("Segment Level 1mK Faceplates Silvered", fontsize=30)
    plot_norm1 = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
    plt.imshow(np.reshape(harris_mode_basis[0], (1000, 1000))[163:263, 804:904], cmap='RdBu', norm=plot_norm1)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("pm", fontsize=30)

    plt.subplot2grid((2, 6), (0, 2), colspan=2)
    plt.title("Segment Level 1mK bulk", fontsize=30)
    plot_norm2 = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=1)
    plt.imshow(np.reshape(harris_mode_basis[1], (1000, 1000))[163:263, 804:904], cmap='RdBu', norm=plot_norm2)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("pm", fontsize=30)

    plt.subplot2grid((2, 6), (0, 4), colspan=2)
    plt.title("Segment Level 1mK gradient radial", fontsize=30)
    plot_norm3 = TwoSlopeNorm(vcenter=0, vmin=-1.5, vmax=1.5)
    plt.imshow(np.reshape(harris_mode_basis[2], (1000, 1000))[163:263, 804:904], cmap='RdBu', norm=plot_norm3)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("pm", fontsize=30)

    plt.subplot2grid((2, 6), (1, 1), colspan=2)
    plt.title("Segment Level 1mK gradient X lateral", fontsize=30)
    plot_norm4 = TwoSlopeNorm(vcenter=0, vmin=-2.5, vmax=2.5)
    plt.imshow(np.reshape(harris_mode_basis[3], (1000, 1000))[163:263, 804:904], cmap='RdBu', norm=plot_norm4)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("pm", fontsize=30)

    plt.subplot2grid((2, 6), (1, 3), colspan=2)
    plt.title("Segment Level 1mK gradient Z axial", fontsize=30)
    plot_norm5 = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3)
    plt.imshow(np.reshape(harris_mode_basis[4], (1000, 1000))[163:263, 804:904], cmap='RdBu', norm=plot_norm5)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("pm", fontsize=30)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'harris_modal_basis.png'))













































    # mus_per_actuator = sort_1d_mus_per_actuator(mus, 5, tel.nseg)  # in nm

    mus = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_aas241/mus_Hex_5_1e-10.csv', delimiter=',')

    mus_per_actuator = sort_1d_mus_per_actuator(mus, 5, tel.nseg)  # in nm

    mu_maps = []
    for mode in range(5):
        coeffs = mus_per_actuator[mode]
        tel.harris_sm.actuators = coeffs * 1e-9 / 2  # in meters of surface
        mu_maps.append(tel.harris_sm.surface)  # in m

    wavescale = 80

    plt.figure(figsize=(15, 10))
    #plt.figure(figsize=(25,4))
    #plt.subplot(1, 5, 1)
    ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    plot_norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
    hcipy.imshow_field((mu_maps[0]) * 1e12 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='RdBu')
    plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=10)
    # cbar.set_label("Surface (pm/s)", fontsize=10)

    #plt.subplot(1, 5, 2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    plot_norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
    hcipy.imshow_field((mu_maps[1]) * 1e12 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='RdBu')
    plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=10)
    # cbar.set_label("Surface (pm/s)", fontsize=10)

    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
    #plt.subplot(1, 5, 3)
    plot_norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
    hcipy.imshow_field((mu_maps[2]) * 1e12 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='RdBu')
    plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=10)
    # cbar.set_label("Surface (pm/s)", fontsize=10)

    ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
    #plt.subplot(1, 5, 4)
    plot_norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
    hcipy.imshow_field((mu_maps[3]) * 1e12 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='RdBu')
    plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=10)
    # cbar.set_label("Surface (pm/s)", fontsize=10)

    ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)
    #plt.subplot(1, 5, 5)
    plot_norm = TwoSlopeNorm(vcenter=0, vmin=-15, vmax=15)
    hcipy.imshow_field((mu_maps[4]) * 1e12 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='RdBu')
    plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Surface (pm/s)", fontsize=30)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'harris_tolerance_maps_5Hex.png'))