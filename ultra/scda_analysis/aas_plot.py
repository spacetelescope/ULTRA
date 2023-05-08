from astropy.io import fits
import numpy as np
import os

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

if __name__ == '__main__':

    # Set number of rings
    NUM_RINGS = 1

    # Define the type of WFE.
    WHICH_DM = 'harris_seg_mirror'

    # Define target contrast
    C_TARGET = 1e-10

    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)
        NUM_MODES = 5 # TODO: works only for thermal modes currently

    # run_matrix = MatrixEfieldHex(which_dm=WHICH_DM, dm_spec=DM_SPEC, num_rings=NUM_RINGS,
    #                              calc_science=True, calc_wfs=False,
    #                              initial_path=CONFIG_PASTIS.get('local', 'local_data_path'), norm_one_photon=True)
    #
    # run_matrix.calc()
    # data_dir = run_matrix.overall_dir
    # print(f'All saved to {data_dir}.')
    #
    # tel = run_matrix.simulator
    #
    # unaber_psf = fits.getdata(os.path.join(data_dir, 'unaberrated_coro_psf.fits'))  # already normalized to max of direct pdf
    # dh_mask_shaped = tel.dh_mask.shaped
    # contrast_floor = dh_mean(unaber_psf, dh_mask_shaped)

    # Calculate static tolerances.
    # pastis_matrix = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'pastis_matrix.fits'))
    # mus = calculate_segment_constraints(pastis_matrix, c_target=C_TARGET, coronagraph_floor=0)
    # np.savetxt(os.path.join(data_dir, 'mus_Hex_%d_%s.csv' % (NUM_RINGS, C_TARGET)), mus, delimiter=',')

    optics_dir = os.path.join(util.find_repo_location(), 'data', 'SCDA')
    sampling = 4
    data_dir = '/Users/asahoo/Desktop/data_repos/plots_aas241'

    # Thermal tolerance coefficients
    # mus5 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_aas241/mus_Hex_5_1e-10.csv', delimiter=',')
    # mus4 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_aas241/mus_Hex_4_1e-10.csv', delimiter=',')
    # mus3 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_aas241/mus_Hex_3_1e-10.csv', delimiter=',')
    # mus2 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_aas241/mus_Hex_2_1e-10.csv', delimiter=',')
    mus1 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_aas241/mus_Hex_1_1e-10.csv', delimiter=',')

    tel = HexRingAPLC(optics_dir, NUM_RINGS, sampling)

    harris_coeffs_table = np.zeros([NUM_MODES, tel.nseg])
    for qq in range(NUM_MODES):
        for kk in range(tel.nseg):
            harris_coeffs_table[qq, kk] = mus1[qq + kk * NUM_MODES]

    # Faceplate silvered
    tel2 = HexRingAPLC(optics_dir, NUM_RINGS, sampling)
    tel2.create_segmented_mirror(1)
    tel2.sm.flatten()
    tel2.sm.actuators = harris_coeffs_table[0]

    # Bulk
    tel3 = HexRingAPLC(optics_dir, NUM_RINGS, sampling)
    tel3.create_segmented_mirror(1)
    tel3.sm.flatten()
    tel3.sm.actuators = harris_coeffs_table[1]

    # Gradient Radial
    tel4 = HexRingAPLC(optics_dir, NUM_RINGS, sampling)
    tel4.create_segmented_mirror(1)
    tel4.sm.actuators = harris_coeffs_table[2]

    # Gradient X Lateral
    tel5 = HexRingAPLC(optics_dir, NUM_RINGS, sampling)
    tel5.create_segmented_mirror(1)
    tel5.sm.actuators = harris_coeffs_table[3]

    # Gradient z axial
    tel6 = HexRingAPLC(optics_dir, NUM_RINGS, sampling)
    tel6.create_segmented_mirror(1)
    tel6.sm.actuators = harris_coeffs_table[4]

    wavescale = 200

    plt.figure(figsize=(25, 6))
    plt.subplot(1, 5, 1)
    # plt.title("Faceplates Silvered", fontsize=10)
    plot_norm = TwoSlopeNorm(vcenter=10, vmin=0, vmax=20)
    hcipy.imshow_field((tel2.sm.surface)*1e3 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='YlOrRd')
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=10)
    # cbar.set_label("mK/s", fontsize=10)

    plt.subplot(1, 5, 2)
    # plt.title("Bulk", fontsize=10)
    plot_norm = TwoSlopeNorm(vcenter=10, vmin=0, vmax=20)
    hcipy.imshow_field((tel3.sm.surface)*1e3 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='YlOrRd')
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=10)
    # cbar.set_label("mK/s", fontsize=10)

    plt.subplot(1, 5, 3)
    # plt.title("Gradiant Radial", fontsize=10)
    plot_norm = TwoSlopeNorm(vcenter=10, vmin=0, vmax=20)
    hcipy.imshow_field((tel4.sm.surface)*1e3 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm,  cmap='YlOrRd')
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=10)
    # cbar.set_label("mK/s", fontsize=10)

    plt.subplot(1, 5, 4)
    # plt.title("Gradient X lateral", fontsize=10)
    plot_norm = TwoSlopeNorm(vcenter=10, vmin=0, vmax=20)
    hcipy.imshow_field((tel5.sm.surface)*1e3 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='YlOrRd')
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=10)
    # cbar.set_label("mK/s", fontsize=10)

    plt.subplot(1, 5, 5)
    plt.figure(figsize=(25, 6))
    # plt.title("Gradient Z axial", fontsize=10)
    plot_norm = TwoSlopeNorm(vcenter=10, vmin=0, vmax=20)
    hcipy.imshow_field((tel6.sm.surface)*1e3 * np.sqrt(0.0001 * wavescale ** 2), norm=plot_norm, cmap='YlOrRd')
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("mK/s", fontsize=15)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'harris_thermal_maps_1_20.png'))
