import astropy.io.fits as fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import exoscene.star

from pastis.config import CONFIG_PASTIS
import pastis.util as util
from pastis.simulators.scda_telescopes import HexRingAPLC

from ultra.close_loop_analysis import req_closedloop_calc_batch
from ultra.config import CONFIG_ULTRA
from ultra.util import calculate_sensitivity_matrices
# from ultra.plotting import plot_iter_wf


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
    plt.tick_params(axis='both', which='major', length=6, width=2)
    plt.tick_params(axis='both', which='minor', length=6, width=2)
    plt.grid()
    plt.savefig(os.path.join(data_dir, 'contrast_wf_%s_%d_%d_%d_%d.png' % (C_TARGET, wavescale_min, wavescale_max, wavescale_step, Vmag)))


if __name__ == '__main__':

    C_TARGET = 1e-11

    # Set number of rings
    NUM_RINGS = 2

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

    data_dir = '/Users/asahoo/Desktop/data_repos/harris_data/2023-05-08T19-29-23_hexringtelescope'

    # Parameters for Temporal Ananlysis
    sptype = CONFIG_ULTRA.get('target', 'sptype')
    Vmag = CONFIG_ULTRA.getfloat('target', 'Vmag')

    minlam = CONFIG_ULTRA.getfloat('target', 'minlam') * u.nanometer
    maxlam = CONFIG_ULTRA.getfloat('target', 'maxlam') * u.nanometer

    dark_current = CONFIG_ULTRA.getfloat('detector', 'dark_current')
    CIC = CONFIG_ULTRA.getfloat('detector', 'dark_current')

    # Set close loop parameters.
    detector_noise = CONFIG_ULTRA.getfloat('detector', 'detector_noise')
    mus = np.genfromtxt(os.path.join(data_dir, 'mus_Hex_2_1e-11.csv'), delimiter=',')

    # Get the efields at wfs and science plane.
    efield_science_real = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_coron_real.fits'))
    efield_science_imag = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_coron_imag.fits'))
    efield_wfs_real = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_obwfs_real.fits'))
    efield_wfs_imag = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_obwfs_imag.fits'))
    ref_coron = fits.getdata(os.path.join(data_dir, 'ref_e0_coron.fits'))
    ref_obwfs = fits.getdata(os.path.join(data_dir, 'ref_e0_wfs.fits'))

    sensitivity_matrices = calculate_sensitivity_matrices(ref_coron, ref_obwfs, efield_science_real,
                                                          efield_science_imag, efield_wfs_real, efield_wfs_imag,
                                                          subsample_factor=8)
    g_coron = sensitivity_matrices['senitivity_image_plane']
    g_wfs = sensitivity_matrices['sensitvity_wfs_plane']
    e0_coron = sensitivity_matrices['ref_image_plane']
    e0_wfs = sensitivity_matrices['ref_wfs_plane']

    # Compute Star flux.
    Vmag = 4

    niter = CONFIG_ULTRA.getint('close_loop', 'niter')
    TimeMinus = -2.5
    TimePlus = 5.5
    Ntimes = CONFIG_ULTRA.getint('close_loop', 'Ntimes')

    npup = int(np.sqrt(tel.pupil_grid.x.shape[0]))
    star_flux = exoscene.star.bpgs_spectype_to_photonrate(spectype=sptype, Vmag=Vmag,
                                                          minlam=minlam.value, maxlam=maxlam.value)
    Nph = star_flux.value * 15 ** 2 * np.sum(tel.apodizer ** 2) / npup ** 2
    flux = Nph

    Qharris = np.diag(np.asarray(mus ** 2))

    unaberrated_coro_psf, ref = tel.calc_psf(ref=True, display_intermediate=False, norm_one_photon=True)
    norm = np.max(ref)

    wavescale_min = 15  # TODO: plot works only for 7 wavescale values, chose the stepsize accordingly.
    wavescale_max = 155
    wavescale_step = 20
    result_wf_test = []
    for wavescale in range(wavescale_min, wavescale_max, wavescale_step):
        print('recurssive close loop batch estimation and wavescale %f' % wavescale)
        niter = 10
        timer1 = time.time()
        StarMag = 0.0
        for tscale in np.logspace(TimeMinus, TimePlus, Ntimes):
            Starfactor = 10 ** (-StarMag / 2.5)
            print(tscale)
            tmp0 = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                             detector_noise, tscale, flux * Starfactor,
                                             0.0001 * wavescale ** 2 * Qharris,
                                             niter, tel.dh_mask, norm)
            tmp1 = tmp0['averaged_hist']
            n_tmp1 = len(tmp1)
            result_wf_test.append(tmp1[n_tmp1 - 1])

    data_dir2 ='/Users/asahoo/Desktop/data_repos/paper2/texp_vs_drift/2_Hex/data_csv'

    np.savetxt(os.path.join(data_dir2,
                            'contrast_wf_%s_%d_%d_%d_%d.csv' % (C_TARGET, wavescale_min, wavescale_max, wavescale_step, Vmag)),
               result_wf_test, delimiter=',')

    contrast_floor = 4.17133735821727E-11   # obtained from data saved in tolerance_table.csv

    data_dir3 = '/Users/asahoo/Desktop/data_repos/paper2/texp_vs_drift/2_Hex/data_png'
    plot_iter_wf(Qharris, wavescale_min, wavescale_max, wavescale_step,
                 TimeMinus, TimePlus, Ntimes, result_wf_test, contrast_floor, C_TARGET, Vmag, data_dir3)
