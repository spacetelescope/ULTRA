from astropy.io import fits
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt

from pastis.config import CONFIG_PASTIS
from pastis.simulators.scda_telescopes import HexRingAPLC
import pastis.util as util
from pastis.util import dh_mean


from ultra.config import CONFIG_ULTRA
from ultra.util import sort_1d_mus_per_actuator

if __name__ == '__main__':
    # Set number of rings
    NUM_RINGS = 1

    # Define the type of WFE.
    WHICH_DM = 'harris_seg_mirror'

    # Define target contrast
    C_TARGET = CONFIG_ULTRA.getfloat('target', 'contrast')

    data_dir = CONFIG_PASTIS.get('local', 'local_data_path')
    resDir = util.create_data_path(data_dir, 'mcmc_simulations')
    os.makedirs(resDir, exist_ok=True)

    optics_dir = os.path.join(util.find_repo_location(), 'data', 'SCDA')
    sampling = 4

    tel = HexRingAPLC(optics_dir, NUM_RINGS, sampling)

    unaberrated_coro_psf, ref = tel.calc_psf(ref=True, display_intermediate=False, norm_one_photon=True)
    norm = np.max(ref)

    nm_aber = 1e-9  # in m

    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)
        NUM_MODES = 5

        tel.create_segmented_harris_mirror(fpath, pad_orientations, thermal=True, mechanical=False, other=False)
        num_actuators = len(tel.harris_sm.actuators)

    local_data_path = CONFIG_ULTRA.get('local_path', 'local_data_path')
    mus_csv_data_path = os.path.join(local_data_path, '2023-10-02T17-18-48_hexringtelescope/mus_Hex_1_1e-11.csv')
    mus = np.genfromtxt(mus_csv_data_path, delimiter=',')

    dh_mask_shaped = tel.dh_mask.shaped
    unaber_psf = fits.getdata(os.path.join(local_data_path, '2023-10-02T17-18-48_hexringtelescope', 'unaberrated_coro_psf.fits'))
    contrast_floor = dh_mean(unaber_psf, dh_mask_shaped)

    coeffs_numaps = sort_1d_mus_per_actuator(mus, NUM_MODES, tel.nseg)

    MODE = 0

    del_contrast = []
    aberrated_contrast = []
    rms_wfs = []
    num_iterations = 1000
    for i in range(1, num_iterations):
        # Add pupil phase aberrations, use 'mu_maps' to poke all segment either with thermal mode
        actuators = np.zeros(num_actuators)
        actuators = np.random.normal(0, coeffs_numaps[MODE] * nm_aber, num_actuators)
        rms = np.sqrt(np.mean(actuators ** 2)) * 1e12 * np.sqrt(NUM_MODES)  # wfe in pm

        # Compute the aberrated psf
        tel.harris_sm.actuators = actuators / 2
        aberrated_coro_psf_t, inter_t = tel.calc_psf(display_intermediate=False, return_intermediate='efield', norm_one_photon=True)

        # compute total dark hole intensity due to pertubation, and change in dh intensity
        dh_intensity = ((np.square(aberrated_coro_psf_t.amplitude)) / norm) * tel.dh_mask
        contrast_floor_aber = np.mean(dh_intensity[np.where(tel.dh_mask != 0)])
        delta_contrast = contrast_floor_aber - contrast_floor

        # convert pupil and focal plane to numpy array for fits file storage
        # pupil_phase = np.array(np.reshape(inter_t['harris_seg_mirror'].phase, (1000, 1000)))
        # focal_int = np.array(np.reshape(aberrated_coro_psf_t.amplitude, (115, 115)))
        # focal_cont = (np.square(focal_int)) / norm

        # Archive
        del_contrast.append(delta_contrast)
        aberrated_contrast.append(contrast_floor_aber)
        rms_wfs.append(rms)
        print(rms, contrast_floor_aber, delta_contrast)

        # For all harris modes:
        # mean_aberration = np.sqrt(np.mean((mu_map_harris*nm_aber)**2))*1e12 # in units of pm

        # For only one thermal mode:
        mean_aberration = np.sqrt(np.mean((coeffs_numaps[MODE] * nm_aber) ** 2)) * 1e12 * np.sqrt(NUM_MODES)  # in units of pm

        print(mean_aberration)

        plt.figure(figsize=(40, 15))
        plt.subplot(1, 3, 1)
        _, bins, _ = plt.hist(rms_wfs, 100, density=1, alpha=1)
        mu, sigma = scipy.stats.norm.fit(rms_wfs)
        best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
        plt.plot(bins, best_fit_line, lw='4', label=r'$\mu$ =%.2f, $\sigma$= %.2f' % (mu, sigma))
        plt.axvline(mean_aberration, c='r', ls='-.', lw='3')
        plt.xlabel("RMS Wavefront Error (in pm)", fontsize=20)
        plt.ylabel("Frequency", fontsize=15)
        plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=20)
        plt.legend(fontsize=25)
        # plt.xlim(0,3)

        plt.subplot(1, 3, 2)
        _, bins, _ = plt.hist(np.log10(aberrated_contrast), 100, density=1, alpha=1)
        mu, sigma = scipy.stats.norm.fit(np.log10(aberrated_contrast))
        best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
        plt.plot(bins, best_fit_line, lw='4', label=r'$\mu$ =%.2f, $\sigma$= %.2f' % (mu, sigma))
        # plt.axvline(np.log10(contrast_floor), c='r', ls='-.', lw='3')
        plt.xlabel("Mean contrast in DH", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=20)
        plt.legend(fontsize=25)
        # plt.xlim(-10.37,-10.34)

        plt.subplot(1, 3, 3)
        _, bins, _ = plt.hist(np.log10(del_contrast), 100, density=1, alpha=1)
        mu, sigma = scipy.stats.norm.fit(np.log10(del_contrast))
        best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
        plt.plot(bins, best_fit_line, lw='4', label=r'$\mu$ =%.2f, $\sigma$= %.2f' % (mu, sigma))
        # plt.axvline(np.log10(c_target), c='r', ls='-.', lw='3')
        plt.xlabel("Change in contrast in DH", fontsize=20)
        plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=20)
        plt.ylabel("Frequency", fontsize=15)
        plt.legend(fontsize=25)
        plt.savefig(os.path.join(resDir, f'mcmc_hist_mode_{MODE}.png'))
