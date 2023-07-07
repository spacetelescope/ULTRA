from astropy.io import fits
import astropy.units as u
import exoscene.star
import numpy as np
import os
import pandas as pd
import time

from pastis.config import CONFIG_PASTIS
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldLuvoirA
from pastis.pastis_analysis import calculate_segment_constraints
from pastis.util import dh_mean

from ultra.config import CONFIG_ULTRA
from ultra.util import calculate_sensitivity_matrices
from ultra.close_loop_analysis import req_closedloop_calc_batch
from ultra.plotting import plot_iter_wf, plot_multimode_surface_maps, plot_pastis_matrix

if __name__ == '__main__':

    # Define the type of WFE.
    WHICH_DM = 'harris_seg_mirror'
    APLC_DESIGN = 'small'

    # Define target contrast
    C_TARGET = CONFIG_ULTRA.getfloat('target', 'Vmag')

    # Parameters for Temporal Ananlysis
    sptype = CONFIG_ULTRA.get('target', 'sptype')
    Vmag = CONFIG_ULTRA.getfloat('target', 'Vmag')

    minlam = CONFIG_ULTRA.getfloat('target', 'minlam') * u.nanometer
    maxlam = CONFIG_ULTRA.getfloat('target', 'maxlam') * u.nanometer

    dark_current = CONFIG_ULTRA.getfloat('detector', 'dark_current')
    CIC = CONFIG_ULTRA.getfloat('detector', 'dark_current')

    # Instantiate the simulator and run PASTIS matrix
    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)
        NUM_MODES = 5  # TODO: works only for thermal modes currently

    run_matrix = MatrixEfieldLuvoirA(which_dm=WHICH_DM, dm_spec=DM_SPEC, design=APLC_DESIGN,
                                     calc_science=True, calc_wfs=True,
                                     initial_path=CONFIG_PASTIS.get('local', 'local_data_path'),
                                     norm_one_photon=True)

    run_matrix.calc()
    data_dir = run_matrix.overall_dir
    print(f'All saved to {data_dir}.')

    tel = run_matrix.simulator

    tel.harris_sm.flatten()
    unaberrated_coro_psf, ref = tel.calc_psf(ref=True, display_intermediate=False, norm_one_photon=True)
    norm = np.max(ref)
    dh_intensity = (unaberrated_coro_psf / norm) * tel.dh_mask
    contrast_floor1 = np.mean(dh_intensity[np.where(tel.dh_mask != 0)])

    unaber_psf = fits.getdata(os.path.join(data_dir, 'unaberrated_coro_psf.fits'))   # already normalized to max of direct pdf
    dh_mask_shaped = tel.dh_mask.shaped
    contrast_floor = dh_mean(unaber_psf, dh_mask_shaped)

    pastis_matrix = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'pastis_matrix.fits'))
    plot_pastis_matrix(pastis_matrix, data_dir, vcenter=0, vmin=-0.005 * 1e-7, vmax=0.005 * 1e-7)

    # Calculate static tolerances.
    mus = calculate_segment_constraints(pastis_matrix, c_target=C_TARGET, coronagraph_floor=0)
    np.savetxt(os.path.join(data_dir, 'mus_harris_%s.csv' % C_TARGET), mus, delimiter=',')

    plot_multimode_surface_maps(tel, mus, NUM_MODES, 'harris_seg_mirror', cmin=-5, cmax=5, data_dir=data_dir, fname=None)

    # Get the efields at wfs and science plane.
    efield_science_real = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_coron_real.fits'))
    efield_science_imag = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_coron_imag.fits'))
    efield_wfs_real = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_obwfs_real.fits'))
    efield_wfs_imag = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_obwfs_imag.fits'))
    ref_coron = fits.getdata(os.path.join(data_dir, 'ref_e0_coron.fits'))
    ref_obwfs = fits.getdata(os.path.join(data_dir, 'ref_e0_wfs.fits'))

    # Compute sensitivity matrices.
    sensitivity_matrices = calculate_sensitivity_matrices(ref_coron, ref_obwfs, efield_science_real,
                                                          efield_science_imag,
                                                          efield_wfs_real, efield_wfs_imag, subsample_factor=8)
    g_coron = sensitivity_matrices['senitivity_image_plane']
    g_wfs = sensitivity_matrices['sensitvity_wfs_plane']
    e0_coron = sensitivity_matrices['ref_image_plane']
    e0_wfs = sensitivity_matrices['ref_wfs_plane']

    # Compute Star flux.
    npup = int(np.sqrt(tel.pupil_grid.x.shape[0]))
    star_flux = exoscene.star.bpgs_spectype_to_photonrate(spectype=sptype, Vmag=Vmag,
                                                          minlam=minlam.value, maxlam=maxlam.value)
    Nph = star_flux.value * tel.diam ** 2 * np.sum(tel.apodizer ** 2) / npup ** 2
    flux = Nph

    # Set close loop parameters.
    detector_noise = 0.0
    niter = 10
    TimeMinus = -2
    TimePlus = 5.5
    Ntimes = 20
    Nwavescale = 8
    Nflux = 3
    Qharris = np.diag(np.asarray(mus ** 2))

    res = np.zeros([Ntimes, Nwavescale, Nflux, 1])
    result_wf_test = []

    for wavescale in range(1, 15, 2):
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

    plot_iter_wf(Qharris, -2, 5.5, 20, result_wf_test, contrast_floor, C_TARGET, Vmag, data_dir)

    # Final Individual Tolerance allocation across 5 modes in units of pm.
    coeffs_table = np.zeros([NUM_MODES, tel.nseg])  # TODO: coeffs_table = sort_1d_mus_per_seg(mus, NUM_MODES, tel.nseg)
    for qq in range(NUM_MODES):
        for kk in range(tel.nseg):
            coeffs_table[qq, kk] = mus[qq + kk * NUM_MODES]

    # check temporal maps for individual modes
    opt_wavescale = 13  # This is wavescale value corresponding to lowest contrast from the graph.
    Q_total = 1e3 * np.sqrt(np.mean(np.diag(0.0001 * opt_wavescale ** 2 * Qharris)))  # in pm
    Q_individual = []
    for mode in range(NUM_MODES):
        Q_modes = 1e3 * np.sqrt(np.mean(0.0001 * opt_wavescale ** 2 * (coeffs_table[mode] ** 2)))  # in pm
        Q_individual.append(Q_modes)

    Q_individuals = np.array(Q_individual)
    print(Q_total, Q_individuals)

    # Sort to individual modes
    num_actuators = NUM_MODES * tel.nseg
    coeffs_numaps = np.zeros([NUM_MODES, num_actuators])
    for qq in range(NUM_MODES):
        coeffs_tmp = np.zeros([num_actuators])
        for kk in range(tel.nseg):
            coeffs_tmp[qq + kk * NUM_MODES] = mus[qq + (kk) * NUM_MODES]  # arranged per modal basis
        coeffs_numaps[qq] = coeffs_tmp  # arranged into 5 groups of 600 elements and in units of nm

    Qharris_individual = []
    for mode in range(NUM_MODES):
        Qharris_per_mode = np.diag(np.asarray(coeffs_numaps[mode]**2))
        Qharris_individual.append(Qharris_per_mode)

    Qmode = np.array(Qharris_individual)

    opt_tscale = 30
    c_total = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise, detector_noise,
                                        opt_tscale, flux * Starfactor, 0.0001 * opt_wavescale**2 * Qharris, niter, tel.dh_mask, norm)

    resultant_c_total = []
    c0 = c_total['averaged_hist']
    n_tmp1 = len(c0)
    resultant_c_total.append(c0[n_tmp1 - 1])
    print(resultant_c_total[0] - contrast_floor)
    c0 = resultant_c_total[0] - contrast_floor

    c_per_modes = []
    for mode in range(NUM_MODES):
        contrast = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                               detector_noise, opt_tscale, flux * Starfactor, 0.0001 * opt_wavescale**2 * Qmode[mode],
                                               niter, tel.dh_mask, norm)
        resultant_contrast = []
        c1 = contrast['averaged_hist']
        n_tmp1 = len(c1)
        resultant_contrast.append(c1[n_tmp1 - 1])
        c_per_modes.append(resultant_contrast[0] - contrast_floor)

    contrast_per_mode = np.array(c_per_modes)

    df = pd.DataFrame()
    df['Harris Modes'] = ['faceplate silvered', 'bulk', 'gradient radial', 'gradient X', 'gradient z', 'total']
    df['Tolerances in pm'] = [Q_individuals[0], Q_individuals[1], Q_individuals[2],
                              Q_individuals[3], Q_individuals[4], Q_total]
    df['Contrast'] = [contrast_per_mode[0], contrast_per_mode[1], contrast_per_mode[2], contrast_per_mode[3], contrast_per_mode[4], c0]
    print(df)
    df.to_csv(os.path.join(data_dir, 'tolerance_table.csv'))

