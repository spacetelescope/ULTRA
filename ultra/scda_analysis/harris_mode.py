"""
This is the main launcher script to do a full wavefront error budget analysis for L3 Harris mode for
all SCDA coronagraph designs.
"""


from astropy.io import fits
import astropy.units as u
import exoscene.star
import numpy as np
import os
import time

from pastis.config import CONFIG_PASTIS
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldHex
from pastis.pastis_analysis import calculate_segment_constraints
from pastis.util import dh_mean

from ultra.config import CONFIG_ULTRA
from ultra.util import calculate_sensitivity_matrices, generate_tolerance_table, copy_ultra_ini, sort_1d_mus_per_seg, sort_1d_mus_per_actuator
from ultra.close_loop_analysis import req_closedloop_calc_batch
from ultra.plotting import plot_iter_wf, plot_iter_mv

if __name__ == '__main__':

    # Set number of rings
    NUM_RINGS = 1

    # Define the type of WFE.
    WHICH_DM = 'harris_seg_mirror'

    # Define target contrast
    C_TARGET = CONFIG_ULTRA.getfloat('target', 'contrast')

    # Parameters for Temporal Analysis
    sptype = CONFIG_ULTRA.get('target', 'sptype')
    Vmag = CONFIG_ULTRA.getfloat('target', 'Vmag')

    minlam = CONFIG_ULTRA.getfloat('target', 'minlam') * u.nanometer
    maxlam = CONFIG_ULTRA.getfloat('target', 'maxlam') * u.nanometer

    dark_current = CONFIG_ULTRA.getfloat('detector', 'dark_current')
    CIC = CONFIG_ULTRA.getfloat('detector', 'dark_current')

    # Set close loop parameters.
    detector_noise = CONFIG_ULTRA.getfloat('detector', 'detector_noise')
    niter = CONFIG_ULTRA.getint('close_loop', 'niter')
    TimeMinus = CONFIG_ULTRA.getfloat('close_loop', 'TimeMinus')
    TimePlus = CONFIG_ULTRA.getfloat('close_loop', 'TimePlus')
    Ntimes = CONFIG_ULTRA.getint('close_loop', 'Ntimes')

    wavescale_min = CONFIG_ULTRA.getint('close_loop', 'wavescale_min')
    wavescale_max = CONFIG_ULTRA.getint('close_loop', 'wavescale_max')
    wavescale_step = CONFIG_ULTRA.getint('close_loop', 'wavescale_step')
    fractional_scale = CONFIG_ULTRA.getfloat('close_loop', 'fractional_scale')

    mv_min = CONFIG_ULTRA.getint('close_loop', 'mv_min')
    mv_max = CONFIG_ULTRA.getint('close_loop', 'mv_max')
    mv_step = CONFIG_ULTRA.getint('close_loop', 'mv_step')

    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)
        NUM_MODES = 5

    # Run PASTIS analysis.
    run_matrix = MatrixEfieldHex(which_dm=WHICH_DM, dm_spec=DM_SPEC, num_rings=NUM_RINGS,
                                 calc_science=True, calc_wfs=True,
                                 initial_path=CONFIG_PASTIS.get('local', 'local_data_path'), norm_one_photon=True)

    run_matrix.calc()
    data_dir = run_matrix.overall_dir
    copy_ultra_ini(data_dir)
    print(f'All saved to {data_dir}.')

    # Call the simulator.
    tel = run_matrix.simulator

    # Calculate raw coronagraphic contrast floor.
    dh_mask_shaped = tel.dh_mask.shaped
    unaber_psf = fits.getdata(os.path.join(data_dir, 'unaberrated_coro_psf.fits'))  # already normalized to max(direct psf)
    contrast_floor = dh_mean(unaber_psf, dh_mask_shaped)

    # Calculate static tolerances.
    pastis_matrix = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'pastis_matrix.fits'))
    mus = calculate_segment_constraints(pastis_matrix, c_target=C_TARGET, coronagraph_floor=0)
    np.savetxt(os.path.join(data_dir, 'mus_Hex_%d_%s.csv' % (NUM_RINGS, C_TARGET)), mus, delimiter=',')

    # Get the efields at wfs and science plane.
    efield_science_real = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_coron_real.fits'))
    efield_science_imag = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_coron_imag.fits'))
    efield_wfs_real = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_obwfs_real.fits'))
    efield_wfs_imag = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_obwfs_imag.fits'))
    ref_coron = fits.getdata(os.path.join(data_dir, 'ref_e0_coron.fits'))
    ref_obwfs = fits.getdata(os.path.join(data_dir, 'ref_e0_wfs.fits'))

    print('Computing Sensitivity Matrices..')
    # Compute sensitivity matrices.
    sensitivity_matrices = calculate_sensitivity_matrices(ref_coron, ref_obwfs, efield_science_real,
                                                          efield_science_imag,
                                                          efield_wfs_real, efield_wfs_imag, subsample_factor=8)
    g_coron = sensitivity_matrices['senitivity_image_plane']
    g_wfs = sensitivity_matrices['sensitvity_wfs_plane']
    e0_coron = sensitivity_matrices['ref_image_plane']
    e0_wfs = sensitivity_matrices['ref_wfs_plane']

    # Compute Temporal tolerances.
    print('Computing close loop contrast estimation..')

    # Compute Star flux.
    npup = int(np.sqrt(tel.pupil_grid.x.shape[0]))
    star_flux = exoscene.star.bpgs_spectype_to_photonrate(spectype=sptype, Vmag=Vmag,
                                                          minlam=minlam.value, maxlam=maxlam.value)
    flux = star_flux.value * tel.diam ** 2 * np.sum(tel.apodizer ** 2) / npup ** 2

    Qharris = np.diag(np.asarray(mus ** 2))

    unaberrated_coro_psf, ref = tel.calc_psf(ref=True, display_intermediate=False, norm_one_photon=True)
    norm = np.max(ref)

    # Calculate contrasts for different wavefront-sensor exposure times, iterating over different values of Q.

    # Note: 'fractional_scale' parameter is introduced to adjust wfe-drift-scaling to float,  if required.
    # Python range() takes only integer values.
    contrasts_delta_wf = []
    for wavescale in range(wavescale_min, wavescale_max, wavescale_step):
        print('recurssive close loop batch estimation and wavescale %f' % wavescale)
        timer1 = time.time()
        StarMag = 0.0
        for tscale in np.logspace(TimeMinus, TimePlus, Ntimes):
            Starfactor = 10 ** (-StarMag / 2.5)
            print(tscale)
            tmp0 = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                             detector_noise, tscale, flux * Starfactor,
                                             fractional_scale * wavescale ** 2 * Qharris,
                                             niter, tel.dh_mask, norm)
            tmp1 = tmp0['averaged_hist']
            n_tmp1 = len(tmp1)
            contrasts_delta_wf.append(tmp1[n_tmp1 - 1])

    np.savetxt(os.path.join(data_dir, 'contrast_wf.csv'), contrasts_delta_wf, delimiter=',')

    # Get optimal contrast, wfs time and drift wavefront.
    opt_delta_contrast, opt_tscale, opt_wavescale = plot_iter_wf(Qharris, contrasts_delta_wf, contrast_floor, data_dir)

    print('Optimal Wavescale found:', opt_wavescale, 'Optimal wfs time scale:', opt_tscale)

    # Calculate contrast vs wavefront sensing time for different values of stellar magnitude.
    contrasts_mv = []
    for mv in range(mv_min, mv_max, mv_step):
        stellar_flux = exoscene.star.bpgs_spectype_to_photonrate(spectype=sptype, Vmag=mv, minlam=minlam.value, maxlam=maxlam.value)
        entrance_flux = stellar_flux.value * tel.diam ** 2 * np.sum(tel.apodizer ** 2) / npup ** 2
        for tscale in np.logspace(TimeMinus, TimePlus, Ntimes):
            Starfactor = 10 ** (-StarMag / 2.5)
            print(tscale)
            tmp0 = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                             detector_noise, tscale, entrance_flux * Starfactor,
                                             fractional_scale * opt_wavescale ** 2 * Qharris,
                                             niter, tel.dh_mask, norm)
            tmp1 = tmp0['averaged_hist']
            n_tmp1 = len(tmp1)
            contrasts_mv.append(tmp1[n_tmp1 - 1])

    np.savetxt(os.path.join(data_dir, 'contrast_mv.csv'), contrasts_mv, delimiter=',')

    plot_iter_mv(contrasts_mv, contrast_floor, data_dir)

    # Final Individual Tolerance allocation across 5 modes in units of pm.
    coeffs_table = sort_1d_mus_per_seg(mus, NUM_MODES, tel.nseg)   # sorts mus into NUM_MODES groups of tel.nseg

    # Sort mus into NUM_MODES groups of tel.nseg * NUM_MODES elements, in nm.
    coeffs_numaps = sort_1d_mus_per_actuator(mus, NUM_MODES, tel.nseg)

    # Calculate total delta-contrast at optimal wfs time, optimal wavefront error scale.
    c_total = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise, detector_noise,
                                        opt_tscale, flux * Starfactor, fractional_scale * opt_wavescale ** 2 * Qharris, niter,
                                        tel.dh_mask, norm)
    resultant_c_total = []
    c0 = c_total['averaged_hist']
    n_tmp1 = len(c0)
    resultant_c_total.append(c0[n_tmp1 - 1])
    c0 = resultant_c_total[0] - contrast_floor

    Qharris_individual = []
    for mode in range(NUM_MODES):
        Qharris_per_mode = np.diag(np.asarray(coeffs_numaps[mode] ** 2))
        Qharris_individual.append(Qharris_per_mode)
    Qharris_individual = np.array(Qharris_individual)

    c_per_modes = []
    for mode in range(NUM_MODES):
        contrast = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                             detector_noise, opt_tscale, flux * Starfactor,
                                             fractional_scale * opt_wavescale ** 2 * Qharris_individual[mode],
                                             niter, tel.dh_mask, norm)
        resultant_contrast = []
        c1 = contrast['averaged_hist']
        n_tmp1 = len(c1)
        resultant_contrast.append(c1[n_tmp1 - 1])
        c_per_modes.append(resultant_contrast[0] - contrast_floor)

    c_per_modes = np.array(c_per_modes)

    # Calculate total mean temporal wavefront error for individual modes.
    Q_total = 1e3 * np.sqrt(np.mean(np.diag(fractional_scale * opt_wavescale ** 2 * Qharris)))  # in pm
    Q_individual = []
    for mode in range(NUM_MODES):
        Q_modes = 1e3 * np.sqrt(np.mean(fractional_scale * opt_wavescale ** 2 * (coeffs_table[mode] ** 2)))  # in pm
        Q_individual.append(Q_modes)
    Q_individual = np.array(Q_individual)  # set of floats only, RMS of Q_individuals should be equal to Q_total.

    print('Generating tolerance tables:..')
    tables = generate_tolerance_table(tel, Q_individual, Q_total,
                                      c_per_modes, c0, contrast_floor, opt_wavescale, opt_tscale, data_dir)

    print(tables[0], '\n', tables[1])
    print(f'All analysis is saved to {data_dir}.')
