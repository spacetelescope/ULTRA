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
from ultra.util import calculate_sensitivity_matrices, generate_tolerance_table
from ultra.close_loop_analysis import req_closedloop_calc_batch
from ultra.plotting import plot_iter_wf, plot_iter_mv

if __name__ == '__main__':

    # Set number of rings
    NUM_RINGS = 1

    # Define the type of WFE.
    WHICH_DM = 'harris_seg_mirror'

    # Define target contrast
    C_TARGET = CONFIG_ULTRA.getfloat('target', 'contrast')

    # Parameters for Temporal Ananlysis
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

    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)
        NUM_MODES = 5

    run_matrix = MatrixEfieldHex(which_dm=WHICH_DM, dm_spec=DM_SPEC, num_rings=NUM_RINGS,
                                 calc_science=True, calc_wfs=True,
                                 initial_path=CONFIG_PASTIS.get('local', 'local_data_path'), norm_one_photon=True)

    run_matrix.calc()
    data_dir = run_matrix.overall_dir
    print(f'All saved to {data_dir}.')

    tel = run_matrix.simulator

    unaber_psf = fits.getdata(os.path.join(data_dir, 'unaberrated_coro_psf.fits'))  # already normalized to max of direct psf
    dh_mask_shaped = tel.dh_mask.shaped
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

    # Calculate contrast vs wavefront sensing time for different values of Q.
    wavescale_min = 100
    wavescale_max = 240
    wavescale_step = 10
    result_wf_test = []
    for wavescale in range(wavescale_min, wavescale_max, wavescale_step):
        print('recurssive close loop batch estimation and wavescale %f' % wavescale)
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

    np.savetxt(os.path.join(data_dir, 'contrast_wf_%s_%d_%d_%d.csv' % (C_TARGET, wavescale_min, wavescale_max, wavescale_step)),
               result_wf_test, delimiter=',')
    plot_iter_wf(Qharris, wavescale_min, wavescale_max, wavescale_step,
                 TimeMinus, TimePlus, Ntimes, result_wf_test, contrast_floor, C_TARGET, Vmag, data_dir)

    # Calculate contrast vs wavefront sensing time for different values of stellar magnitude.
    contrasts_mv = []
    mv_min = 1
    mv_max = 10
    mv_step = 2
    for mv in range(mv_min, mv_max, mv_step):
        stellar_flux = exoscene.star.bpgs_spectype_to_photonrate(spectype=sptype, Vmag=mv, minlam=minlam.value, maxlam=maxlam.value)
        entrace_flux = stellar_flux.value * tel.diam ** 2 * np.sum(tel.apodizer ** 2) / npup ** 2
        for tscale in np.logspace(TimeMinus, TimePlus, Ntimes):
            Starfactor = 10 ** (-StarMag / 2.5)
            print(tscale)
            tmp0 = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                             detector_noise, tscale, entrace_flux * Starfactor,
                                             0.0001 * wavescale ** 2 * Qharris,
                                             niter, tel.dh_mask, norm)
            tmp1 = tmp0['averaged_hist']
            n_tmp1 = len(tmp1)
            contrasts_mv.append(tmp1[n_tmp1 - 1])

    plot_iter_mv(contrasts_mv, mv_min, mv_max, mv_step,
                 TimeMinus, TimePlus, Ntimes, contrast_floor, C_TARGET, Vmag, data_dir)

    # Final Individual Tolerance allocation across 5 modes in units of pm.
    coeffs_table = np.zeros([NUM_MODES, tel.nseg])  # TODO : coeffs_table = sort_1d_mus_per_seg(mus, NUM_MODES, tel.nseg)
    for qq in range(NUM_MODES):
        for kk in range(tel.nseg):
            coeffs_table[qq, kk] = mus[qq + kk * NUM_MODES]

    print('Computing tolerance table...')
    # check temporal maps for individual modes
    opt_wavescale = 200  # This is wavescale value corresponding to local minima contrast from the graph saved above.
    opt_tscale = 0.1

    Q_total = 1e3 * np.sqrt(np.mean(np.diag(0.0001 * opt_wavescale ** 2 * Qharris)))  # in pm
    Q_individual = []
    for mode in range(NUM_MODES):
        Q_modes = 1e3 * np.sqrt(np.mean(0.0001 * opt_wavescale ** 2 * (coeffs_table[mode] ** 2)))  # in pm
        Q_individual.append(Q_modes)

    Q_individuals = np.array(Q_individual)

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
        Qharris_per_mode = np.diag(np.asarray(coeffs_numaps[mode] ** 2))
        Qharris_individual.append(Qharris_per_mode)

    Qmode = np.array(Qharris_individual)

    c_total = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise, detector_noise,
                                        opt_tscale, flux * Starfactor, 0.0001 * opt_wavescale ** 2 * Qharris, niter,
                                        tel.dh_mask, norm)

    resultant_c_total = []
    c0 = c_total['averaged_hist']
    n_tmp1 = len(c0)
    resultant_c_total.append(c0[n_tmp1 - 1])
    c0 = resultant_c_total[0] - contrast_floor

    c_per_modes = []
    for mode in range(NUM_MODES):

        contrast = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                             detector_noise, opt_tscale, flux * Starfactor,
                                             0.0001 * opt_wavescale ** 2 * Qmode[mode],
                                             niter, tel.dh_mask, norm)
        resultant_contrast = []
        c1 = contrast['averaged_hist']
        n_tmp1 = len(c1)
        resultant_contrast.append(c1[n_tmp1 - 1])
        c_per_modes.append(resultant_contrast[0] - contrast_floor)

    contrast_per_mode = np.array(c_per_modes)

    tables = generate_tolerance_table(tel, Q_individuals, Q_total,
                                      contrast_per_mode, c0, contrast_floor, opt_wavescale, opt_tscale, data_dir)

    print(tables[0], '\n', tables[1])
    print(f'All analysis is saved to {data_dir}.')
