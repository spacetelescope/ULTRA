from astropy.io import fits
import astropy.units as u
import exoscene.star
import numpy as np
import os
import pandas as pd

from pastis.config import CONFIG_PASTIS
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldHex
from pastis.pastis_analysis import calculate_segment_constraints
from pastis.util import dh_mean

from ultra.config import CONFIG_ULTRA
from ultra.util import calculate_sensitivity_matrices
from ultra.close_loop_analysis import req_closedloop_calc_batch, req_closedloop_calc_recursive
from ultra.plotting import plot_iter_wf


if __name__ == '__main__':

    # Set number of rings
    NUM_RINGS = 1

    # Define which WFS to calculate, "obwfs" or "lowfs".
    # The science plane is always calculated.
    WFS = 'lowfs'

    # Plane to run algorithm on, "coronagraph" or "wfs"
    PLANE = "coronagraph"

    # Algorithm to use, "batch" or "recursive"
    ALGORITHM = "batch"
    algo_function = req_closedloop_calc_batch if ALGORITHM == "batch" else req_closedloop_calc_recursive
    niter = 10 if ALGORITHM == "batch" else 100

    # Define target contrast
    C_TARGET = 1e-11

    # Define the type of WFE.
    WHICH_DM = 'seg_mirror'

    # DM_SPEC = tuple or int, specification for the used DM -
    # for seg_mirror: int, number of local Zernike modes on each segment
    # for harris_seg_mirror: tuple (string, array, bool, bool, bool),
    # absolute path to Harris spreadsheet, pad orientations, choice of Harris mode sets (thermal, mechanical, other)
    # for zernike_mirror: int, number of global Zernikes
    DM_SPEC = 7
    NUM_MODES = 7

    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)

    # Parameters for temporal analysis
    sptype = CONFIG_ULTRA.get('target', 'sptype')
    Vmag = CONFIG_ULTRA.getfloat('target', 'Vmag')

    minlam = CONFIG_ULTRA.getfloat('target', 'minlam') * u.nanometer
    maxlam = CONFIG_ULTRA.getfloat('target', 'maxlam') * u.nanometer

    # Set close loop parameters.
    detector_noise = CONFIG_ULTRA.getfloat('detector', 'detector_noise')
    TimeMinus = CONFIG_ULTRA.getfloat('close_loop', 'TimeMinus')
    TimePlus = CONFIG_ULTRA.getfloat('close_loop', 'TimePlus')
    Ntimes = CONFIG_ULTRA.getint('close_loop', 'Ntimes')
    Nwavescale = CONFIG_ULTRA.getfloat('close_loop', 'Nwavescale')

    # Calculate the E-fields in the science and WFS planes.
    calc_wfs = False if WFS == 'lowfs' else True
    calc_lowfs = True if WFS == 'lowfs' else False

    run_matrix = MatrixEfieldHex(which_dm=WHICH_DM, dm_spec=DM_SPEC, num_rings=NUM_RINGS,
                                 calc_science=True, calc_wfs=calc_wfs, calc_lowfs=calc_lowfs,
                                 initial_path=CONFIG_PASTIS.get('local', 'local_data_path'), norm_one_photon=True)

    run_matrix.calc()
    data_dir = run_matrix.overall_dir
    print(f'All saved to {data_dir}.')

    # Retrieve the telescope simulator object
    tel = run_matrix.simulator

    # Calculate the unaberrated coronagraphic PSF for normalization, and contrast floor.
    unaberrated_coro_psf, ref = tel.calc_psf(ref=True, display_intermediate=False, norm_one_photon=True)
    norm = np.max(ref)
    contrast_floor = dh_mean(unaberrated_coro_psf.shaped / norm, tel.dh_mask.shaped)

    # Calculate static tolerances.
    pastis_matrix = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'pastis_matrix.fits'))
    mus = calculate_segment_constraints(pastis_matrix, c_target=C_TARGET, coronagraph_floor=0)
    np.savetxt(os.path.join(data_dir, 'mus_Hex_%d_%s.csv' % (NUM_RINGS, C_TARGET)), mus, delimiter=',')
    Qharris = np.diag(np.asarray(mus ** 2))

    # Get the efields at wfs and science plane.
    efield_science_real = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_coron_real.fits'))
    efield_science_imag = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_coron_imag.fits'))
    efield_wfs_real = fits.getdata(os.path.join(data_dir, 'matrix_numerical', f'efield_{WFS}_real.fits'))
    efield_wfs_imag = fits.getdata(os.path.join(data_dir, 'matrix_numerical', f'efield_{WFS}_imag.fits'))
    ref_coron = fits.getdata(os.path.join(data_dir, 'ref_e0_coron.fits'))
    ref_wfs = fits.getdata(os.path.join(data_dir, f'ref_e0_{WFS}.fits'))

    # Compute sensitivity matrices.
    print('Computing Sensitivity Matrices..')
    sensitivity_matrices = calculate_sensitivity_matrices(ref_coron, ref_wfs, efield_science_real, efield_science_imag,
                                                          efield_wfs_real, efield_wfs_imag, subsample_factor=8)
    g_coron = sensitivity_matrices['sensitivity_image_plane']
    g_wfs = sensitivity_matrices['sensitivity_wfs_plane']
    e0_coron = sensitivity_matrices['ref_image_plane']
    e0_wfs = sensitivity_matrices['ref_wfs_plane']

    # Compute Temporal tolerances.
    print('Computing closed-loop contrast estimation..')

    # Compute Star flux.
    Npup = int(np.sqrt(tel.aperture.shape[0]))
    star_flux = exoscene.star.bpgs_spectype_to_photonrate(spectype=sptype, Vmag=Vmag,
                                                          minlam=minlam.value, maxlam=maxlam.value)
    flux = star_flux.value * 15 ** 2 * np.sum(tel.apodizer ** 2) / Npup ** 2

    # Detector plane to iterate over
    e0_iter = e0_coron if PLANE == "coronagraph" else e0_wfs
    g_iter = g_coron if PLANE == "coronagraph" else g_wfs

    wavescale_min = 10    # TODO: plot works only for 7 wavescale values, chose the stepsize accordingly.
    wavescale_max = 260
    wavescale_step = 50
    result_wf_test = []
    for wavescale in range(wavescale_min, wavescale_max, wavescale_step):
        print(f'Closed-loop estimation using {ALGORITHM} algorithm and wavescale {wavescale}')
        StarMag = 0.0
        for tscale in np.logspace(TimeMinus, TimePlus, Ntimes):
            Starfactor = 10 ** (-StarMag / 2.5)
            print(f"tscale: {tscale}")
            tmp0 = algo_function(g_coron, g_iter, e0_coron, e0_iter, detector_noise, detector_noise, tscale,
                                 flux * Starfactor, 0.0001 * wavescale ** 2 * Qharris, niter, tel.dh_mask, norm)
            tmp1 = tmp0['averaged_hist']
            n_tmp1 = len(tmp1)
            result_wf_test.append(tmp1[n_tmp1 - 1])

    np.savetxt(os.path.join(data_dir, 'contrast_wf_%s_%d_%d_%d.csv' % (C_TARGET, wavescale_min, wavescale_max, wavescale_step)),
               result_wf_test, delimiter=',')
    plot_iter_wf(Qharris, wavescale_min, wavescale_max, wavescale_step, TimeMinus, TimePlus, Ntimes, result_wf_test,
                 contrast_floor, C_TARGET, Vmag, data_dir)

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
            coeffs_tmp[qq + kk * NUM_MODES] = mus[qq + kk * NUM_MODES]  # arranged per modal basis
        coeffs_numaps[qq] = coeffs_tmp  # arranged into 5 groups of 600 elements and in units of nm

    Qharris_individual = []
    for mode in range(NUM_MODES):
        Qharris_per_mode = np.diag(np.asarray(coeffs_numaps[mode] ** 2))
        Qharris_individual.append(Qharris_per_mode)

    Qmode = np.array(Qharris_individual)

    c_total = algo_function(g_coron, g_iter, e0_coron, e0_iter, detector_noise, detector_noise, opt_tscale,
                            flux * Starfactor, 0.0001 * opt_wavescale ** 2 * Qharris, niter, tel.dh_mask, norm)

    resultant_c_total = []
    c0 = c_total['averaged_hist']
    n_tmp1 = len(c0)
    resultant_c_total.append(c0[n_tmp1 - 1])
    c0 = resultant_c_total[0] - contrast_floor

    c_per_modes = []
    for mode in range(NUM_MODES):
        contrast = algo_function(g_coron, g_iter, e0_coron, e0_iter, detector_noise, detector_noise, opt_tscale,
                                 flux * Starfactor, 0.0001 * opt_wavescale ** 2 * Qmode[mode], niter, tel.dh_mask, norm)

        resultant_contrast = []
        c1 = contrast['averaged_hist']
        n_tmp1 = len(c1)
        resultant_contrast.append(c1[n_tmp1 - 1])
        c_per_modes.append(resultant_contrast[0] - contrast_floor)

    contrast_per_mode = np.array(c_per_modes)

    df = pd.DataFrame()
    df['Segment zernike Modes'] = ['Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'All']
    df['Tolerances in pm'] = [Q_individuals[0], Q_individuals[1], Q_individuals[2],
                              Q_individuals[3], Q_individuals[4], Q_individuals[5], Q_individuals[6], Q_total]
    df['Contrast'] = [contrast_per_mode[0], contrast_per_mode[1], contrast_per_mode[2], contrast_per_mode[3],
                      contrast_per_mode[4], contrast_per_mode[5], contrast_per_mode[6], c0]
    df[''] = None
    df['Telescope'] = ['total segs', 'diam', 'seg diam', 'contrast_floor', 'iwa', 'owa']
    df['Values'] = [tel.nseg, tel.diam, tel.diam / (2 * NUM_RINGS + 1), contrast_floor, tel.iwa, tel.owa]
    df['opt_wv'] = [opt_wavescale, '', '', '', '', '']
    df['opt_t'] = [opt_tscale, '', '', '', '', '']
    print(df)
    df.to_csv(os.path.join(data_dir, 'tolerance_table.csv'))
    print(f'All analysis is saved to {data_dir}.')
