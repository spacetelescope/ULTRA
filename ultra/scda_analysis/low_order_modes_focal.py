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
from ultra.util import calculate_sensitivity_matrices
from ultra.close_loop_analysis import req_closedloop_calc_batch
from ultra.plotting import plot_iter_wf


if __name__ == '__main__':

    # Set number of rings
    NUM_RINGS = 1

    # Define target contrast
    C_TARGET = 1e-11

    # Define the type of WFE.
    WHICH_DM = 'zernike_mirror'

    # DM_SPEC = tuple or int, specification for the used DM -
    # for seg_mirror: int, number of local Zernike modes on each segment
    # for harris_seg_mirror: tuple (string, array, bool, bool, bool),
    # absolute path to Harris spreadsheet, pad orientations, choice of Harris mode sets (thermal, mechanical, other)
    # for zernike_mirror: int, number of global Zernikes
    DM_SPEC = 15
    NUM_MODES = 15

    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)
        NUM_MODES = 7

    # Parameters for temporal analysis
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
    Nwavescale = CONFIG_ULTRA.getfloat('close_loop', 'Nwavescale')

    run_matrix = MatrixEfieldHex(which_dm=WHICH_DM, dm_spec=DM_SPEC, num_rings=NUM_RINGS,
                                 calc_science=True, calc_wfs=False, calc_lowfs=True,
                                 initial_path=CONFIG_PASTIS.get('local', 'local_data_path'), norm_one_photon=True)

    run_matrix.calc()
    data_dir = run_matrix.overall_dir
    print(f'All saved to {data_dir}.')

    # Retrieve the telescope simulator object
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
    efield_wfs_real = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_lowfs_real.fits'))
    efield_wfs_imag = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_lowfs_imag.fits'))
    ref_coron = fits.getdata(os.path.join(data_dir, 'ref_e0_coron.fits'))
    ref_lowfs = fits.getdata(os.path.join(data_dir, 'ref_e0_wfs.fits'))

    print('Computing Sensitivity Matrices..')
    # Compute sensitivity matrices.
    sensitivity_matrices = calculate_sensitivity_matrices(ref_coron, ref_lowfs, efield_science_real,
                                                          efield_science_imag,
                                                          efield_wfs_real, efield_wfs_imag, subsample_factor=8)
    g_coron = sensitivity_matrices['sensitivity_image_plane']
    g_wfs = sensitivity_matrices['sensitivity_wfs_plane']
    e0_coron = sensitivity_matrices['ref_image_plane']
    e0_wfs = sensitivity_matrices['ref_wfs_plane']

    # Compute temporal tolerances.
    print('Computing close loop contrast estimation..')

    # Compute stellar flux.
    npup = int(np.sqrt(tel.pupil_grid.x.shape[0]))
    star_flux = exoscene.star.bpgs_spectype_to_photonrate(spectype=sptype, Vmag=Vmag,
                                                          minlam=minlam.value, maxlam=maxlam.value)
    Nph = star_flux.value * 15 ** 2 * np.sum(tel.apodizer ** 2) / npup ** 2
    flux = Nph

    Qharris = np.diag(np.asarray(mus**2))

    unaberrated_coro_psf, ref = tel.calc_psf(ref=True, display_intermediate=False, norm_one_photon=True)
    norm = np.max(ref)

    wavescale_min = 20    # TODO: plot works only for 7 wavescale values, chose the stepsize accordingly.
    wavescale_max = 200
    wavescale_step = 10
    result_wf_test = []
    for wavescale in range(wavescale_min, wavescale_max, wavescale_step):
        print('Recursive, closed-loop, batch estimation and wavescale %f' % wavescale)
        niter = 10
        timer1 = time.time()
        StarMag = 0.0
        for tscale in np.logspace(TimeMinus, TimePlus, Ntimes):
            Starfactor = 10 ** (-StarMag / 2.5)
            print(tscale)
            # tmp0 = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
            #                                  detector_noise, tscale, flux * Starfactor,
            #                                  0.0001 * wavescale ** 2 * Qharris,
            #                                  niter, tel.dh_mask, norm)

            tmp0 = req_closedloop_calc_batch(g_coron, g_coron, e0_coron, e0_coron, detector_noise,
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
