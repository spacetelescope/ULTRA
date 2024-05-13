from astropy.io import fits
import astropy.units as u
import exoscene.star
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from pastis.config import CONFIG_PASTIS
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldLuvoirA
from pastis.pastis_analysis import calculate_segment_constraints
from pastis.util import dh_mean

from ultra.config import CONFIG_ULTRA
from ultra.util import calculate_sensitivity_matrices
from ultra.util import matrix_subsample
from ultra.close_loop_analysis import req_closedloop_calc_batch
from ultra.close_loop_analysis import req_closedloop_calc_recursive
from ultra.plotting import plot_iter_wf_log

if __name__ == '__main__':

    # Set number of rings
    NUM_RINGS = 1

    # Define the type of WFE.
    WHICH_DM = 'zernike_mirror'

    # Define target contrast
    C_TARGET = 1e-10

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
    Nwavescale = CONFIG_ULTRA.getfloat('close_loop', 'Nwavescale')

    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)
        NUM_MODES = 7

    DM_SPEC = 15
    NUM_MODES = 15
    run_matrix = MatrixEfieldLuvoirA(which_dm=WHICH_DM, dm_spec=DM_SPEC,
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
    mus = calculate_segment_constraints(pastis_matrix[1:15,1:15], c_target=C_TARGET, coronagraph_floor=0)
    np.savetxt(os.path.join(data_dir, 'mus_Hex_%d_%s.csv' % (NUM_RINGS, C_TARGET)), mus, delimiter=',')

    # Get the efields at wfs and science plane.
    efield_science_real = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_coron_real.fits'))[1:NUM_MODES]
    efield_science_imag = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_coron_imag.fits'))[1:NUM_MODES]
    efield_wfs_real = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_obwfs_real.fits'))[1:NUM_MODES]
    efield_wfs_imag = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'efield_obwfs_imag.fits'))[1:NUM_MODES]
    ref_coron = fits.getdata(os.path.join(data_dir, 'ref_e0_coron.fits'))
    ref_obwfs = fits.getdata(os.path.join(data_dir, 'ref_e0_wfs.fits'))

    print('Computing Sensitivity Matrices..')
    # Compute sensitivity matrices.
    sensitivity_matrices = calculate_sensitivity_matrices(ref_coron, ref_obwfs, efield_science_real,
                                                          efield_science_imag,
                                                          efield_wfs_real, efield_wfs_imag, subsample_factor=8)
    g_coron = sensitivity_matrices['sensitivity_image_plane']
    g_wfs = sensitivity_matrices['sensitivity_wfs_plane']
    e0_coron = sensitivity_matrices['ref_image_plane']
    e0_wfs = sensitivity_matrices['ref_wfs_plane']

    # Compute Temporal tolerances.
    print('Computing close loop contrast estimation..')

    for Vmag in range(0, 11, 2):
        print(Vmag)

        # Compute Star flux.
        npup = int(np.sqrt(tel.pupil_grid.x.shape[0]))
        star_flux = exoscene.star.bpgs_spectype_to_photonrate(spectype=sptype, Vmag=Vmag,
                                                              minlam=minlam.value, maxlam=maxlam.value)
        Nph = star_flux.value *  tel.diam ** 2 * np.sum(tel.apodizer ** 2) / npup ** 2
        flux = Nph

        Qharris = np.diag(np.asarray(mus**2))

        unaberrated_coro_psf, ref = tel.calc_psf(ref=True, display_intermediate=False, norm_one_photon=True)
        norm = np.max(ref)

        # wavescale_min = 2    # TODO: plot works only for 7 wavescale values, chose the stepsize accordingly.
        # wavescale_max = 200
        # wavescale_step = 40
        Nwavescale = 5
        WaveScaleMinus = -1
        WaveScalePlus = 0
        wavescaleVec = np.logspace(WaveScaleMinus, WaveScalePlus, Nwavescale)
        result_wf_test = []
        # for wavescale in range(wavescale_min, wavescale_max, wavescale_step):
        for wavescale in wavescaleVec:
            print('Recursive close loop batch estimation and wavescale %f' % wavescale)
            niter = 10
            timer1 = time.time()
            StarMag = 0.0
            for tscale in np.logspace(TimeMinus, TimePlus, Ntimes):
                Starfactor = 10 ** (-StarMag / 2.5)
                print(tscale)
                #with Lowfs
                tmp0 = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                                 detector_noise, tscale, flux * Starfactor,
                                                 wavescale ** 2 * Qharris,
                                                 niter, tel.dh_mask, norm)
                # # #With DH
                # tmp0 = req_closedloop_calc_batch(g_coron, g_coron, e0_coron, e0_coron, detector_noise,
                #                                  detector_noise, tscale, flux * Starfactor,
                #                                  wavescale ** 2 * Qharris,
                #                                  niter, tel.dh_mask, norm)


                tmp1 = tmp0['averaged_hist']
                n_tmp1 = len(tmp1)
                result_wf_test.append(tmp1[n_tmp1 - 1])

                # For corono
                dh_mask_per_act = np.transpose(np.tile(tel.dh_mask, (g_coron.shape[2], 1)))
                real_coron = dh_mask_per_act * g_coron[:, 0, :]
                imag_coron = dh_mask_per_act * g_coron[:, 1, :]
                g_coron_flat = np.concatenate((real_coron,imag_coron),axis = 0)
                #For wfs
                subsample_factor = 8
                Npup = int(np.sqrt(tel.aperture.shape[0]))
                aperture_square_array = np.reshape(tel.aperture, [Npup, Npup])
                n_sub_pix = int(Npup/ subsample_factor)
                aperture_square_array_small =  matrix_subsample(aperture_square_array, n_sub_pix, n_sub_pix)/subsample_factor**2
                aperture_small = np.reshape(aperture_square_array_small, n_sub_pix ** 2)
                aperture_mask = aperture_small == 1

                aperture_mask_per_act = np.transpose(np.tile(aperture_mask, (g_wfs.shape[2], 1)))
                real_wfs = aperture_mask_per_act * g_wfs[:, 0, :]
                imag_wfs = aperture_mask_per_act * g_wfs[:, 1, :]
                g_wfs_flat = np.concatenate((real_wfs,imag_wfs),axis = 0)

                #mat_coron = np.dot(np.transpose(g_coron_flat), g_coron_flat)/np.sum(tel.dh_mask)
                mat_coron = np.dot(np.transpose(g_coron_flat), g_coron_flat)
                #mat_wfs  = np.dot(np.transpose(g_wfs_flat), g_wfs_flat)/np.sum(aperture_mask)
                mat_wfs  = np.dot(np.transpose(g_wfs_flat), g_wfs_flat)
                (eval, evec) = np.linalg.eig(mat_coron)
                eigen_coron = np.diagonal(mat_coron)
                (eval, evec) = np.linalg.eig(mat_wfs)
                eigen_wfs = np.diagonal(mat_wfs)
                numerator = np.sum(eigen_coron / eigen_wfs)
                denominator_sum = np.sum(wavescale**2*np.diagonal(Qharris)*eigen_coron)
                t_min = 1 / np.sqrt(2 * flux) * np.sqrt(numerator / denominator_sum)
                delta_C_batch = 2 * np.sqrt(2 / flux) * np.sqrt(numerator * denominator_sum)*norm
                delta_C_rec =  np.sqrt(2 / flux) * np.sum(wavescale*np.sqrt(np.diagonal(Qharris))*eigen_coron/np.sqrt(eigen_wfs))*norm

                print(t_min)
                print(delta_C_batch)
                print(delta_C_rec)


        np.savetxt(os.path.join(data_dir, 'contrast_wf_%s_%d_%d_%d_%d.csv' % (C_TARGET, WaveScaleMinus, WaveScalePlus, Nwavescale,Vmag)),
                   result_wf_test, delimiter=',')

        plot_iter_wf_log(Qharris, WaveScaleMinus, WaveScalePlus, Nwavescale,
                     TimeMinus, TimePlus, Ntimes, result_wf_test, contrast_floor, C_TARGET, Vmag, data_dir)