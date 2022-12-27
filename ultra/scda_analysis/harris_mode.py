from astropy.io import fits
import astropy.units as u
import exoscene.star
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from pastis.config import CONFIG_PASTIS
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldHex
from pastis.pastis_analysis import calculate_segment_constraints

from ultra.config import CONFIG_ULTRA
from ultra.util import calculate_sensitivity_matrices
from ultra.close_loop_analysis import req_closedloop_calc_batch

if __name__ == '__main__':

    # Define the type of WFE.
    WHICH_DM = 'harris_seg_mirror'
    APLC_DESIGN = 'small'

    # Define target contrast
    C_TARGET = 1e-10

    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)
        NUM_MODES = 5 # TODO: works only for thermal modes currently

    NUM_RINGS = 1
    run_matrix = MatrixEfieldHex(which_dm=WHICH_DM, dm_spec=DM_SPEC, num_rings=NUM_RINGS,
                                 calc_science=True, calc_wfs=True,
                                 initial_path=CONFIG_PASTIS.get('local', 'local_data_path'), norm_one_photon=True)

    run_matrix.calc()
    data_dir = run_matrix.overall_dir
    print(f'All saved to {data_dir}.')

    # Calculate static tolerances.
    pastis_matrix = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'pastis_matrix.fits'))
    mus = calculate_segment_constraints(pastis_matrix, c_target=C_TARGET, coronagraph_floor=0)
    np.savetxt(os.path.join(data_dir, 'mus_harris_%s.csv' % C_TARGET), mus, delimiter=',')

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

    # Compute Temporal tolerances.
    tel = run_matrix.simulator
    npup = int(np.sqrt(tel.pupil_grid.x.shape[0]))

    sptype = CONFIG_ULTRA.get('target', 'sptype')
    Vmag = CONFIG_ULTRA.getint('target', 'vmag')

    minlam = CONFIG_ULTRA.get('target', 'minlam') * u.nanometer
    maxlam = CONFIG_ULTRA.get('target', 'maxlam') * u.nanometer

    dark_current = CONFIG_ULTRA.getfloat('detector', 'dark_current')
    CIC = CONFIG_ULTRA.getfloat('detector', 'dark_current')

    star_flux = exoscene.star.bpgs_spectype_to_photonrate(spectype=sptype, Vmag=Vmag, minlam=minlam.value,
                                                          maxlam=maxlam.value)
    Nph = star_flux.value * 15 ** 2 * np.sum(tel.apodizer ** 2) / npup ** 2
    flux = Nph

    Qharris = np.diag(np.asarray(mus ** 2))

    detector_noise = 0.0
    niter = 10
    TimeMinus = -2
    TimePlus = 5.5
    Ntimes = 20
    Nwavescale = 8
    Nflux = 3
    res = np.zeros([Ntimes, Nwavescale, Nflux, 1])
    result_wf_test = []

    unaberrated_coro_psf, ref = tel.calc_psf(ref=True, display_intermediate=False, norm_one_photon=True)
    norm = np.max(ref)

    print(norm)


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

    delta_wf = []
    for wavescale in range(1, 15, 2):
        wf = np.sqrt(np.mean(np.diag(0.0001 * wavescale ** 2 * Qharris))) * 1e3
        delta_wf.append(wf)

    texp = np.logspace(TimeMinus, TimePlus, Ntimes)
    font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 20}
    contrast_floor = 4.237636070056418e-11
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
    plt.ylabel("$\Delta$ contrast", fontsize=20)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='upper center', fontsize=20)
    plt.tick_params(top=True, bottom=True, left=True,
                    right=True, labelleft=True, labelbottom=True,
                    labelsize=20)
    plt.tick_params(axis='both', which='major', length=10, width=2)
    plt.tick_params(axis='both', which='minor', length=6, width=2)
    plt.grid()
    plt.savefig(os.path.join(data_dir, 'cont_wf_%s.png' % C_TARGET))