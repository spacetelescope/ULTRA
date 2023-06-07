import numpy as np
import os
import pandas as pd
from astropy.table import QTable


def matrix_subsample(matrix, n, m):
    # return a matrix of shape (n,m)
    arr_sum = []
    length = matrix.shape[0] // n  # block length
    breadth = matrix.shape[1] // m  # block breadth
    for i in range(n):
        for j in range(m):
            sum_pixels = np.sum(matrix[i * length: (i + 1) * length, j * breadth: (j + 1) * breadth])
            arr_sum.append(sum_pixels)
    data_reduced = np.reshape(np.array(arr_sum), (n, m))
    return data_reduced


def matrix_subsample_fast(matrix, n, m):
    length = matrix.shape[0] // n   # block length
    breadth = matrix.shape[1] // m  # block breadth
    new_shape = (n, length, m, breadth)
    reshaped_array = matrix.reshape(new_shape)
    data_reduced = np.sum(reshaped_array, axis=(1, 3))
    return data_reduced


def calculate_sensitivity_matrices(e0_coron, e0_obwfs, efield_coron_real, efield_coron_imag,
                                   efield_obwfs_real, efield_obwfs_imag, subsample_factor):

    total_sci_pix = np.square(e0_coron.shape[1])
    total_pupil_pix = np.square(e0_obwfs.shape[1])

    ref_coron_real = np.reshape(e0_coron[0], total_sci_pix)
    ref_coron_imag = np.reshape(e0_coron[1], total_sci_pix)

    ref_obwfs_real = np.reshape(e0_obwfs[0], total_pupil_pix)
    ref_obwfs_imag = np.reshape(e0_obwfs[1], total_pupil_pix)

    ref_coron = np.zeros([total_sci_pix, 1, 2])
    ref_coron[:, 0, 0] = ref_coron_real
    ref_coron[:, 0, 1] = ref_coron_imag

    n_sub_pix = int(np.sqrt(total_pupil_pix) // subsample_factor)
    ref_wfs_real_sub = np.reshape(matrix_subsample(e0_obwfs[0], n_sub_pix, n_sub_pix), int(np.square(n_sub_pix)))
    ref_wfs_imag_sub = np.reshape(matrix_subsample(e0_obwfs[1], n_sub_pix, n_sub_pix), int(np.square(n_sub_pix)))
    ref_wfs_sub = (ref_wfs_real_sub + 1j * ref_wfs_imag_sub) / subsample_factor
    # subsample_factor**2 is multiplied here to preserve total intensity, same goes for g_obwfs_downsampled

    ref_obwfs_downsampled = np.zeros([int(np.square(n_sub_pix)), 1, 2])
    ref_obwfs_downsampled[:, 0, 0] = ref_wfs_sub.real
    ref_obwfs_downsampled[:, 0, 1] = ref_wfs_sub.imag

    num_all_modes = efield_coron_real.shape[0]
    g_coron = np.zeros([total_sci_pix, 2, num_all_modes])
    for i in range(num_all_modes):
        g_coron[:, 0, i] = np.reshape(efield_coron_real[i], total_sci_pix) - ref_coron_real
        g_coron[:, 1, i] = np.reshape(efield_coron_imag[i], total_sci_pix) - ref_coron_imag

    g_obwfs = np.zeros([total_pupil_pix, 2, num_all_modes])
    for i in range(num_all_modes):
        g_obwfs[:, 0, i] = np.reshape(efield_obwfs_real[i], total_pupil_pix) - ref_obwfs_real
        g_obwfs[:, 1, i] = np.reshape(efield_obwfs_real[i], total_pupil_pix) - ref_obwfs_imag

    g_obwfs_downsampled = np.zeros([int(np.square(n_sub_pix)), 2, num_all_modes])
    for i in range(num_all_modes):
        efields_per_mode_wfs_real_sub = np.reshape(matrix_subsample(efield_obwfs_real[i], n_sub_pix, n_sub_pix),
                                                   int(np.square(n_sub_pix))) / subsample_factor
        efields_per_mode_wfs_imag_sub = np.reshape(matrix_subsample(efield_obwfs_imag[i], n_sub_pix, n_sub_pix),
                                                   int(np.square(n_sub_pix))) / subsample_factor
        g_obwfs_downsampled[:, 0, i] = efields_per_mode_wfs_real_sub - ref_wfs_sub.real
        g_obwfs_downsampled[:, 1, i] = efields_per_mode_wfs_imag_sub - ref_wfs_sub.imag

    matrix = {"ref_image_plane": ref_coron,
              "ref_wfs_plane": ref_obwfs_downsampled,
              "senitivity_image_plane": g_coron,
              "sensitvity_wfs_plane": g_obwfs_downsampled}

    return matrix


def sort_1d_mus_per_actuator(mus, nmodes, nsegs):
    num_actuators = nmodes * nsegs
    coeffs_numaps = np.zeros([nmodes, num_actuators])
    for mode in range(nmodes):
        coeffs_tmp = np.zeros([num_actuators])
        for seg in range(nsegs):
            coeffs_tmp[mode + seg * nmodes] = mus[mode + seg * nmodes]  # arranged per modal basis
        coeffs_numaps[mode] = coeffs_tmp  # arranged into 5 groups of 600 elements and in units of nm

    return coeffs_numaps


def sort_1d_mus_per_seg(mus, nmodes, nsegs):
    coeffs_table = np.zeros([nmodes, nsegs])
    for mode in range(nmodes):
        for seg in range(nsegs):
            coeffs_table[mode, seg] = mus[mode + seg * nmodes]
    return coeffs_table


def calc_mean_tolerance_per_mode(opt_wavescale, mus, nmodes, nsegs, tscale):
    coeffs_table = sort_1d_mus_per_seg(mus, nmodes, nsegs)
    Qtotal = np.diag(np.asarray(mus ** 2))

    total_dynamic_tolerances = np.diag(0.0001 * opt_wavescale ** 2 * Qtotal)
    per_mode_dynamic_tolerances = []
    for mode in range(nmodes):
        individual_dynamic_tolerance = 0.0001 * opt_wavescale ** 2 * (coeffs_table[mode] ** 2)
        per_mode_dynamic_tolerances.append(individual_dynamic_tolerance)

    per_mode_temporal_tolerances = np.array(per_mode_dynamic_tolerances)
    return total_dynamic_tolerances, np.sqrt(np.mean(per_mode_temporal_tolerances)) * 1e3 * tscale


def generate_tolerance_table(tel, Q_per_mode, Q_total, c_per_mode, c_total, contrast_floor,
                             opt_wavescale, opt_tscale, data_dir):
    """
    Creates a tolerance table which includes individual RMS weights across all segments per modal basis (can be
    segment-level Zernike, or Harris modes), contrast allocation for each mode, total contrast due to all modes, and
    telescope properties.

    Parameters
    ----------
    tel : class instance of the telescope simulator
        The simulator for which the tolerance analysis is computed.
    nmodes : int
        Total number of local modes used to poke a segment (in case of a mid-order tolerance analysis)
        or total number of global Zernike modes (for high-order tolerance analysis).
    Q_per_mode : 1d numpy array
        RMS tolerance allocation across all segments for only one mode
    Q_total : float
        Total RMS tolerance allocation all segments for all nmodes.
    c_per_mode : float
        Average contrast in DH when only one modal tolerance surface map is applied on the tel primary mirror.
    c_total : float
        Average contrast in DH when all tolerance surface maps are applied on the tel primary mirror.
    contrast_floor : float
        Average minimum static contrast in DH, in presence of no external wavefront aberration.
    opt_wavescale : float
        The extra delta_wf scale multiplied to Q_total in the batch / recursive estimation algorithm.
    opt_tscale : float
        Optimal exposure of the WFS found from the batch or recursive estimation algorithm to achieve the
        target DH contrast.
    data_dir : str
        path to save the tables.

    Returns
    -------
    tables : tuple of length 2
        Astropy table

    """

    mode = np.arange(0, len(Q_per_mode), 1)
    data = np.array([mode, Q_per_mode, c_per_mode]).T
    df1 = pd.DataFrame(data)
    df1.columns = ["Mode Number", "Tolerance (in pm)", "Contrast"]
    df1.loc[len(df1.index)] = ['RMS Total', Q_total, c_total]

    table1 = QTable.from_pandas(df1)

    df2 = pd.DataFrame()
    df2[''] = None
    df2['Telescope'] = ['total segs', 'diam', 'seg diam', 'contrast_floor', 'iwa', 'owa', 'opt_wavescale', 'opt_tscale']
    df2['Values'] = [tel.nseg, format(tel.diam, ".2f"), format(tel.segment_circumscribed_diameter, ".2f"),
                     format(contrast_floor, ".2f"), tel.iwa, tel.owa, opt_wavescale, opt_tscale]

    table2 = QTable.from_pandas(df2)

    pd.concat([df1, df2], axis=1).to_csv(os.path.join(data_dir, 'tolerance_table.csv'))
    table1.write(os.path.join(data_dir, 'tolerance_table1.txt'), format='latex', overwrite=True)
    table2.write(os.path.join(data_dir, 'tolerance_table2.txt'), format='latex', overwrite=True)

    return table1, table2
