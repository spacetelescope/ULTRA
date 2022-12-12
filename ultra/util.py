import numpy as np


def matrix_subsample(matrix, n, m):
    # return a matrix of shape (n,m)
    arr_sum = []
    length = matrix.shape[0] // n  # block length
    breadth = matrix.shape[1] // m  # block breadth
    for i in range(n):
        for j in range(m):
            sum_pixels = np.sum(matrix[i*length: (i+1)*length, j*breadth: (j+1)*breadth])
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
