import hcipy
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.interpolate import griddata

from pastis.config import CONFIG_PASTIS
from pastis.util import sort_1d_mus_per_actuator


def plot_multimode_surface_maps(type, tel, mus, num_modes, mirror, cmin, cmax, wavescale=None, data_dir=None, fname=None):

    """Creates surface deformation maps (not WFE) for localized wavefront aberrations.
    The input mode coefficients 'mus' are in units of *WFE* and need to be grouped by segment, meaning the array holds
    the mode coefficients as:
        mode1 on seg1, mode2 on seg1, ..., mode'nmodes' on seg1, mode1 on seg2, mode2 on seg2 and so on.
    Parameters:
    -----------
    type: str
        can be either of 'static' and 'dynamic'
    tel : class instance of internal simulator
        the simulator to plot the surface maps for
    mus : 1d array
        1d array of standard deviations for all modes on each segment, in nm WFE
    num_modes : int
        number of local modes used to poke each segment
    mirror : str
        'harris_seg_mirror' or 'seg_mirror', segmented mirror of simulator 'tel' to use for plotting
    cmin : float
        minimum value for colorbar
    cmax : float
        maximum value for colorbar
    data_dir : str, default None
        path to save the plots; if None, then not saved to disk
    fname : str, default None
        file name for surface maps saved to disk
    """
    if fname is None:
        fname = f'surface_on_{mirror}'

    mus_per_actuator = sort_1d_mus_per_actuator(mus, num_modes, tel.nseg)  # in nm

    mu_maps = []
    for mode in range(num_modes):
        coeffs = mus_per_actuator[mode]
        if mirror == 'harris_seg_mirror':
            tel.harris_sm.actuators = coeffs * 1e-9 / 2  # in meters of surface
            mu_maps.append(tel.harris_sm.surface)  # in m
        if mirror == 'seg_mirror':
            tel.sm.actuators = coeffs * 1e-9 / 2  # in meters of surface
            mu_maps.append(tel.sm.surface)  # in m

    plot_norm = TwoSlopeNorm(vcenter=0, vmin=cmin, vmax=cmax)
    for i in range(num_modes):
        plt.figure(figsize=(7, 5))

        if type == 'dynamic':
            if wavescale is not None:
                scale = np.sqrt(0.0001 * wavescale ** 2)
                hcipy.imshow_field((mu_maps[i]) * 1e12 * scale, norm=plot_norm, cmap='RdBu')
                cbar = plt.colorbar()
                cbar.set_label("Surface (pm/s)", fontsize=10)
        else:
            hcipy.imshow_field((mu_maps[i]) * 1e12, norm=plot_norm, cmap='RdBu')
            cbar = plt.colorbar()
            cbar.set_label("Surface (pm)", fontsize=10)

        cbar.ax.tick_params(labelsize=10)
        plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)
        plt.tight_layout()

        if data_dir is not None:
            os.makedirs(os.path.join(data_dir, 'mu_maps'), exist_ok=True)
            image_name = fname + f'_mode_{i}.pdf'
            plt.savefig(os.path.join(data_dir, 'mu_maps', image_name))


def plot_iter_wf(Qharris, wavescale_min, wavescale_max, wavescale_step,
                 TimeMinus, TimePlus, Ntimes, result_wf_test, contrast_floor, C_TARGET, Vmag, data_dir):
    delta_wf = []
    for wavescale in range(wavescale_min, wavescale_max, wavescale_step):
        wf = np.sqrt(np.mean(np.diag(0.0001 * wavescale ** 2 * Qharris))) * 1e3
        delta_wf.append(wf)

    texp = np.logspace(TimeMinus, TimePlus, Ntimes)
    font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 20}

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
    plt.ylabel(r"$ \Delta $ contrast", fontsize=20)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='upper center', fontsize=20)
    plt.tick_params(top=True, bottom=True, left=True,
                    right=True, labelleft=True, labelbottom=True,
                    labelsize=20)
    plt.tick_params(axis='both', which='major', length=10, width=2)
    plt.tick_params(axis='both', which='minor', length=6, width=2)
    plt.grid()
    plt.savefig(os.path.join(data_dir, 'contrast_wf_%s_%d_%d_%d.png' % (C_TARGET, wavescale_min, wavescale_max, wavescale_step)))


def plot_pastis_matrix(pastis_matrix, data_dir, vcenter, vmin, vmax):
    """
    Parameters
    ----------
    pastis_matrix : 2d array
        PASTIS matrix, where each element is stored in units of contrast/nm**2.
    data_dir : str
        path to save the image
    vcenter : float
        center of matplotlib colorbar
    vmin : float
        minimum extent of matplotlib colorbar
    vmax : float
        maximum extent of matplotlib colorbar
    """
    print("hello6")
    clist = [(0.1, 0.6, 1.0), (0.05, 0.05, 0.05), (0.8, 0.5, 0.1)]
    blue_orange_divergent = LinearSegmentedColormap.from_list("custom_blue_orange", clist)

    list_of_ticks = np.arange(0, pastis_matrix.shape[0], 20)

    plt.figure(figsize=(6, 4), dpi=150)
    norm_mat = TwoSlopeNorm(vcenter, vmin, vmax)
    plt.imshow(pastis_matrix, origin='lower', cmap=blue_orange_divergent, norm=norm_mat)
    plt.title(r"PASTIS matrix $M$", fontsize=15)
    plt.xlabel("Mode Index", fontsize=15)
    plt.ylabel("Mode Index", fontsize=15)
    plt.tick_params(labelsize=10)
    plt.xticks(list_of_ticks)
    plt.yticks(list_of_ticks)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label(r"in units of ${contrast\ per\ {nm}^2}$", fontsize=10)
    plt.tight_layout()

    plt.savefig(os.path.join(data_dir, 'pastis_matrix.png'))


def create_segmented_harris_mirror(tel, filepath, pad_orientation, thermal=True, mechanical=True, other=True):
    """Create an actuated segmented mirror with a modal basis made of the thermal modes provided by Harris.

    Thermal modes: a, h, i, j, k
    Mechanical modes: e, f, g
    Other modes: b, c, d

    If all modes are created, they will be ordered as:
    a, h, i, j, k, e, f, g, b, c, d
    If only a subset is created, the ordering will be retained but the non-chosen modes dropped.

    Parameters
    ----------
    tel : class instance of the internal simulator
        telescope primary mirror on which the thermal basis is to be projected.
    filepath : string
        absolute path to the xls spreadsheet containing the Harris segment modes
    pad_orientation : ndarray
        angles of orientation of the mounting pads of the primary, in rad, one per segment

    Returns
    -------
    harris_mode_basis : hcipy field
        all the basis vector projected on the segmented mirror
    """

    # Read the spreadsheet containing the Harris segment modes

    df = pd.read_excel(filepath)

    # Read all modes as arrays
    valuesA = np.asarray(df.a)
    valuesB = np.asarray(df.b)
    valuesC = np.asarray(df.c)
    valuesD = np.asarray(df.d)
    valuesE = np.asarray(df.e)
    valuesF = np.asarray(df.f)
    valuesG = np.asarray(df.g)
    valuesH = np.asarray(df.h)
    valuesI = np.asarray(df.i)
    valuesJ = np.asarray(df.j)
    valuesK = np.asarray(df.k)

    seg_x = np.asarray(df.X)
    seg_y = np.asarray(df.Y)
    tel.harris_seg_diameter = np.max([np.max(seg_x) - np.min(seg_x), np.max(seg_y) - np.min(seg_y)])

    pup_dims = tel.pupil_grid.dims
    x_grid = np.asarray(df.X) * tel.segment_circumscribed_diameter / tel.harris_seg_diameter
    y_grid = np.asarray(df.Y) * tel.segment_circumscribed_diameter / tel.harris_seg_diameter
    points = np.transpose(np.asarray([x_grid, y_grid]))

    seg_evaluated = tel._create_evaluated_segment_grid()

    def _transform_harris_mode(values, xrot, yrot, points, seg_evaluated, seg_num):
        """ Take imported Harris mode data and transform into a segment mode on our aperture. """
        zval = griddata(points, values, (xrot, yrot), method='linear')
        zval[np.isnan(zval)] = 0
        zval = zval.ravel() * seg_evaluated[seg_num]
        return zval

    harris_base = []
    for seg_num in range(0, tel.nseg):
        mode_set_per_segment = []

        grid_seg = tel.pupil_grid.shifted(-tel.seg_pos[seg_num])
        x_line_grid = np.asarray(grid_seg.x)
        y_line_grid = np.asarray(grid_seg.y)

        # Rotate the modes grids according to the orientation of the mounting pads
        phi = pad_orientation[seg_num]
        x_rotation = x_line_grid * np.cos(phi) + y_line_grid * np.sin(phi)
        y_rotation = -x_line_grid * np.sin(phi) + y_line_grid * np.cos(phi)

        # Transform all needed Harris modes from data to modes on our segmented aperture
        # Use only the sets of modes that have been specified in the input parameters
        if thermal:
            ZA = _transform_harris_mode(valuesA, x_rotation, y_rotation, points, seg_evaluated, seg_num)
            ZH = _transform_harris_mode(valuesH, x_rotation, y_rotation, points, seg_evaluated, seg_num)
            ZI = _transform_harris_mode(valuesI, x_rotation, y_rotation, points, seg_evaluated, seg_num)
            ZJ = _transform_harris_mode(valuesJ, x_rotation, y_rotation, points, seg_evaluated, seg_num)
            ZK = _transform_harris_mode(valuesK, x_rotation, y_rotation, points, seg_evaluated, seg_num)
            mode_set_per_segment.extend([ZA, ZH, ZI, ZJ, ZK])
        if mechanical:
            ZE = _transform_harris_mode(valuesE, x_rotation, y_rotation, points, seg_evaluated, seg_num)
            ZF = _transform_harris_mode(valuesF, x_rotation, y_rotation, points, seg_evaluated, seg_num)
            ZG = _transform_harris_mode(valuesG, x_rotation, y_rotation, points, seg_evaluated, seg_num)
            mode_set_per_segment.extend([ZE, ZF, ZG])
        if other:
            ZB = _transform_harris_mode(valuesB, x_rotation, y_rotation, points, seg_evaluated, seg_num)
            ZC = _transform_harris_mode(valuesC, x_rotation, y_rotation, points, seg_evaluated, seg_num)
            ZD = _transform_harris_mode(valuesD, x_rotation, y_rotation, points, seg_evaluated, seg_num)
            mode_set_per_segment.extend([ZB, ZC, ZD])

        harris_base.append(mode_set_per_segment)

    # Create full mode basis of selected Harris modes on all segments
    harris_base = np.asarray(harris_base)
    tel.n_harris_modes = harris_base.shape[1]
    harris_base = harris_base.reshape(tel.nseg * tel.n_harris_modes, pup_dims[0] ** 2)
    harris_mode_basis = hcipy.ModeBasis(np.transpose(harris_base), grid=tel.pupil_grid)
    tel.harris_sm = hcipy.optics.DeformableMirror(harris_mode_basis)

    return harris_mode_basis


def plot_harris_mode(tel, data_dir):

    fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')
    pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
    harris_mode_basis = tel.create_segmented_harris_mirror(fpath, pad_orientations,
                                                           thermal=True, mechanical=False, other=False)

    plt.figure(figsize=(40, 20))
    plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    plt.title("Segment Level 1mK Faceplates Silvered", fontsize=30)
    plot_norm1 = TwoSlopeNorm(vcenter=0, vmin=-10,  vmax=10)
    plt.imshow(np.reshape(harris_mode_basis[0], (1000, 1000))[163:263, 804:904], cmap='RdBu', norm=plot_norm1)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("pm", fontsize=30)

    plt.subplot2grid((2, 6), (0, 2), colspan=2)
    plt.title("Segment Level 1mK bulk", fontsize=30)
    plot_norm2 = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=1)
    plt.imshow(np.reshape(harris_mode_basis[1], (1000, 1000))[163:263, 804:904], cmap='RdBu', norm=plot_norm2)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("pm", fontsize=30)

    plt.subplot2grid((2, 6), (0, 4), colspan=2)
    plt.title("Segment Level 1mK gradient radial", fontsize=30)
    plot_norm3 = TwoSlopeNorm(vcenter=0, vmin=-1.5, vmax=1.5)
    plt.imshow(np.reshape(harris_mode_basis[2], (1000, 1000))[163:263, 804:904],cmap='RdBu', norm=plot_norm3)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("pm", fontsize=30)

    plt.subplot2grid((2, 6), (1, 1), colspan=2)
    plt.title("Segment Level 1mK gradient X lateral", fontsize=30)
    plot_norm4 = TwoSlopeNorm(vcenter=0, vmin=-2.5, vmax=2.5)
    plt.imshow(np.reshape(harris_mode_basis[3], (1000, 1000))[163:263, 804:904], cmap='RdBu', norm=plot_norm4)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("pm", fontsize=30)

    plt.subplot2grid((2, 6), (1, 3), colspan=2)
    plt.title("Segment Level 1mK gradient Z axial",fontsize=30)
    plot_norm5 = TwoSlopeNorm(vcenter=0, vmin=-3,  vmax=3)
    plt.imshow(np.reshape(harris_mode_basis[4], (1000, 1000))[163:263, 804:904], cmap='RdBu', norm=plot_norm5)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("pm", fontsize=30)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'harris_modal_basis.png'))

