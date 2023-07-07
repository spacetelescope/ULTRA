import pandas as pd
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import hcipy
import os

from pastis.config import CONFIG_PASTIS
from pastis.simulators.scda_telescopes import HexRingAPLC
import pastis.util as util


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
    filepath : string
        absolute path to the xls spreadsheet containing the Harris segment modes
    pad_orientation : ndarray
        angles of orientation of the mounting pads of the primary, in rad, one per segment
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


if __name__ == '__main__':

    # Set number of rings
    NUM_RINGS = 1

    # Define the type of WFE.
    WHICH_DM = 'harris_seg_mirror'

    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)
        NUM_MODES = 5 # TODO: works only for thermal modes currently

    optics_dir = os.path.join(util.find_repo_location(), 'data', 'SCDA')
    sampling = 4
    tel = HexRingAPLC(optics_dir, NUM_RINGS, sampling)

    harris_maps = create_segmented_harris_mirror(tel, fpath, pad_orientations, thermal=True, mechanical=True, other=True)

    size = int(np.sqrt(len(harris_maps[0])))

    # Square region of interest
    l1 = int(250)
    l2 = int(750)

    plt.figure(figsize=(14, 8))
    plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    plt.title("Segment Level 1mK Faceplates Silvered", fontsize=10)
    plot_norm1 = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
    plt.imshow(np.reshape(harris_maps[0], (size, size))[l1:l2, l1:l2], cmap='RdBu', norm=plot_norm1)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("pm", fontsize=10)

    plt.subplot2grid((2, 6), (0, 2), colspan=2)
    plt.title("Segment Level 1mK bulk", fontsize=10)
    plot_norm2 = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=1)
    plt.imshow(np.reshape(harris_maps[1], (size, size))[l1:l2, l1:l2], cmap='RdBu', norm=plot_norm2)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("pm", fontsize=10)

    plt.subplot2grid((2, 6), (0, 4), colspan=2)
    plt.title("Segment Level 1mK gradient radial", fontsize=10)
    plot_norm3 = TwoSlopeNorm(vcenter=0, vmin=-1.5, vmax=1.5)
    plt.imshow(np.reshape(harris_maps[2], (size, size))[l1:l2, l1:l2], cmap='RdBu', norm=plot_norm3)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("pm", fontsize=10)

    plt.subplot2grid((2, 6), (1, 1), colspan=2)
    plt.title("Segment Level 1mK gradient X lateral", fontsize=10)
    plot_norm4 = TwoSlopeNorm(vcenter=0, vmin=-2.5, vmax=2.5)
    plt.imshow(np.reshape(harris_maps[3], (size, size))[l1:l2, l1:l2], cmap='RdBu', norm=plot_norm4)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("pm", fontsize=10)

    plt.subplot2grid((2, 6), (1, 3), colspan=2)
    plt.title("Segment Level 1mK gradient Z axial", fontsize=10)
    plot_norm5 = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3)
    plt.imshow(np.reshape(harris_maps[4], (size, size))[l1:l2, l1:l2], cmap='RdBu', norm=plot_norm5)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("pm", fontsize=10)

    plt.tight_layout()