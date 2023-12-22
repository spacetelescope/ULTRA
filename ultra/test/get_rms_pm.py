"""
This script generates mean rms tolerance per ring for a SCDA telescope simulator.
"""

import os
import numpy as np

from pastis.config import CONFIG_PASTIS
import pastis.util as util
from pastis.simulators.scda_telescopes import HexRingAPLC
from pastis.util import sort_1d_mus_per_actuator

from ultra.config import CONFIG_ULTRA


def calculate_rms_surface_aberration_per_ring(tel, mus):
    """Calculates RMS tolerances per ring for an SCDA telescope simulator.

    Parameters
    ----------
    tel :  class instantiate of the internal simulator.
        the simulator to plot the surface maps for
    mus : 1d numpy array
        toleranes computed for the telescope simulator, in units of nm
    """
    num_mode = int(len(mus) / tel.nseg)
    mus_per_actuator = sort_1d_mus_per_actuator(mus, num_mode, tel.nseg)

    def calc_per_ring_rms(seg1, seg2, design_name, ring):
        for mode in range(0, num_mode):
            tel.harris_sm.flatten()
            coeffs = mus_per_actuator[mode]
            for seg in range(seg1, seg2):
                tel.harris_sm.actuators[seg * 5 + mode] = coeffs[seg * 5 + mode] * 1e-9 / 2  # in meters of surface
            surface = tel.harris_sm.surface
            aberrated_surface = surface[surface != 0]
            ring_rms = float(np.sqrt((np.square(aberrated_surface)).mean()) * 1e12)  # in units of picometers
            ring_rms = round(ring_rms, 2)
            print("RMS surface deformation for", design_name, "and", ring, "is,", "for mode:", mode, "is", ring_rms, "pm.")

    if tel.nseg == 7:
        calc_per_ring_rms(1, 7, "Hex1", "Ring1")

    if tel.nseg == 19:
        calc_per_ring_rms(1, 7, "2Hex", "Ring1")
        calc_per_ring_rms(7, 19, "2Hex", "Ring2")

    if tel.nseg == 31:
        calc_per_ring_rms(1, 7, "3Hex", "Ring1")
        calc_per_ring_rms(7, 19, "3Hex", "Ring2")
        calc_per_ring_rms(19, 31, "3Hex", "Ring3")

    if tel.nseg == 55:
        calc_per_ring_rms(1, 7, "4Hex", "Ring1")
        calc_per_ring_rms(7, 19, "4Hex", "Ring2")
        calc_per_ring_rms(19, 37, "4Hex", "Ring3")
        calc_per_ring_rms(37, 55, "4Hex", "Ring4")

    if tel.nseg == 85:
        calc_per_ring_rms(1, 7, "5Hex", "Ring1")
        calc_per_ring_rms(7, 19, "5Hex", "Ring2")
        calc_per_ring_rms(19, 37, "5Hex", "Ring3")
        calc_per_ring_rms(37, 61, "5Hex", "Ring4")
        calc_per_ring_rms(61, 85, "5Hex", "Ring5")


if __name__ == '__main__':

    # Set number of rings
    NUM_RINGS = 1

    # Define the type of WFE.
    WHICH_DM = 'harris_seg_mirror'

    optics_dir = os.path.join(util.find_repo_location(), 'data', 'SCDA')
    sampling = 4

    tel = HexRingAPLC(optics_dir, NUM_RINGS, sampling)
    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)
        NUM_MODES = 5
    tel.create_segmented_harris_mirror(fpath, pad_orientations, thermal=True, mechanical=False, other=False)

    # Thermal tolerance coefficients
    mus_data_path = CONFIG_ULTRA.get('local_path', 'local_data_path')
    mus = np.genfromtxt(os.path.join(mus_data_path, 'mus_Hex_1_1e-11.csv'), delimiter=',')

    calculate_rms_surface_aberration_per_ring(tel, mus)
