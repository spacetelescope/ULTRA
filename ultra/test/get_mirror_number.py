"""
Code to visualize mirror number against actuator poke.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from pastis.config import CONFIG_PASTIS
from pastis.simulators.scda_telescopes import HexRingAPLC
import pastis.util as util
import hcipy

if __name__ == '__main__':

    # Set number of rings
    NUM_RINGS = 1

    # Define the type of WFE.
    WHICH_DM = 'harris_seg_mirror'

    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)
        NUM_MODES = 5  # TODO: works only for thermal modes currently

    optics_dir = os.path.join(util.find_repo_location(), 'data', 'SCDA')
    sampling = 4
    tel = HexRingAPLC(optics_dir, NUM_RINGS, sampling)

    tel.create_segmented_harris_mirror(fpath, pad_orientations, thermal=True, mechanical=False, other=False)

    num_actuators = tel.harris_sm.num_actuators   # equal to NUM_MODES * Total segments

    # specify the segment to be poked.
    segnum = 3   # largest values is equal to total no. of segments
    mode_number = 0  # can be 0, 1, 2 ... NUM_MODES-1
    actuator_id = NUM_MODES * (segnum - 1) + mode_number

    harris_actuators = np.zeros(num_actuators)
    harris_actuators[actuator_id] = 1e-9   # poking an actuator with 1nm aberration.
    tel.harris_sm.actuators = harris_actuators
    psf, inter = tel.calc_psf(display_intermediate=False, return_intermediate='efield', norm_one_photon=True)

    plt.figure()
    hcipy.imshow_field(inter['harris_seg_mirror'].phase, mask=tel.aperture)
    plt.show()
