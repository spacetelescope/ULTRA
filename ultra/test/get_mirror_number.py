import numpy as np
import os

from pastis.config import CONFIG_PASTIS
from pastis.simulators.scda_telescopes import HexRingAPLC
import pastis.util as util
import hcipy

if __name__ == '__main__':

    # Set number of rings
    NUM_RINGS = 5

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
    num_actuators = tel.harris_sm.num_actuators

    harris_actuators = np.zeros(num_actuators)
    segnum = 85
    segid = 5 * (segnum - 1)  # increase by step size of +5
    harris_actuators[segid] = 1e-9
    tel.harris_sm.actuators = harris_actuators
    psf, inter = tel.calc_psf(display_intermediate=False, return_intermediate='efield', norm_one_photon=True)
    hcipy.imshow_field(inter['harris_seg_mirror'].phase, mask=tel.aperture)
