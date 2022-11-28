import numpy as np

from pastis.util import dh_mean


if __name__ == '__main__':

    im1 = np.ones((4, 4))
    mean_contrast = dh_mean(im1, im1)
