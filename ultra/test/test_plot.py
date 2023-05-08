from ultra.plotting import plot_pastis_matrix
from astropy.io import fits


if __name__ == '__main__':

    pastis_matrix = fits.getdata('/Users/asahoo/Desktop/data_repos/harris_data'
                                 '/2023-01-05T06-33-43_hexringtelescope/matrix_numerical/pastis_matrix.fits')

    plot_pastis_matrix(pastis_matrix, data_dir='/Users/asahoo/Desktop/test_plots',
                       vcenter=0, vmin=-1e-9, vmax=1e-9)


