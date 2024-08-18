
import numpy as np
from local_global_mean_covariance import get_m8, get_cov8
from spectral import *
import matplotlib.pyplot as plt
from PCA import get_pca
from find_nu import find_nu
from scipy.stats import t as t_dist
import datetime


class HyperSpectralCube:
    """ This class is an object the hold hyperspectral data."""

    def __init__(self, header=None, cube=None):
        """ This function initializes the object.
        :param header: the header of the data
        :param cube: the data cube
        """

        if header is not None:
            self.data = open_image(header)
            self.cube = self.data.load(dtype=np.single).copy()
        elif cube is not None:
            self.data = None
            self.cube = cube
        else:
            raise Exception('You must provide either a header or a cube')

        self.rows, self.cols, self.bands = self.cube.shape
        self.mean = None
        self.cov = None
        self.nu = None
        self.eigenvectors = None
        self.eigenvalues = None

    def calc_mean(self, method):
        """ This function calculates the mean of the data.
        :param method: the method to calculate the mean
        :return: None
        """
        self.mean = get_m8(self.cube, method)

    def calc_cov(self, method):
        """ This function calculates the covariance of the data.
        :param method: the method to calculate the covariance
        :return: None
        """
        self.cov = get_cov8(self.cube, method)

    def calc_nu(self, method='Constant2'):
        """ This function calculates the degree of freedom for the data.
        :param method: the method to calculate the degree of freedom
        :return: None
        """
        self.nu = find_nu(self.cube, self.mean, self.cov, method=method)

    def pca_transform(self):
        """ This function transforms the data to the PCA space.
        :return: None
        """
        transformed_cube, self.eigenvectors, self.eigenvalues = get_pca(self.cube, self.mean, self.cov)
        return HyperSpectralCube(cube=transformed_cube)

    def plot_band(self, band, title=None):
        """ This function plots a specific band of the data.
        :param band: the band to plot
        :param title: the title of the plot
        :return: None
        """
        plt.figure()
        plt.imshow(self.cube[:, :, band], cmap='gray')
        plt.colorbar()
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_all_bands(self, index_list, title=None):
        """ This function plots all the bands of the data.
        :param title: the title of the plot
        :return: None
        """
        for band in index_list:
            if title is not None:
                title = title + ' band ' + str(band)
            else:
                title = 'band ' + str(band)
            self.plot_band(band, title=title)

    def plot_mean(self, title=None):
        """ This function plots the mean of the data.
        :param title: the title of the plot
        :return: None
        """
        plt.figure()
        plt.imshow(self.mean.reshape(self.rows * self.cols, self.bands), cmap='heat')
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_cov(self, title=None):
        """ This function plots the covariance of the data.
        :param title: the title of the plot
        :return: None
        """
        plt.figure()
        plt.imshow(self.cov, cmap='hot')
        plt.colorbar()
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_nu(self, title=None):
        """ This function plots the degree of freedom of the data.
        :param title: the title of the plot
        :return: None
        """
        plt.figure()
        plt.semilogy(self.nu)
        if title is not None:
            plt.title(title)
        plt.xlabel("Band")
        plt.ylabel("DOF")
        plt.grid()
        plt.savefig(f"{title}_{datetime.datetime.now().strftime('%d_%m_%Y__%H_%M_%S')}.png")
        plt.show()

    def __str__(self):
        """ This function prints the data.
        :return: None
        """
        print(f"This is Hyperspectral cube with {self.bands} bands, {self.rows} rows and {self.cols} columns")
        print(f"The data type is {self.cube.dtype}")
        if self.nu is not None:
            print(f"The degree of freedom is: \n{self.nu}")
        return ""

if __name__ == "__main__":
    pass
