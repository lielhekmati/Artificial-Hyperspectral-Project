import numpy as np
from scipy.stats import t as t_dist
from Hyperspectral_class import HyperSpectralCube
from sklearn.feature_selection import mutual_info_regression


def rand_art_pca_cube(pca_data, stats_arr):
  cubef = np.zeros((pca_data.rows, pca_data.cols, pca_data.bands))
  for band in range(pca_data.bands):
          cubef[:,:,band] =  t_dist.rvs(stats_arr[band][0], loc=stats_arr[band][1], scale=stats_arr[band][2], size=(pca_data.rows, pca_data.cols))

  art_cube = HyperSpectralCube(None, cubef)

  return art_cube

def inv_pca (art_pca_cube, eigenvectors, eigenvalues) :
  cubef = np.zeros((art_pca_cube.rows, art_pca_cube.cols, art_pca_cube.bands))
  for r in range(art_pca_cube.rows):
      for c in range(art_pca_cube.cols):
          cubef[r, c, :] = np.matmul(eigenvectors, art_pca_cube.cube[r, c, :] * np.sqrt(eigenvalues))

  art_cube = HyperSpectralCube(None, cubef)

  return art_cube

def calc_t_dist (x, nu, loc, scale, const=1) :
  random_var = t_dist(df=nu, loc=loc, scale=scale)
  pdf = lambda x : random_var.pdf(x / const) / np.abs(const)
  return pdf(x)

def sum_of_t_distributions(pdf_list, dx):
  sum_pdf = pdf_list[0]
  for pdf in pdf_list[1:] :
      sum_pdf = np.convolve(sum_pdf, pdf, mode='same') * dx
      sum_pdf /= np.sum(sum_pdf) * dx

  return sum_pdf

def get_mutual_info_matrix(cube):
  new_shape = [cube.shape[0]*cube.shape[1], cube.shape[2]]
  data = cube.reshape(new_shape)
  num_of_bands = cube.shape[2]
  mutual_info_matrix = np.zeros((num_of_bands, num_of_bands))

  for i in range(num_of_bands):
    for j in range(num_of_bands):
      mutual_info_matrix[i, j] = mutual_info_regression(data[:, i].reshape(-1, 1), data[:, j])[0]

  return mutual_info_matrix
