import numpy as np

PRECISION = np.double


def get_m8(cube, method='local'):
    """this function calculate the 8 neighbors average for subtracting the background,
     include the case when the pixel is on the edge and
    the cube are a 3D
    :param cube: the cube of the image
    :param method: the method of calculating the 8 neighbors average or global mean.
    :return: the 8 neighbors average cube in this shape"""

    row_num, col_num, band_num = cube.shape
    if method == 'local':
        m8cube = np.zeros(shape=(row_num, col_num, band_num), dtype=PRECISION)

        m8cube[1:row_num - 1, 1:col_num - 1] = (cube[1:row_num - 1, 2:col_num] + cube[1:row_num - 1, 0:col_num - 2] +
                                                cube[2:row_num, 1:col_num - 1] + cube[0:row_num - 2, 1:col_num - 1] +
                                                cube[2:row_num, 2:col_num] + cube[2:row_num, 0:col_num - 2] +
                                                cube[0:row_num - 2, 2:col_num] + cube[0:row_num - 2, 0:col_num - 2]) / 8

        # the edge pixels
        m8cube[0, 1:col_num - 1] = np.squeeze((cube[0, 2:col_num] + cube[0, 0:col_num - 2] +
                                               cube[1, 1:col_num - 1] + cube[1, 2:col_num] + cube[1, 0:col_num - 2]) / 5)
        m8cube[row_num - 1, 1:col_num - 1] = np.squeeze((cube[row_num - 1, 2:col_num] + cube[row_num - 1, 0:col_num - 2] +
                                                         cube[row_num - 2, 0:col_num - 2] + cube[row_num - 2,
                                                                                            1:col_num - 1] + cube[
                                                                                                             row_num - 2,
                                                                                                             2:col_num]) / 5)

        m8cube[1:row_num - 1, 0] = np.squeeze((cube[0:row_num - 2, 0] + cube[2:row_num, 0] +
                                               cube[0:row_num - 2, 1] + cube[2:row_num, 1] + cube[1:row_num - 1, 1]) / 5)
        m8cube[1:row_num - 1, col_num - 1] = np.squeeze((cube[0:row_num - 2, col_num - 1] + cube[2:row_num, col_num - 1] +
                                                         cube[0:row_num - 2, col_num - 2] + cube[1:row_num - 1,
                                                                                            col_num - 2] + cube[2:row_num,
                                                                                                           col_num - 2]) / 5)

        # the corner pixels
        m8cube[0, 0] = np.squeeze((cube[0, 1] + cube[1, 0] + cube[1, 1]) / 3)
        m8cube[0, col_num - 1] = np.squeeze((cube[0, col_num - 2] + cube[1, col_num - 1] + cube[1, col_num - 2]) / 3)
        m8cube[row_num - 1, 0] = np.squeeze((cube[row_num - 1, 1] + cube[row_num - 2, 0] + cube[row_num - 2, 1]) / 3)
        m8cube[row_num - 1, col_num - 1] = np.squeeze((cube[row_num - 1, col_num - 2] + cube[row_num - 2, col_num - 1] +
                                                       cube[row_num - 2, col_num - 2]) / 3)

    elif method == 'global':
        m8cube = np.mean(cube, (0, 1))

    else:
        raise ValueError('method must be "local" or "global"')

    return m8cube


def get_cov8(cube, method='local'):
    """this function calculate the covariance matrix of the cube using the 8 neighbors average
    :param cube: the cube of the image
    :param m8: the 8 neighbors average cube
    :param method: the method of calculating the 8 neighbors average or global mean.
    :return: the covariance matrix of the cube"""

    rows, cols, bands = cube.shape
    
    if method == "local":
        m8_cube = get_m8(cube, method)
        x = np.subtract(cube, m8_cube, dtype=PRECISION)
        x = x.reshape(rows * cols, bands)  # flatten to 2D array
        return np.cov(x, rowvar=False, bias=False)
    elif method == "global":
        x = cube.reshape(rows * cols, bands)  # flatten to 2D array
        return np.cov(x, rowvar=False, bias=False)  # compute covariance

def get_autocorr(cube):
    """this function caluclate the autocorellation matrix estimation
    :param cube: the cube of the image
    """
    rows, cols, bands = cube.shape
    x = cube.reshape(rows * cols, bands)
    return np.dot(x.T , x) / x.shape[0]
    

if __name__ == "__main__":
    import spectral as spy

    pass
