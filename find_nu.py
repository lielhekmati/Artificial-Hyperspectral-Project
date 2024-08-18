
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import t as t_dist
import torch
import torch.nn as nn
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_nu(cube, mean_matrix, cov, method='Constant2'):
    """return the nu vector for the given cube.
    to possible methods for finding nu:\n
    ['Tyler', 'KS', 'Constant2', 'Constant3', 'MLE', 'NN']
    :param cube: the given cube
    :param mean_matrix: the mean matrix of the given cube
    :param cov: the covariance matrix of the given cube
    :param method: the method for finding nu
    :return: nu vector
    """
    cube_no_mean = np.subtract(cube, mean_matrix)

    stats_arr = []
    
    if method == 'MLE':
        nu = np.zeros((cube.shape[2], 1))
        for band in range(cube.shape[2]):
            stats_arr.append(t_dist.fit((cube_no_mean[:, :, band]).flatten()))

    elif method == 'MLE fixed var':
        nu = np.zeros((cube.shape[2], 1))
        for band in range(cube.shape[2]):
            stats_arr.append(t_dist.fit((cube_no_mean[:, :, band].flatten()), fscale=1))


    else:
        raise ValueError('method not found')

    return stats_arr


if __name__ == "__main__":
    # weights_path = r"C:\Users\gast\PycharmRepos\HyperSpectralProject\weights\best_model.pt"
    # net = DOFNet()
    # net.load_state_dict(torch.load(weights_path, map_location=device))
    # print(net)
    # net.eval()
    # net.to(device)
    # import spectral as spy
    # data = spy.open_image('self_test_rad.hdr')
    # # convert the data to a numpy array
    # data = np.array(data.open_memmap())
    # data = data[:, :, 0:5].astype(np.float32)
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((224, 224))])
    # data = transform(data).to(device)
    # output = net(data)
    # print(output)
    pass