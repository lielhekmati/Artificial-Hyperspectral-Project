import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import datetime


def calc_stats(target_cube, no_target_cube, bins=1000):
    """this function calculates the statistics of the detection algorithm
    target_cube - the cube with the target
    no_target_cube - the cube without the target
    bins - the number of bins in the histogram

    returns: the histogram of the WT+NT, the false positive rate, the true positive rate, the thresholds
    """
    bins_value = 1000
    x = np.linspace(-bins_value, bins_value, bins)

    histogram_wt = np.histogram(target_cube, x)
    histogram_nt = np.histogram(no_target_cube, x)

    fpr, tpr, thresholds = roc_curve(np.concatenate([np.zeros_like(no_target_cube.flatten()),
                                                     np.ones_like(target_cube.flatten())]),
                                     np.concatenate([no_target_cube.flatten(), target_cube.flatten()]))

    return histogram_wt, histogram_nt, fpr, tpr, thresholds


def plot_stats(hist_wt, hist_nt, fpr, tpr, thresholds,fig, ax,color,
               legends=None, algo_name='MF', name_of_the_dataset=None,
               name_of_estimation_method=None, save_fig=True):
    """this function plots the results of the detection algorithm
    axis - the axis of the cumulative probability
    hist_wt - the histogram of the WT
    hist_nt - the histogram of the NT
    inv_cumulative_wt - the inverse cumulative probability of the WT
    inv_cumulative_nt - the inverse cumulative probability of the NT
    legends - the legends of the plots
    algo_name - the name of the algorithm

    returns: None"""
    number_of_cubes = len(hist_wt)
    if len(hist_wt) != len(hist_nt):
        raise ValueError('hist_wt and hist_nt must have the same length')
    if len(fpr) != len(tpr):
        raise ValueError('fpr and tpr must have the same length')
    if legends is None:
        print('legends is None, using default legends')
        legends = ['Cube ' + str(i) for i in range(number_of_cubes)]
    if name_of_estimation_method is None:
        print('name_of_estimation_method is None, using default name')
        name_of_estimation_method = "Generic"
    if name_of_the_dataset is None:
        print('name_of_the_dataset is None, using default name')
        name_of_the_dataset = "Generic"

    title1 = f'histogram results'
    title2 = f'log10 histogram results'
    title3 = f'inverse cumulative probability'
    title4 = f'ROC curve with limited pfa'

    for i in range(number_of_cubes):

        ax[0].plot(hist_wt[i][1][1:], hist_wt[i][0],
                      label=f'{legends[i]}_WT', color=color, linewidth=1)
        ax[0].plot(hist_nt[i][1][1:], hist_nt[i][0],
                      '--', label=f'{legends[i]}_NT', color=color, linewidth=1)
        ax[0].set_xlim(-200, 400)
        ax[0].set_ylabel('Number of samples')
        ax[0].set_xlabel('Detection score')
        ax[0].grid(True)
        ax[0].legend(loc='upper left')

        ax[1].plot(hist_wt[i][1][1:], hist_wt[i][0],
                      label=f'{legends[i]}_WT', color=color, linewidth=1)
        ax[1].plot(hist_nt[i][1][1:], hist_nt[i][0],
                      '--', label=f'{legends[i]}_NT', color=color, linewidth=1)
        ax[1].set_xlim(-500, 500)
        ax[1].set_ylabel('Number of samples')
        ax[1].set_xlabel('Detection score')
        ax[1].grid(True)
        ax[1].set_yscale('log')
        ax[1].legend(loc='upper left')

        ax[2].plot(thresholds[i][::-1], tpr[i][::-1],
                      label=f'{legends[i]}_WT', color=color, linewidth=1)
        ax[2].plot(thresholds[i][::-1], fpr[i][::-1],
                      '--', label=f'{legends[i]}_NT', color=color, linewidth=1)
        # ax[1, 0].set_xlim([np.min(hist_wt[i][1][1:]), np.max(hist_wt[i][1][1:])])
        ax[2].set_xlim(-500, 500)
        ax[2].set_ylabel('Probability')
        ax[2].set_xlabel('Detection score')
        ax[2].grid(True)
        ax[2].legend(loc='lower left')

        idx = len(fpr[i][fpr[i] <= 0.1])
        roc_auc = auc(fpr[i][:idx], tpr[i][:idx])
        print(f"the AUC for {legends[i]} is {roc_auc}")
        ax[3].plot(fpr[i], tpr[i],
                   label=f"{legends[i]}: AUC = {np.round(roc_auc, 3)}", color=color,
                   linewidth=1)
        X_auc = roc_auc
        ax[3].set_xlabel('False Positive Rate')
        ax[3].set_ylabel('True Positive Rate')
        ax[3].set_xlim([0, 0.1])
        ax[3].grid(True)
        ax[3].legend(loc='lower right')

    ax[0].set_title(title1)
    ax[1].set_title(title2)
    ax[2].set_title(title3)
    ax[3].set_title(title4)

    fig.tight_layout()
    if save_fig:
        try:
            plt.savefig(f"plots/{name_of_the_dataset}_{name_of_estimation_method}_{datetime.datetime.now().strftime('_%d_%m_%Y__%H_%M_%S')}.png")
        except Exception as e:
            print(e)
            os.makedirs('plots')
            plt.savefig(f"plots/{name_of_the_dataset}_{name_of_estimation_method}_{datetime.datetime.now().strftime('_%d_%m_%Y__%H_%M_%S')}.png")
   # plt.show()


if __name__ == '__main__':
    print('This is a function file, not a main file')
