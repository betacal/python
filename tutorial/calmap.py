import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np


def plot_calibration_map(scores_set, prob, legend_set, alpha=1, **kwargs):
    rc('text', usetex=True)
    fig_reliability_map = plt.figure('reliability_map')
    fig_reliability_map.clf()
    ax_reliability_map = plt.subplot(111)
    ax = ax_reliability_map
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlim([-0.01, 1.01])
    ax.set_xlabel((r'$s$'), fontsize=16)
    ax.set_ylabel((r'$\hat{p}$'), fontsize=16)
    n_lines = len(legend_set)
    bins = np.linspace(0, 1, 11)
    hist_tot = np.histogram(prob[0], bins=bins)
    hist_pos = np.histogram(prob[0][prob[1] == 1], bins=bins)
    centers = (bins[:-1] + bins[1:])/2.0
    empirical_p = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+2*alpha)
    ax.plot(centers, empirical_p, 'ko', label='empirical')

    for (scores, legend) in zip(scores_set, legend_set):
        if legend != 'uncalib':
            ax.plot(prob[2], scores, '-', label=legend, linewidth=n_lines,
                    **kwargs)
        n_lines -= 1
    ax.legend(loc='upper left')
    return fig_reliability_map
