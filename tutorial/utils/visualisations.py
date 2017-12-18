import numpy as np

import matplotlib.pyplot as plt


def plot_reliability_diagram(score, labels, linspace, scores_set, legend_set,
                             alpha=1, scatter_prop=0.0, fig=None, n_bins=10,
                             bins_count=True, title=None, **kwargs):
    '''
    Parameters
    ==========
    scores_set : list of array_like of floats
        List of scores given by different methods, the first one is always the
        original one
    labels : array_like of ints
        Labels corresponding to the scores
    legend_set : list of strings
        Description of each array in the scores_set
    alpha : float
        Laplace regularization when computing the elements in the bins
    scatter_prop : float
        If original first specifies the proportion of points (score, label) to
        show
    fig : matplotlib.pyplot.figure
        Plots the axis in the given figure
    bins_count : bool
        If True, show the number of samples in each bin

    Regurns
    =======
    fig : matplotlib.pyplot.figure
        Figure with the reliability diagram
    '''
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)

    n_lines = len(legend_set)

    # Draw the empirical values in a histogram style
    # TODO careful that now the min and max depend on the scores
    s_min = min(score)
    s_max = max(score)
    bins = np.linspace(s_min, s_max, n_bins+1)
    hist_tot = np.histogram(score, bins=bins)
    hist_pos = np.histogram(score[labels == 1], bins=bins)
    edges = np.insert(bins, np.arange(len(bins)), bins)
    empirical_p = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+2*alpha)
    empirical_p = np.insert(empirical_p, np.arange(len(empirical_p)),
                            empirical_p)
    p = plt.plot(edges[1:-1], empirical_p, label='original')
    # Draw the centroids of each bin
    centroids = [np.mean(np.append(
                 score[np.where(np.logical_and(score >= bins[i],
                                               score < bins[i+1]))],
                 bins[i]+0.05)) for i in range(len(hist_tot[1])-1)]
    proportion = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+alpha*2)
    plt.plot(centroids, proportion, 'o', color=p[-1].get_color(), linewidth=2,
             label='centroid')
    for (x, y, text) in zip(centroids, proportion, hist_tot[0]):
        if y < 0.95:
            y += 0.05
        else:
            y -= 0.05
        plt.text(x, y, text, horizontalalignment='center',
                 verticalalignment='center')

    # Draw the rest of the lines
    for (scores, legend) in zip(scores_set, legend_set):
        # reliability_diagram(scores, labels, marker='o-', label=legend,
        #                     linewidth=n_lines, alpha=alpha, n_bins=n_bins,
        #                     **kwargs)
        plt.plot(linspace, scores, label=legend, linewidth=n_lines)
        n_lines -= 1

    # Draw some samples with the labels
    if scatter_prop:
        n_points = int(scatter_prop*len(labels))
        plt.plot(score[:n_points], labels[:n_points], 'kx',
                 label='samples ({:d}%)'.format(int(scatter_prop*100)),
                 markersize=6, markeredgewidth=1, alpha=0.4)

    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$\hat p$')
    ax.legend(loc='lower right')
    ax.grid(True)

    return fig
