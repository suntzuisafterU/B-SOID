"""
Functionality for visualizing plots and saving those plots.
"""
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from mpl_toolkits.mplot3d import Axes3D  # Despite being "unused", this import MUST stay for 3d plotting to work. PLO!
from typing import Any, Collection, Dict, List, Tuple
import inspect
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from bsoid import config
from bsoid.util import likelihoodprocessing

logger = config.initialize_logger(__name__)
matplotlib_axes_logger.setLevel('ERROR')

TM = NotImplementedError('TODO: HIGH: The source of TM has not been determined. Find and fix as such.')  # TODO: low


### New

def plot_GM_assignments_in_3d_new(data: np.ndarray, assignments, show_now=True, **kwargs) -> Tuple[object, object]:
    """
    Plot trained TSNE for EM-GMM assignments.

    :param data: 2D array, trained_tsne array (3 columns)
    :param assignments: 1D array, EM-GMM assignments
    :param show_now:
    """
    # TODO: find out why attaching the log entry/exit decorator kills the streamlit graph-rotation app
    if not isinstance(data, np.ndarray):
        err = f'Expected `data` to be of type numpy.ndarray but instead found: {type(data)} (value = {data}).'
        logger.error(err)
        raise TypeError(err)
    # Parse kwargs
    s = kwargs.get('s', 0.5)
    marker = kwargs.get('marker', 'o')
    alpha = kwargs.get('alpha', 0.8)
    title = kwargs.get('title', 'Assignments by GMM')
    azim_elev = kwargs.get('azim_elev', (70, 135))
    # Plot graph
    unique_assignments = list(np.unique(assignments))
    R = np.linspace(0, 1, len(unique_assignments))
    colormap = plt.cm.get_cmap("Spectral")(R)
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Loop over assignments
    for i, g in enumerate(unique_assignments):
        # Select data for only assignment i
        idx = np.where(assignments == g)
        # Assign to colour and plot
        ax.scatter(tsne_x[idx], tsne_y[idx], tsne_z[idx], c=colormap[i], label=g, s=s, marker=marker, alpha=alpha)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    ax.view_init(*azim_elev)
    plt.title(title)
    plt.legend(ncol=3)
    # Draw now?
    if show_now:
        plt.show()
    else:
        plt.draw()

    return fig, ax


def plot_assignment_distribution_histogram(assignments: Collection, **kwargs) -> Tuple[object, object]:
    """
    Produce histogram plot of assignments. Useful for seeing lop-sided outcomes.
    :param assignments:
    :param kwargs:
    :return:
    """
    # Arg checking
    if not isinstance(assignments, np.ndarray):
        assignments = np.array(assignments)
    # Kwarg resolution
    histtype = kwargs.get('histtype', 'stepfilled')
    # Do
    unique_assignments = np.unique(assignments)
    R = np.linspace(0, 1, len(unique_assignments))
    colormap = plt.cm.get_cmap("Spectral")(R)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, g in enumerate(unique_assignments):
        idx = np.where(assignments == g)
        plt.hist(assignments[idx], histtype=histtype, color=colormap[i])

    return fig, ax


#######################################################################################################################

@config.deco__log_entry_exit(logger)
def plot_tsne_in_3d(data, **kwargs):
    # TODO: HIGH: consider reducing the total data when plotting because, if TONS of data is
    #  plotted in 3d, it can be very laggy when viewing and when especially rotating
    """
    Plot trained tsne
    :param data: trained_tsne TODO: expand desc. and include type
    """
    if not isinstance(data, np.ndarray):
        err = f'data was expected to be of type array but instead found type: {type(data)}.'
        logger.error(err)
        raise TypeError(err)
    # TODO: low: reduce
    # Parse kwargs
    x_label = kwargs.get('x_label', 'Dim. 1')
    y_label = kwargs.get('y_label', 'Dim. 2')
    z_label = kwargs.get('z_label', 'Dim. 3')
    # Produce graph
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_x, tsne_y, tsne_z, s=1, marker='o', alpha=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.view_init(70, 135)
    plt.title('Embedding of the training set by t-SNE')
    plt.show()


def plot_duration_histogram(lengths, grp, save_fig_to_file: bool = config.SAVE_GRAPHS_TO_FILE, fig_file_prefix='duration_histogram_100ms_bins') -> object:
    """
    TODO: low: purpose
    :param lengths: 1D array, run lengths of each bout.
    :param grp: 1D array, corresponding label.
    :param save_fig_to_file: (bool)
    :param fig_file_prefix: (str)
    """
    fig, ax = plt.subplots()
    R = np.linspace(0, 1, len(np.unique(grp)))
    colormap = plt.cm.get_cmap("Spectral")(R)
    for i in range(len(np.unique(grp))):
        fig.suptitle(f"Duration histogram of {len(np.unique(TM))} behaviors")  # TODO: address `TM` source
        x = lengths[np.where(grp == i)]
        ax.hist(x, density=True, color=colormap[i], alpha=0.3, label=f'Group {i}')
    plt.legend(loc='Upper right')
    plt.show()
    if save_fig_to_file:
        fig_file_name = f'{fig_file_prefix}_{config.runtime_timestr}'
        save_graph_to_file(fig, fig_file_name)
    return fig


def plot_transition_matrix(transition_matrix: np.ndarray, fps, save_fig_to_file, fig_file_prefix) -> object:
    """
    New interface for original function named plot_tmat()
        TODO: low: purpose
    :param transition_matrix: object, transition matrix data frame  TODO: Q: what is it transitioning from?
    :param fps: scalar, camera frame-rate
    :param save_fig_to_file: bool,
    :param fig_file_prefix: str,
    :return (matplotlib.figure.Figure)
    """
    # TODO: HIGH: Important: type checking was implemented; however, sometimes
    #   DataFrames are input into this function so type checking disabled for now.

    # if not isinstance(transition_matrix, np.ndarray):
    #     err = f'Expected transition matrix to be of type numpy.ndarray but '\
    #           f'instead found: {type(transition_matrix)}.'
    #     logger.error(err)
    #     raise TypeError(err)

    fig = plt.figure()
    fig.suptitle(f"Transition matrix of {transition_matrix.shape[0]} behaviors")
    sn.heatmap(transition_matrix, annot=True)
    plt.xlabel("Next frame behavior")
    plt.ylabel("Current frame behavior")
    plt.show()
    if save_fig_to_file:
        fig_file_name = f'{fig_file_prefix}_{fps}FPS'
        save_graph_to_file(fig, fig_file_name)
    return fig


@config.deco__log_entry_exit(logger)
def plot_feats_bsoidpy_NEW(features, labels) -> None:
    """ *  *
    :param features: list, features for multiple sessions
    :param labels: list, labels for multiple sessions
    """
    # NOTE: the order of the below feature labels is arbitrary and conserved from the original
    #   implementation where the ordering was hard-coded into feature-engineering process.
    feature_labels = ("Relative snout to forepaws placement",
                      "Relative snout to hind paws placement",
                      "Inter-forepaw distance",
                      "Body length",
                      "Body angle",
                      "Snout displacement",
                      "Tail-base displacement", )
    time_str = config.runtime_timestr
    if isinstance(labels, list):
        for idx_feature_k in range(len(features)):
            labels_k = np.array(labels[idx_feature_k])
            feats_k = np.array(features[idx_feature_k])
            R = np.linspace(0, 1, len(np.unique(labels_k)))
            colormap = plt.cm.get_cmap("Spectral")(R)
            for i in range(feats_k.shape[0]):
                fig = plt.figure(facecolor='w', edgecolor='k')
                for k in range(len(np.unique(labels_k))):
                    plt.subplot(len(np.unique(labels_k)), 1, k + 1)
                    if i == 2 or i == 3 or i == 5 or i == 6:  # TODO: Q: KS: what is this equality trying to do?
                        plt.hist(feats_k[i, labels_k == k],
                                 bins=np.linspace(0, np.mean(feats_k[i, :]) + 3 * np.std(feats_k[i, :]), num=50),
                                 range=(0, np.mean(feats_k[i, :]) + 3 * np.std(feats_k[i, :])),
                                 color=colormap[k],
                                 density=True)
                        fig.suptitle(f"{feature_labels[i]} pixels")
                        plt.xlim(0, np.mean(feats_k[i, :]) + 3 * np.std(feats_k[i, :]))
                        if k < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    else:
                        plt.hist(feats_k[i, labels_k == k],
                                 bins=np.linspace(np.mean(feats_k[i, :]) - 3 * np.std(feats_k[i, :]),
                                                  np.mean(feats_k[i, :]) + 3 * np.std(feats_k[i, :]), num=50),
                                 range=(np.mean(feats_k[i, :]) - 3 * np.std(feats_k[i, :]),
                                        np.mean(feats_k[i, :]) + 3 * np.std(feats_k[i, :])),
                                 color=colormap[k],
                                 density=True)
                        plt.xlim(np.mean(feats_k[i, :]) - 3 * np.std(feats_k[i, :]),
                                 np.mean(feats_k[i, :]) + 3 * np.std(feats_k[i, :]))
                        fig.suptitle(f"{feature_labels[i]} pixels")
                        if k < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                if config.SAVE_GRAPHS_TO_FILE:
                    file_name = f'session_{idx_feature_k+1}__feature_{i+1}__histogram__{time_str}'  # = my_file+'_'+time_str  # TODO: HIGH: clarify on what a "session" is/means to the user.
                    save_graph_to_file(fig, file_name)
            plt.show()
    elif isinstance(labels, np.ndarray):
        R = np.linspace(0, 1, len(np.unique(labels)))
        colormap = plt.cm.get_cmap("Spectral")(R)
        for i in range(features.shape[0]):
            fig = plt.figure(facecolor='w', edgecolor='k')
            for k in range(len(np.unique(labels))):
                plt.subplot(len(np.unique(labels)), 1, k + 1)
                if i == 2 or i == 3 or i == 5 or i == 6:
                    plt.hist(features[i, labels == k],
                             bins=np.linspace(0, np.mean(features[i, :]) + 3 * np.std(features[i, :]), num=50),
                             range=(0, np.mean(features[i, :]) + 3 * np.std(features[i, :])),
                             color=colormap[k], density=True)
                    fig.suptitle(f"{feature_labels[i]} pixels")
                    plt.xlim(0, np.mean(features[i, :]) + 3 * np.std(features[i, :]))
                    if k < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                else:
                    plt.hist(features[i, labels == k],
                             bins=np.linspace(np.mean(features[i, :]) - 3 * np.std(features[i, :]),
                                              np.mean(features[i, :]) + 3 * np.std(features[i, :]), num=50),
                             range=(np.mean(features[i, :]) - 3 * np.std(features[i, :]),
                                    np.mean(features[i, :]) + 3 * np.std(features[i, :])),
                             color=colormap[k], density=True)
                    plt.xlim(np.mean(features[i, :]) - 3 * np.std(features[i, :]),
                             np.mean(features[i, :]) + 3 * np.std(features[i, :]))
                    fig.suptitle(f"{feature_labels[i]} pixels")
                    if k < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            if config.SAVE_GRAPHS_TO_FILE:
                file_name = f'feature_{i + 1}__histogram__{time_str}'  # my_file = f'feat{j + 1}_hist' ;fig.savefig(os.path.join(config.OUTPUT_PATH, f'{my_file}_{time_str}.svg'))
                save_graph_to_file(fig, file_name)
        plt.show()
    else: raise TypeError(f'invalid type detected for labels: {type(labels)}')


### PLOT CLASSES ######################################################################################################

def plot_classes_bsoidumap(data, assignments, **kwargs) -> object:
    """ Plot umap_embeddings for HDBSCAN assignments
    Function copied from the original bsoid_umapimplementation
    :param data:
    2D array, umap_embeddings
    :param assignments: 1D array, HDBSCAN assignments
    """
    # Parse kwargs
    s = kwargs.get('s', 0.5)
    marker = kwargs.get('marker', 'o')
    alpha = kwargs.get('alpha', 0.8)
    title = kwargs.get('title', 'UMAP enhanced clustering')

    # Prepare graph
    uk = list(np.unique(assignments))
    R = np.linspace(0, 1, len(uk))
    colormap = plt.cm.get_cmap("Spectral")(R)
    umap_x, umap_y, umap_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for g in np.unique(assignments):
        idx = np.where(np.array(assignments) == g)
        ax.scatter(umap_x[idx], umap_y[idx], umap_z[idx], c=colormap[g], label=g, s=s, marker=marker, alpha=alpha)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    plt.title(title)
    plt.legend(ncol=3)
    plt.show()
    return fig


def plot_GM_assignments_in_3d(data: np.ndarray, assignments, save_fig_to_file: bool, fig_file_prefix='train_assignments', show_now=True, show_later=False, **kwargs) -> object:
    """
    Plot trained TSNE for EM-GMM assignments
    This follows old implementation that only returned the figure
    :param data: 2D array, trained_tsne array (3 columns)
    :param assignments: 1D array, EM-GMM assignments
    :param save_fig_to_file:
    :param fig_file_prefix:
    :param show_later: use draw() instead of show()
    """
    ax, fig = plot_GM_assignments_in_3d_tuple(data=data, assignments=assignments, save_fig_to_file=save_fig_to_file, fig_file_prefix=fig_file_prefix, show_now=show_now, **kwargs)

    return fig


def plot_GM_assignments_in_3d_tuple(data: np.ndarray, assignments, save_fig_to_file: bool, fig_file_prefix='train_assignments', show_now=True, **kwargs) -> Tuple[object, object]:
    """
    Plot trained TSNE for EM-GMM assignments
    :param data: 2D array, trained_tsne array (3 columns)
    :param assignments: 1D array, EM-GMM assignments
    :param save_fig_to_file:
    :param fig_file_prefix:
    :param show_later: use draw() instead of show()
    """
    # TODO: find out why attaching the log entry/exit decorator kills the streamlit rotation app
    if not isinstance(data, np.ndarray):
        err = f'Expected `data` to be of type numpy.ndarray but instead found: {type(data)} (value = {data}).'
        logger.error(err)
        raise TypeError(err)
    # Parse kwargs
    s = kwargs.get('s', 0.5)
    marker = kwargs.get('marker', 'o')
    alpha = kwargs.get('alpha', 0.8)
    title = kwargs.get('title', 'Assignments by GMM')
    azim_elev = kwargs.get('azim_elev', (70, 135))
    # Plot graph
    unique_assignments = list(np.unique(assignments))
    R = np.linspace(0, 1, len(unique_assignments))
    colormap = plt.cm.get_cmap("Spectral")(R)
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Loop over assignments
    for i, g in enumerate(unique_assignments):
        # Select data for only assignment i
        idx = np.where(assignments == g)
        # Assign to colour and plot
        ax.scatter(tsne_x[idx], tsne_y[idx], tsne_z[idx], c=colormap[i], label=g, s=s, marker=marker, alpha=alpha)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    ax.view_init(*azim_elev)
    plt.title(title)
    plt.legend(ncol=3)
    # Draw now?
    if show_now:
        plt.show()
    else:
        plt.draw()
    # Save to graph to file?
    if save_fig_to_file:
        file_name = f'{fig_file_prefix}_{config.runtime_timestr}'  # fig.savefig(os.path.join(config.OUTPUT_PATH, f'{fig_file_prefix}_{time_str}.svg'))
        save_graph_to_file(fig, file_name)

    return fig, ax


# TODO: central difference b/w APP and UMAP is what is returned -- Tuple or solely a Fig
def plot_classes_bsoidapp(data, assignments) -> Tuple:
    """
    Plot umap_embeddings for HDBSCAN assignments.
    Function copied from the original bsoid_app implementation
    :param data: 2D array, umap_embeddings
    :param assignments: 1D array, HDBSCAN assignments
    """
    uk = list(np.unique(assignments))
    R = np.linspace(0, 1, len(uk))
    colormap = plt.cm.get_cmap("Spectral")(R)
    umap_x, umap_y, umap_z = data[:, 0], data[:, 1], data[:, 2]  # TODO: Q: if data is supposed to be 2-D, why are 3 indices referenced?
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for g in np.unique(assignments):
        idx = np.where(np.array(assignments) == g)
        ax.scatter(umap_x[idx], umap_y[idx], umap_z[idx], c=colormap[g], label=g, s=0.5, marker='o', alpha=0.8)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    # plt.title('UMAP enhanced clustering')
    plt.legend(ncol=3)
    # plt.show()
    return fig, plt


### PLOT ACCURACY ######################################################################################################

@config.deco__log_entry_exit(logger)
def plot_accuracy_MLP(scores, show_plot=config.PLOT_GRAPHS, save_fig_to_file=config.SAVE_GRAPHS_TO_FILE, fig_file_prefix='classifier_accuracy_score', **kwargs) -> Tuple:
    """
    TODO: low: purpose
    :param scores: 1D array, cross-validated accuracies for MLP classifier.
    :param show_plot: (bool) if True, plots figure (using plt.show()). In some cases, like the app, showing the
        plot is not desired.
    :param save_fig_to_file: (bool) if True, saves the figure to file.
    :param fig_file_prefix: (str) prefix for file when saving figure to file. Has no effect if
        `save_fig_to_file` is False.
    """
    # Parse kwargs
    facecolor = kwargs.get('facecolor', 'w')
    edgecolor = kwargs.get('edgecolor', 'k')
    s = kwargs.get('s', 40)
    c = kwargs.get('c', 'r')
    alpha = kwargs.get('alpha', 0.5)
    xlabel = kwargs.get('xlabel', 'MLP classifier')
    ylabel = kwargs.get('ylabel', 'Accuracy')
    # Plot as needed
    fig = plt.figure(facecolor=facecolor, edgecolor=edgecolor)
    fig.suptitle(f"Performance on {config.HOLDOUT_PERCENT * 100} % data")
    ax = fig.add_subplot(111)
    ax.boxplot(scores, notch=None)
    x = np.random.normal(1, 0.04, size=len(scores))
    plt.scatter(x, scores, s=s, c=c, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    time_str = config.runtime_timestr  # time_str = time.strftime("%Y%m%d_%H%M")
    # Plot and save as specified
    if show_plot:
        plt.show()
    if save_fig_to_file:
        fig_file_name = f'{fig_file_prefix}_{time_str}'
        save_graph_to_file(fig, fig_file_name)
    return fig, plt


@config.deco__log_entry_exit(logger)
def plot_accuracy_SVM(scores, save_fig_to_file=config.SAVE_GRAPHS_TO_FILE,
                      fig_file_prefix='classifier_accuracy_score', **kwargs):
    """
    This is the new interface for plotting accuracy for an SVM classifier.
    :param scores: (1D array) cross-validated accuracies for SVM classifier.
    :param save_fig_to_file:
    :param fig_file_prefix:
    :return: None
    """
    # logger.debug(f'{inspect.stack()[0][3]}(): entering function.')  # Temporary debugging measure
    # Parse kwargs
    facecolor = kwargs.get('facecolor', 'w')
    edgecolor = kwargs.get('edgecolor', 'k')
    s = kwargs.get('s', 40)
    c = kwargs.get('c', 'r')
    alpha = kwargs.get('alpha', 0.5)
    xlabel = kwargs.get('xlabel', 'SVM classifier')
    ylabel = kwargs.get('ylabel', 'Accuracy')
    #
    # TODO: decouple the fig saving and the plotting. Current state is due to legacy.
    time_str = config.runtime_timestr  # time_str = time.strftime("%Y%m%d_%H%M")
    fig = plt.figure(facecolor=facecolor, edgecolor=edgecolor)
    fig.suptitle(f"Performance on {config.HOLDOUT_PERCENT * 100} % data")
    ax = fig.add_subplot(111)
    ax.boxplot(scores, notch=None)
    x = np.random.normal(1, 0.04, size=len(scores))
    if len(x) != len(scores):
        logger.error(f'len(x) does not equal len(scores). '
                     f'If you see an error next, check the logs! x = {x} / scores = {scores}.')
    if isinstance(x, np.ndarray) and isinstance(scores, np.ndarray):
        logger.debug(f'{likelihoodprocessing.get_current_function()}: both inputs are arrays. '
                     f'x.shape = {x.shape} // scores.shape = {scores.shape}')
        if x.shape != scores.shape:
            logger.error(f'{inspect.stack()[0][3]}(): x = {x} // scores = {scores}')

    plt.scatter(x, scores, s=s, c=c, alpha=alpha)  # TODO: HIGH!!!! Why does this error occur?:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    if save_fig_to_file:
        fig_file_name = f'{fig_file_prefix}_{time_str}'
        save_graph_to_file(fig, fig_file_name)  # fig.savefig(os.path.join(config.OUTPUT_PATH, f'{fig_file_prefix}_{timestr}.svg'))


### PLOT FEATS ########################################################################################################

@config.deco__log_entry_exit(logger)
def plot_feats_bsoidpy(features, labels) -> None:
    """ *Legacy*
    :param features: list, features for multiple sessions
    :param labels: list, labels for multiple sessions
    """
    # NOTE: the order of the below feature labels is arbitrary and conserved from the original
    #   implementation where the ordering was hard-coded into feature-engineering process.
    feature_labels = ("Relative snout to forepaws placement",
                      "Relative snout to hind paws placement",
                      "Inter-forepaw distance",
                      "Body length",
                      "Body angle",
                      "Snout displacement",
                      "Tail-base displacement", )
    time_str = config.runtime_timestr  # time_str = time.strftime("%Y%m%d_%H%M")
    if isinstance(labels, list):
        for k in range(len(features)):
            labels_k = np.array(labels[k])
            feats_k = np.array(features[k])
            R = np.linspace(0, 1, len(np.unique(labels_k)))
            colormap = plt.cm.get_cmap("Spectral")(R)
            for j in range(feats_k.shape[0]):
                fig = plt.figure(facecolor='w', edgecolor='k')
                for i in range(len(np.unique(labels_k))):
                    plt.subplot(len(np.unique(labels_k)), 1, i + 1)
                    if j == 2 or j == 3 or j == 5 or j == 6:  # TODO: Q: KS: what is this equality trying to do?
                        plt.hist(feats_k[j, labels_k == i],
                                 bins=np.linspace(0, np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]), num=50),
                                 range=(0, np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :])),
                                 color=colormap[i],
                                 density=True)
                        fig.suptitle(f"{feature_labels[j]} pixels")
                        plt.xlim(0, np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]))
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    else:
                        plt.hist(feats_k[j, labels_k == i],
                                 bins=np.linspace(np.mean(feats_k[j, :]) - 3 * np.std(feats_k[j, :]),
                                                  np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]), num=50),
                                 range=(np.mean(feats_k[j, :]) - 3 * np.std(feats_k[j, :]),
                                        np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :])),
                                 color=colormap[i],
                                 density=True)
                        plt.xlim(np.mean(feats_k[j, :]) - 3 * np.std(feats_k[j, :]),
                                 np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]))
                        fig.suptitle(f"{feature_labels[j]} pixels")
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                if config.SAVE_GRAPHS_TO_FILE:
                    file_name = f'session_{k+1}__feature_{j+1}__histogram__{time_str}'  # = my_file+'_'+time_str  # TODO: HIGH: clarify on what a "session" is/means to the user.
                    save_graph_to_file(fig, file_name)  # fig.savefig(os.path.join(config.OUTPUT_PATH, my_file+'_'+time_str+'.svg'))
            plt.show()
    elif isinstance(labels, np.ndarray):
        R = np.linspace(0, 1, len(np.unique(labels)))
        colormap = plt.cm.get_cmap("Spectral")(R)
        for j in range(features.shape[0]):
            fig = plt.figure(facecolor='w', edgecolor='k')
            for i in range(len(np.unique(labels))):
                plt.subplot(len(np.unique(labels)), 1, i + 1)
                if j == 2 or j == 3 or j == 5 or j == 6:
                    plt.hist(features[j, labels == i],
                             bins=np.linspace(0, np.mean(features[j, :]) + 3 * np.std(features[j, :]), num=50),
                             range=(0, np.mean(features[j, :]) + 3 * np.std(features[j, :])),
                             color=colormap[i], density=True)
                    fig.suptitle(f"{feature_labels[j]} pixels")
                    plt.xlim(0, np.mean(features[j, :]) + 3 * np.std(features[j, :]))
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                else:
                    plt.hist(features[j, labels == i],
                             bins=np.linspace(np.mean(features[j, :]) - 3 * np.std(features[j, :]),
                                              np.mean(features[j, :]) + 3 * np.std(features[j, :]), num=50),
                             range=(np.mean(features[j, :]) - 3 * np.std(features[j, :]),
                                    np.mean(features[j, :]) + 3 * np.std(features[j, :])),
                             color=colormap[i], density=True)
                    plt.xlim(np.mean(features[j, :]) - 3 * np.std(features[j, :]),
                             np.mean(features[j, :]) + 3 * np.std(features[j, :]))
                    fig.suptitle(f"{feature_labels[j]} pixels")
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            if config.SAVE_GRAPHS_TO_FILE:
                file_name = f'feature_{j + 1}__histogram__{time_str}'  # my_file = f'feat{j + 1}_hist' ;fig.savefig(os.path.join(config.OUTPUT_PATH, f'{my_file}_{time_str}.svg'))
                save_graph_to_file(fig, file_name)
        plt.show()
    else: raise TypeError(f'invalid type detected for labels: {type(labels)}')


@config.deco__log_entry_exit(logger)
def plot_feats_bsoidUMAPAPP(features, labels: list) -> None:
    """
    :param features: list (or numpy array??), features for multiple sessions  # TODO
    :param labels: list, labels for multiple sessions
    """
    # NOTE: the order of the below feature labels is arbitrary and conserved from the original
    #   implementation where the ordering was hard-coded into feature-engineering process.
    feature_labels = ("Relative snout to forepaws placement",
                      "Relative snout to hind paws placement",
                      "Inter-forepaw distance",
                      "Body length",
                      "Body angle",
                      "Snout displacement",
                      "Tail-base displacement", )
    time_str = config.runtime_timestr  # time_str = time.strftime("%Y%m%d_%H%M")
    # labels_is_type_list = isinstance(labels, list)
    if isinstance(labels, list):
        for k in range(len(features)):
            labels_k = np.array(labels[k])
            feats_k = np.array(features[k])
            R = np.linspace(0, 1, len(np.unique(labels_k)))
            color_map = plt.cm.get_cmap("Spectral")(R)
            # feat_ls = ("Relative snout to forepaws placement", "Relative snout to hind paws placement",
            #            "Inter-forepaw distance", "Body length", "Body angle",
            #            "Snout displacement", "Tail-base displacement")
            for j in range(feats_k.shape[0]):
                fig = plt.figure(facecolor='w', edgecolor='k')
                for i in range(len(np.unique(labels_k))-1):
                    plt.subplot(len(np.unique(labels_k)), 1, i + 1)
                    if j == 2 or j == 3 or j == 5 or j == 6:
                        plt.hist(feats_k[j, labels_k == i],
                                 bins=np.linspace(0, np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]), num=50),
                                 range=(0, np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :])),
                                 color=color_map[i], density=True)
                        fig.suptitle(f"{feature_labels[j]} pixels")
                        plt.xlim(0, np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]))
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    else:
                        plt.hist(feats_k[j, labels_k == i],
                                 bins=np.linspace(np.mean(feats_k[j, :]) - 3 * np.std(feats_k[j, :]),
                                                  np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]), num=50),
                                 range=(np.mean(feats_k[j, :]) - 3 * np.std(feats_k[j, :]),
                                        np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :])),
                                 color=color_map[i], density=True)
                        plt.xlim(np.mean(feats_k[j, :]) - 3 * np.std(feats_k[j, :]),
                                 np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]))
                        fig.suptitle(f"{feature_labels[j]} pixels")
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                if config.SAVE_GRAPHS_TO_FILE:
                    fig_file_name = f'sess{k + 1}_feat{j + 1}_hist'
                    fig.savefig(os.path.join(config.OUTPUT_PATH, fig_file_name+'_'+time_str+'.svg'))
            plt.show()
    elif isinstance(labels, np.ndarray):
        R = np.linspace(0, 1, len(np.unique(labels)))
        color_map = plt.cm.get_cmap("Spectral")(R)
        # feat_ls = ("Relative snout to forepaws placement", "Relative snout to hind paws placement",
        #            "Inter-forepaw distance", "Body length", "Body angle",
        #            "Snout displacement", "Tail-base displacement")
        for j in range(features.shape[0]):
            fig = plt.figure(facecolor='w', edgecolor='k')
            for i in range(len(np.unique(labels))-1):
                plt.subplot(len(np.unique(labels)), 1, i + 1)
                if j == 2 or j == 3 or j == 5 or j == 6:
                    plt.hist(features[j, labels == i],
                             bins=np.linspace(0, np.mean(features[j, :]) + 3 * np.std(features[j, :]), num=50),
                             range=(0, np.mean(features[j, :]) + 3 * np.std(features[j, :])),
                             color=color_map[i], density=True)
                    fig.suptitle(f"{feature_labels[j]} pixels")
                    plt.xlim(0, np.mean(features[j, :]) + 3 * np.std(features[j, :]))
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                else:
                    plt.hist(features[j, labels == i],
                             bins=np.linspace(np.mean(features[j, :]) - 3 * np.std(features[j, :]),
                                              np.mean(features[j, :]) + 3 * np.std(features[j, :]), num=50),
                             range=(np.mean(features[j, :]) - 3 * np.std(features[j, :]),
                                    np.mean(features[j, :]) + 3 * np.std(features[j, :])),
                             color=color_map[i], density=True)
                    plt.xlim(np.mean(features[j, :]) - 3 * np.std(features[j, :]),
                             np.mean(features[j, :]) + 3 * np.std(features[j, :]))
                    fig.suptitle(f"{feature_labels[j]} pixels")
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            if config.SAVE_GRAPHS_TO_FILE:
                fig_file_name = f'feat{j + 1}_hist_{time_str}'
                save_graph_to_file(fig, fig_file_name)  # fig.savefig(os.path.join(config.OUTPUT_PATH, f'{my_file}_{time_str}.svg'))

        plt.show()
    else:
        type_err = f''
        logger.error(type_err)
        raise TypeError(type_err)


@config.deco__log_entry_exit(logger)
def plot_feats_bsoidvoc(features, labels: list) -> None:
    """
    :param features: list, features for multiple sessions
    :param labels: list, labels for multiple sessions
    """
    time_str = config.runtime_timestr  # timestr = time.strftime("%Y%m%d_%H%M")
    feature_labels = ("Distance between points 1 & 5", "Distance between points 1 & 8",
                      "Angle change between points 1 & 2", "Angle change between points 1 & 4",
                      "Point 3 displacement", "Point 7 displacement")

    if isinstance(labels, list):
        for k in range(len(features)):
            labels_k = np.array(labels[k])
            feats_k = np.array(features[k])
            R = np.linspace(0, 1, len(np.unique(labels_k)))
            color = plt.cm.get_cmap("Spectral")(R)
            for j in range(feats_k.shape[0]):  # iterating over number of rows
                fig = plt.figure(facecolor='w', edgecolor='k')
                for i in range(0, len(np.unique(labels_k))):
                    plt.subplot(len(np.unique(labels_k)), 1, i + 1)
                    if j == 0 or j == 1 or j == 4 or j == 5:
                        plt.hist(feats_k[j, labels_k == i],
                                 bins=np.linspace(0, np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]), num=50),
                                 range=(0, np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :])),
                                 color=color[i], density=True)
                        fig.suptitle(f"{feature_labels[j]} pixels")
                        plt.xlim(0, np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]))
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    else:
                        plt.hist(feats_k[j, labels_k == i],
                                 bins=np.linspace(np.mean(feats_k[j, :]) - 3 * np.std(feats_k[j, :]),
                                                  np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]), num=50),
                                 range=(np.mean(feats_k[j, :]) - 3 * np.std(feats_k[j, :]),
                                        np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :])),
                                 color=color[i], density=True)
                        plt.xlim(np.mean(feats_k[j, :]) - 3 * np.std(feats_k[j, :]),
                                 np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]))
                        fig.suptitle(f"{feature_labels[j]} pixels")
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                if config.SAVE_GRAPHS_TO_FILE:
                    file_name = f'session_{k + 1}__feature_{j+1}__histogram_{time_str}'  # TODO: HIGH: clarify on what a "session" is/means to the user.
                    save_graph_to_file(fig, file_name)  # fig.savefig(os.path.join(config.OUTPUT_PATH, str.join('', (my_file, '_'+timestr, '.svg'))))
            plt.show()
    elif isinstance(labels, np.ndarray):
        R = np.linspace(0, 1, len(np.unique(labels)))
        color_map = plt.cm.get_cmap("Spectral")(R)
        # Iterating over number of rows
        for j in range(features.shape[0]):
            fig = plt.figure(facecolor='w', edgecolor='k')
            # Iterating over unique labels (by index)
            for i in range(len(np.unique(labels))):
                plt.subplot(len(np.unique(labels)), 1, i + 1)
                if j == 0 or j == 1 or j == 4 or j == 5:
                    plt.hist(features[j, labels == i],
                             bins=np.linspace(0, np.mean(features[j, :]) + 3 * np.std(features[j, :]), num=50),
                             range=(0, np.mean(features[j, :]) + 3 * np.std(features[j, :])),
                             color=color_map[i], density=True)
                    fig.suptitle(f"{feature_labels[j]} pixels")
                    plt.xlim(0, np.mean(features[j, :]) + 3 * np.std(features[j, :]))
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                else:
                    plt.hist(features[j, labels == i],
                             bins=np.linspace(np.mean(features[j, :]) - 3 * np.std(features[j, :]),
                                              np.mean(features[j, :]) + 3 * np.std(features[j, :]), num=50),
                             range=(np.mean(features[j, :]) - 3 * np.std(features[j, :]),
                                    np.mean(features[j, :]) + 3 * np.std(features[j, :])),
                             color=color_map[i], density=True)
                    plt.xlim(np.mean(features[j, :]) - 3 * np.std(features[j, :]),
                             np.mean(features[j, :]) + 3 * np.std(features[j, :]))
                    fig.suptitle(f"{feature_labels[j]} pixels")
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            if config.SAVE_GRAPHS_TO_FILE:
                fig_file_name = f'feature_{j + 1}__histogram_{time_str}'  # my_file = f'feat{j + 1}_hist'; fig.savefig(os.path.join(config.OUTPUT_PATH, str.join('', (my_file, '_'+timestr, '.svg'))))
                save_graph_to_file(fig, fig_file_name)
        plt.show()
    else: raise TypeError(f'unexpected type for labels: {type(labels)}')


def save_graph_to_file(figure: object, file_title: str, file_type_extension=config.DEFAULT_SAVED_GRAPH_FILE_FORMAT,
                       alternate_save_path: str = None) -> None:
    """
    :param figure: (object) a figure object. Must have a savefig() function
    :param file_title: (str)
    :param alternate_save_path:
    :param file_type_extension: (str)
    :return:
    """
    if alternate_save_path and not os.path.isdir(alternate_save_path):
        path_not_exists_err = f'Alternate save file path does not exist. Cannot save image to path: {alternate_save_path}.'
        logger.error(path_not_exists_err)
        raise ValueError(path_not_exists_err)
    if not hasattr(figure, 'savefig'):
        cannot_save_input_figure_error = f'Figure is not savable with current interface. ' \
                                         f'Requires ability to use .savefig() method. ' \
                                         f'repr(figure) = {repr(figure)}.'
        logger.error(cannot_save_input_figure_error)
        raise AttributeError(cannot_save_input_figure_error)
    # After type checking: save fig to file
    figure.savefig(os.path.join(config.GRAPH_OUTPUT_PATH, f'{file_title}.{file_type_extension}'))
    return


### LEGACY FUNCTIONS ###################################################################################################

def plot_tmat(transition_matrix: np.ndarray, fps: int, save_fig_to_file=True, fig_file_prefix='transition_matrix'):
    """Original implementation as plot_tmat()"""
    replacement_func = plot_transition_matrix
    warning_msg = f'This function, {inspect.stack()[0][3]}, will be deprecated soon. Instead, ' \
                  f'use: {replacement_func.__qualname__} with args: ' \
                  f'transition_matrix, fps, save_fig_to_file=True, fig_file_prefix="transition_matrix".'
    logger.warning(warning_msg)
    return replacement_func(transition_matrix, fps, save_fig_to_file, fig_file_prefix)

def plot_classes_bsoidvoc(data, assignments, save_fig_to_file=config.SAVE_GRAPHS_TO_FILE) -> None:
    replacement_func = plot_GM_assignments_in_3d
    warning = f'This function, {inspect.stack()[0][3]}, will be deprecated and instead replaced with: ' \
              f'{replacement_func.__qualname__}. Find caller, {inspect.stack()[1][3]}, and replace use.'
    logger.warning(warning)
    return replacement_func(data, assignments, save_fig_to_file=save_fig_to_file)
def plot_accuracy_bsoidvoc(scores) -> None:  # (MLP)
    # fig = plt.figure(facecolor='w', edgecolor='k')
    # fig.suptitle(f"Performance on {HLDOUT * 100} % data")
    # ax = fig.add_subplot(111)
    # ax.boxplot(scores, notch=None)
    # x = np.random.normal(1, 0.04, size=len(scores))
    # plt.scatter(x, scores, s=40, c='r', alpha=0.5)
    # ax.set_xlabel('MLP classifier')
    # ax.set_ylabel('Accuracy')
    # plt.show()
    # timestr = time.strftime("_%Y%m%d_%H%M")
    # my_file = 'clf_scores'
    # fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
    replacement_func = plot_accuracy_MLP
    logger.error(f'DEPRECATION WARNING: {inspect.stack()[0][3]} // Instead, '
                 f'use replacement function: {replacement_func.__qualname__}')
    replacement_func(scores, show_plot=True, save_fig_to_file=True, facecolor='w', edgecolor='k')
def plot_accuracy_bsoidapp(scores) -> Tuple:
    # fig = plt.figure(facecolor='w', edgecolor='k')
    # fig.suptitle("Performance on {} % data".format(HLDOUT * 100))
    # ax = fig.add_subplot(111)
    # ax.boxplot(scores, notch=None)
    # x = np.random.normal(1, 0.04, size=len(scores))
    # plt.scatter(x, scores, s=40, c='r', alpha=0.5)
    # ax.set_xlabel('MLP classifier')
    # ax.set_ylabel('Accuracy')
    # # plt.show()
    # return fig, plt
    return plot_accuracy_MLP(scores, show_plot=False, save_fig_to_file=False, facecolor='w', edgecolor='k')
def plot_accuracy_bsoidumap(scores) -> None:
    # fig = plt.figure(facecolor='w', edgecolor='k')
    # fig.suptitle(f"Performance on {HLDOUT * 100} % data")
    # ax = fig.add_subplot(111)
    # ax.boxplot(scores, notch=None)
    # x = np.random.normal(1, 0.04, size=len(scores))
    # plt.scatter(x, scores, s=40, c='r', alpha=0.5)
    # ax.set_xlabel('MLP classifier')
    # ax.set_ylabel('Accuracy')
    # plt.show()
    # my_file = 'clf_scores'
    # fig.savefig(os.path.join(OUTPUT_PATH, f'{fig_file_prefix}_{timestr}.svg'))
    plot_accuracy_MLP(scores, show_plot=True, save_fig_to_file=True)
@config.deco__log_entry_exit(logger)
def plot_accuracy_bsoidpy(scores) -> None:
    """ ** DEPRECATION WARNING ** """
    # fig = plt.figure(facecolor='w', edgecolor='k')
    # fig.suptitle(f"Performance on {HLDOUT * 100} % data")
    # ax = fig.add_subplot(111)
    # ax.boxplot(scores, notch=None)
    # x = np.random.normal(1, 0.04, size=len(scores))
    # plt.scatter(x, scores, s=40, c='r', alpha=0.5)
    # ax.set_xlabel('SVM classifier')
    # ax.set_ylabel('Accuracy')
    # plt.show()
    # if save_fig_to_file:
    #     fig.savefig(os.path.join(OUTPUT_PATH, fig_file_prefix+'_'+timestr+'.svg'))
    replacement_func = plot_accuracy_SVM
    logger.warning(f'This function, {inspect.stack()[0][3]}, will be deprecated.'
                   f'Instead use: {replacement_func.__qualname__}')
    return replacement_func(scores, True, 'clf_scores')
