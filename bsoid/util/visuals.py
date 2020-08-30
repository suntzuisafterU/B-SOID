"""
Visualization functions and saving plots.
"""
# TODO: med: Currently, the file naming pattern that uses `timestr` does not take into account when time pases, so
#   `timestr` will only have the time recorded at program start but not the running time during runtime.
#   Whether or not we need to ensure that the output of file timestamps needs to be exactly current should be discussed.

# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from mpl_toolkits.mplot3d import Axes3D  # <-- This seemingly unused import is NECESSARY for 3d plotting. Keep as is.
from typing import Tuple
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from bsoid import config

logger = config.create_file_specific_logger(__name__)

#

matplotlib_axes_logger.setLevel('ERROR')


TM = NotImplementedError('TODO: HIGH: The source of TM has not been determined. Find and fix as such.')  # TODO: HIGH

# def plot_classes():raise NotImplemented('disambiguate functions')
# def plot_accuracy(): raise NotImplemented('disambiguate functions')
# def plot_feats(): raise NotImplemented('disambiguate functions')

#######################################################################################################################
def plot_tsne_in_3d(data):
    """
    Plot trained tsne
    :param data: trained_tsne TODO: expand desc. and include type
    """
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_x, tsne_y, tsne_z, s=1, marker='o', alpha=0.8)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    ax.view_init(70, 135)
    plt.title('Embedding of the training set by t-SNE')
    plt.show()
#######################################################################################################################
def plot_duration_histogram(lengths, grp, save_fig_to_file: bool = True, fig_file_prefix='duration_hist_100msbins'):
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
    for i in range(0, len(np.unique(grp))):
        fig.suptitle(f"Duration histogram of {len(np.unique(TM))} behaviors")
        x = lengths[np.where(grp == i)]
        ax.hist(x, density=True, color=colormap[i], alpha=0.3, label=f'Group {i}')
    plt.legend(loc='Upper right')
    plt.show()
    timestr = time.strftime("%Y%m%d_%H%M")  # TODO: low: move this to config file
    if save_fig_to_file:
        fig.savefig(os.path.join(config.OUTPUT_PATH, f'{fig_file_prefix}_{timestr}.svg'))
    return fig
#######################################################################################################################
def plot_tmat(transition_matrix: np.ndarray, fps: int, save_fig_to_file=True, fig_file_prefix='transition_matrix'):
    """Original implementation as plot_tmat()"""
    replacement_func = plot_transition_matrix
    warning = f'This function, plot_tmat(), will be deprecated soon. Instead, use: ' \
              f'{replacement_func.__qualname__}.'
    logger.warning(warning)
    # warnings.warn(warning)
    return replacement_func(transition_matrix, fps, save_fig_to_file, fig_file_prefix)

def plot_transition_matrix(transition_matrix, fps, save_fig_to_file, fig_file_prefix):
    """
    New interface for original function named plot_tmat()
        TODO: low: purpose
    :param transition_matrix: object, transition matrix data frame  TODO: Q: what is it transitioning from?
    :param fps: scalar, camera frame-rate
    :param save_fig_to_file: bool,
    :param fig_file_prefix: str,
    :return TODO: low
    """
    fig = plt.figure()
    fig.suptitle(f"Transition matrix of {transition_matrix.shape[0]} behaviors")
    sn.heatmap(transition_matrix, annot=True)
    plt.xlabel("Next frame behavior")
    plt.ylabel("Current frame behavior")
    plt.show()
    if save_fig_to_file:
        fig.savefig(os.path.join(config.OUTPUT_PATH, f'{fig_file_prefix}{fps}.svg'))
    return fig

#######################################################################################################################
def plot_classes_EMGMM_assignments(data, assignments, save_fig_to_file: bool, fig_file_prefix='train_assignments'):
    """ Plot trained tsne for EM-GMM assignments
    :param data: 2D array, trained_tsne
    :param assignments: 1D array, EM-GMM assignments
    """
    time_str = time.strftime("%Y%m%d_%H%M")
    uk = list(np.unique(assignments))
    R = np.linspace(0, 1, len(uk))
    colormap = plt.cm.get_cmap("Spectral")(R)
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for g in np.unique(assignments):
        idx = np.where(np.array(assignments) == g)
        ax.scatter(tsne_x[idx], tsne_y[idx], tsne_z[idx], c=colormap[g], label=g, s=0.5, marker='o', alpha=0.8)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    ax.view_init(70, 135)
    plt.title('Assignments by GMM')
    plt.legend(ncol=3)
    plt.show()
    # my_file = 'train_assignments'
    if save_fig_to_file:
        fig.savefig(os.path.join(config.OUTPUT_PATH, f'{fig_file_prefix}_{time_str}.svg'))
# def plot_classes_app(data, assignments):
#     """ Plot umap_embeddings for HDBSCAN assignments
#     :param data: 2D array, umap_embeddings
#     :param assignments: 1D array, HDBSCAN assignments
#     """
#     uk = list(np.unique(assignments))
#     R = np.linspace(0, 1, len(uk))
#     cmap = plt.cm.get_cmap("Spectral")(R)
#     umap_x, umap_y, umap_z = data[:, 0], data[:, 1], data[:, 2]
#     # umap_x, umap_y= data[:, 0], data[:, 1]
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     # ax = fig.add_subplot(111)
#     for g in np.unique(assignments):
#         idx = np.where(np.array(assignments) == g)
#         ax.scatter(umap_x[idx], umap_y[idx], umap_z[idx], c=cmap[g],
#                    label=g, s=0.5, marker='o', alpha=0.8)
#         # ax.scatter(umap_x[idx], umap_y[idx], c=cmap[g],
#         #            label=g, s=0.5, marker='o', alpha=0.8)
#     ax.set_xlabel('Dim. 1')
#     ax.set_ylabel('Dim. 2')
#     ax.set_zlabel('Dim. 3')
#     # plt.title('UMAP enhanced clustering')
#     plt.legend(ncol=3)
#     # plt.show()
#
#     return fig, plt

def plot_classes_bsoidvoc(data, assignments) -> None:
    return plot_classes_EMGMM_assignments(data, assignments, save_fig_to_file=True)
# def plot_classes_bsoidpy(data, assignments) -> None:
#     replacement_func = plot_classes_EMGMM_assignments
#     warn = f'This function will be deprecated. Instead, replace its use with: {replacement_func.__qualname__}'
#     # warnings.warn(warn)
#     logger.warning(warn)
#     return replacement_func(data, assignments, save_fig_to_file=True)

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
def plot_classes_bsoidumap(data, assignments) -> object:
    """ Plot umap_embeddings for HDBSCAN assignments
    Function copied from the original bsoid_umapimplementation
    :param data: 2D array, umap_embeddings
    :param assignments: 1D array, HDBSCAN assignments
    """
    uk = list(np.unique(assignments))
    R = np.linspace(0, 1, len(uk))
    colormap = plt.cm.get_cmap("Spectral")(R)
    umap_x, umap_y, umap_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for g in np.unique(assignments):
        idx = np.where(np.array(assignments) == g)
        ax.scatter(umap_x[idx], umap_y[idx], umap_z[idx], c=colormap[g], label=g, s=0.5, marker='o', alpha=0.8)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    plt.title('UMAP enhanced clustering')
    plt.legend(ncol=3)
    plt.show()
    return fig

#######################################################################################################################
def plot_accuracy_MLP(scores, show_plot: bool, save_fig_to_file: bool, fig_file_prefix='clf_scores') -> Tuple:
    """
    TODO: low: purpose
    :param scores: 1D array, cross-validated accuracies for MLP classifier.
    :param show_plot: (bool) if True, plots figure (using plt.show()). In some cases, like the app, showing the
        plot is not desired.
    :param save_fig_to_file: (bool) if True, saves the figure to file.
    :param fig_file_prefix: (str) prefix for file when saving figure to file. Has no effect if
        `save_fig_to_file` is False.
    """
    fig = plt.figure(facecolor='w', edgecolor='k')
    fig.suptitle(f"Performance on {config.holdout_percent * 100} % data")
    ax = fig.add_subplot(111)
    ax.boxplot(scores, notch=None)
    x = np.random.normal(1, 0.04, size=len(scores))
    plt.scatter(x, scores, s=40, c='r', alpha=0.5)
    ax.set_xlabel('MLP classifier')
    ax.set_ylabel('Accuracy')
    if show_plot:
        plt.show()
    if save_fig_to_file:
        timestr = time.strftime("%Y%m%d_%H%M")
        fig.savefig(os.path.join(config.OUTPUT_PATH, f'{fig_file_prefix}_{timestr}.svg'))
    return fig, plt
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
    plot_accuracy_MLP(scores, show_plot=True, save_fig_to_file=True)
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
    return plot_accuracy_MLP(scores, show_plot=False, save_fig_to_file=False)
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
def plot_accuracy_bsoidpy(scores) -> None:
    """ *** NOTE: Do not alter or remove this function. It's a trace
             back to the original bsoid_py function. First ensure that the new function works, then jettison garbage"""
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
    logger.warning(f'This function, plot_accuracy_bsoidpy(), will be deprecated.'
                   f'Instead use: {replacement_func.__qualname__}')
    return replacement_func(scores, True, 'clf_scores')
def plot_accuracy_SVM(scores, save_fig_to_file=True, fig_file_prefix='clf_scores'):
    """
    This is the new interface for plotting accuracy for an SVM classifier.
    :param scores: (1D array) cross-validated accuracies for SVM classifier.
    :param save_fig_to_file:
    :param fig_file_prefix:
    :return: None
    """
    timestr = time.strftime("%Y%m%d_%H%M")
    fig = plt.figure(facecolor='w', edgecolor='k')
    fig.suptitle(f"Performance on {config.holdout_percent * 100} % data")
    ax = fig.add_subplot(111)
    ax.boxplot(scores, notch=None)
    x = np.random.normal(1, 0.04, size=len(scores))
    plt.scatter(x, scores, s=40, c='r', alpha=0.5)
    ax.set_xlabel('SVM classifier')
    ax.set_ylabel('Accuracy')
    plt.show()
    if save_fig_to_file:
        fig.savefig(os.path.join(config.OUTPUT_PATH, f'{fig_file_prefix}_{timestr}.svg'))

#######################################################################################################################
#######################################################################################################################
def plot_feats_bsoidUMAPAPP(feats: list, labels: list) -> None:
    """
    :param feats: list, features for multiple sessions
    :param labels: list, labels for multiple sessions
    """
    time_str = time.strftime("%Y%m%d_%H%M")
    result = isinstance(labels, list)
    if result:
        for k in range(0, len(feats)):
            labels_k = np.array(labels[k])
            feats_k = np.array(feats[k])
            R = np.linspace(0, 1, len(np.unique(labels_k)))
            color = plt.cm.get_cmap("Spectral")(R)
            feat_ls = ("Relative snout to forepaws placement", "Relative snout to hind paws placement",
                       "Inter-forepaw distance", "Body length", "Body angle",
                       "Snout displacement", "Tail-base displacement")
            for j in range(0, feats_k.shape[0]):
                fig = plt.figure(facecolor='w', edgecolor='k')
                for i in range(0, len(np.unique(labels_k))-1):
                    plt.subplot(len(np.unique(labels_k)), 1, i + 1)
                    if j == 2 or j == 3 or j == 5 or j == 6:
                        plt.hist(feats_k[j, labels_k == i],
                                 bins=np.linspace(0, np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]), num=50),
                                 range=(0, np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :])),
                                 color=color[i], density=True)
                        fig.suptitle(f"{feat_ls[j]} pixels")
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
                        fig.suptitle(f"{feat_ls[j]} pixels")
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                my_file = f'sess{k + 1}_feat{j + 1}_hist'
                fig.savefig(os.path.join(config.OUTPUT_PATH, my_file+'_'+time_str+'.svg'))
            plt.show()
    else:
        R = np.linspace(0, 1, len(np.unique(labels)))
        color = plt.cm.get_cmap("Spectral")(R)
        feat_ls = ("Relative snout to forepaws placement", "Relative snout to hind paws placement",
                   "Inter-forepaw distance", "Body length", "Body angle",
                   "Snout displacement", "Tail-base displacement")
        for j in range(0, feats.shape[0]):
            fig = plt.figure(facecolor='w', edgecolor='k')
            for i in range(0, len(np.unique(labels))-1):
                plt.subplot(len(np.unique(labels)), 1, i + 1)
                if j == 2 or j == 3 or j == 5 or j == 6:
                    plt.hist(feats[j, labels == i],
                             bins=np.linspace(0, np.mean(feats[j, :]) + 3 * np.std(feats[j, :]), num=50),
                             range=(0, np.mean(feats[j, :]) + 3 * np.std(feats[j, :])),
                             color=color[i], density=True)
                    fig.suptitle(f"{feat_ls[j]} pixels")
                    plt.xlim(0, np.mean(feats[j, :]) + 3 * np.std(feats[j, :]))
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                else:
                    plt.hist(feats[j, labels == i],
                             bins=np.linspace(np.mean(feats[j, :]) - 3 * np.std(feats[j, :]),
                                              np.mean(feats[j, :]) + 3 * np.std(feats[j, :]), num=50),
                             range=(np.mean(feats[j, :]) - 3 * np.std(feats[j, :]),
                                    np.mean(feats[j, :]) + 3 * np.std(feats[j, :])),
                             color=color[i], density=True)
                    plt.xlim(np.mean(feats[j, :]) - 3 * np.std(feats[j, :]),
                             np.mean(feats[j, :]) + 3 * np.std(feats[j, :]))
                    fig.suptitle(f"{feat_ls[j]} pixels")
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            my_file = f'feat{j+1}_hist'
            fig.savefig(os.path.join(config.OUTPUT_PATH, f'{my_file}_{time_str}.svg'))
        plt.show()
def plot_feats_bsoidpy(feats, labels) -> None:
    """
    :param feats: list, features for multiple sessions
    :param labels: list, labels for multiple sessions
    """
    timestr = time.strftime("%Y%m%d_%H%M")
    feat_ls = ("Relative snout to forepaws placement",
               "Relative snout to hind paws placement",
               "Inter-forepaw distance",
               "Body length",
               "Body angle",
               "Snout displacement",
               "Tail-base displacement")
    is_labels_list = isinstance(labels, list)
    if is_labels_list:
        for k in range(len(feats)):
            labels_k = np.array(labels[k])
            feats_k = np.array(feats[k])
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
                        fig.suptitle(f"{feat_ls[j]} pixels")
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
                        fig.suptitle(f"{feat_ls[j]} pixels")
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                my_file = f'sess{k+1}_feat{j+1}_hist'
                timestr = time.strftime("%Y%m%d_%H%M")
                fig.savefig(os.path.join(config.OUTPUT_PATH, my_file+'_'+timestr+'.svg'))
            plt.show()
    else:
        R = np.linspace(0, 1, len(np.unique(labels)))
        colormap = plt.cm.get_cmap("Spectral")(R)
        for j in range(feats.shape[0]):
            fig = plt.figure(facecolor='w', edgecolor='k')
            for i in range(len(np.unique(labels))):
                plt.subplot(len(np.unique(labels)), 1, i + 1)
                if j == 2 or j == 3 or j == 5 or j == 6:
                    plt.hist(feats[j, labels == i],
                             bins=np.linspace(0, np.mean(feats[j, :]) + 3 * np.std(feats[j, :]), num=50),
                             range=(0, np.mean(feats[j, :]) + 3 * np.std(feats[j, :])),
                             color=colormap[i], density=True)
                    fig.suptitle(f"{feat_ls[j]} pixels")
                    plt.xlim(0, np.mean(feats[j, :]) + 3 * np.std(feats[j, :]))
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                else:
                    plt.hist(feats[j, labels == i],
                             bins=np.linspace(np.mean(feats[j, :]) - 3 * np.std(feats[j, :]),
                                              np.mean(feats[j, :]) + 3 * np.std(feats[j, :]), num=50),
                             range=(np.mean(feats[j, :]) - 3 * np.std(feats[j, :]),
                                    np.mean(feats[j, :]) + 3 * np.std(feats[j, :])),
                             color=colormap[i], density=True)
                    plt.xlim(np.mean(feats[j, :]) - 3 * np.std(feats[j, :]),
                             np.mean(feats[j, :]) + 3 * np.std(feats[j, :]))
                    fig.suptitle(f"{feat_ls[j]} pixels")
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            my_file = f'feat{j + 1}_hist'
            fig.savefig(os.path.join(config.OUTPUT_PATH, f'{my_file}_{timestr}.svg'))
        plt.show()
def plot_feats_bsoidvoc(feats: list, labels: list) -> None:
    """
    :param feats: list, features for multiple sessions
    :param labels: list, labels for multiple sessions
    """
    timestr = time.strftime("%Y%m%d_%H%M")
    is_labels_type_list = isinstance(labels, list)
    if is_labels_type_list:
        for k in range(0, len(feats)):
            labels_k = np.array(labels[k])
            feats_k = np.array(feats[k])
            R = np.linspace(0, 1, len(np.unique(labels_k)))
            color = plt.cm.get_cmap("Spectral")(R)
            feat_ls = ("Distance between points 1 & 5", "Distance between points 1 & 8",
                       "Angle change between points 1 & 2", "Angle change between points 1 & 4",
                       "Point 3 displacement", "Point 7 displacement")
            for j in range(0, feats_k.shape[0]):
                fig = plt.figure(facecolor='w', edgecolor='k')
                for i in range(0, len(np.unique(labels_k))):
                    plt.subplot(len(np.unique(labels_k)), 1, i + 1)
                    if j == 0 or j == 1 or j == 4 or j == 5:
                        plt.hist(feats_k[j, labels_k == i],
                                 bins=np.linspace(0, np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :]), num=50),
                                 range=(0, np.mean(feats_k[j, :]) + 3 * np.std(feats_k[j, :])),
                                 color=color[i], density=True)
                        fig.suptitle(f"{feat_ls[j]} pixels")
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
                        fig.suptitle(f"{feat_ls[j]} pixels")
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                my_file = f'sess{k + 1}_feat{j + 1}_hist'
                fig.savefig(os.path.join(config.OUTPUT_PATH, str.join('', (my_file, '_'+timestr, '.svg'))))
            plt.show()
    else:
        R = np.linspace(0, 1, len(np.unique(labels)))
        color = plt.cm.get_cmap("Spectral")(R)
        feat_ls = ("Distance between points 1 & 5", "Distance between points 1 & 8",
                   "Angle change between points 1 & 2", "Angle change between points 1 & 4",
                   "Point 3 displacement", "Point 7 displacement")
        for j in range(0, feats.shape[0]):
            fig = plt.figure(facecolor='w', edgecolor='k')
            for i in range(0, len(np.unique(labels))):
                plt.subplot(len(np.unique(labels)), 1, i + 1)
                if j == 0 or j == 1 or j == 4 or j == 5:
                    plt.hist(feats[j, labels == i],
                             bins=np.linspace(0, np.mean(feats[j, :]) + 3 * np.std(feats[j, :]), num=50),
                             range=(0, np.mean(feats[j, :]) + 3 * np.std(feats[j, :])),
                             color=color[i], density=True)
                    fig.suptitle(f"{feat_ls[j]} pixels")
                    plt.xlim(0, np.mean(feats[j, :]) + 3 * np.std(feats[j, :]))
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                else:
                    plt.hist(feats[j, labels == i],
                             bins=np.linspace(np.mean(feats[j, :]) - 3 * np.std(feats[j, :]),
                                              np.mean(feats[j, :]) + 3 * np.std(feats[j, :]), num=50),
                             range=(np.mean(feats[j, :]) - 3 * np.std(feats[j, :]),
                                    np.mean(feats[j, :]) + 3 * np.std(feats[j, :])),
                             color=color[i], density=True)
                    plt.xlim(np.mean(feats[j, :]) - 3 * np.std(feats[j, :]),
                             np.mean(feats[j, :]) + 3 * np.std(feats[j, :]))
                    fig.suptitle(f"{feat_ls[j]} pixels")
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            my_file = f'feat{j + 1}_hist'
            fig.savefig(os.path.join(config.OUTPUT_PATH, str.join('', (my_file, '_'+timestr, '.svg'))))
        plt.show()
#######################################################################################################################

# Legacy functions

