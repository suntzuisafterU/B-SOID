#bsoid app
"""
Visualization functions and saving plots.
"""
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import numpy as np
import pandas as pd
import seaborn as sn
from bsoid.config.LOCAL_CONFIG import BASE_PATH, OUTPUT_PATH
from bsoid.config.GLOBAL_CONFIG import HLDOUT, SVM_PARAMS
matplotlib_axes_logger.setLevel('ERROR')
timestr = time.strftime("_%Y%m%d_%H%M")


def plot_classes(data, assignments):
    raise NotImplemented('disambiguate functions')
def plot_accuracy(): raise NotImplemented()


def plot_tsne3d(data):
    """
    Plot trained_tsne
    :param data: trained_tsne TODO: expand and include type
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


def plot_classes_bsoidapp(data, assignments):
    """ Plot umap_embeddings for HDBSCAN assignments
    :param data: 2D array, umap_embeddings
    :param assignments: 1D array, HDBSCAN assignments
    """
    uk = list(np.unique(assignments))
    R = np.linspace(0, 1, len(uk))
    cmap = plt.cm.get_cmap("Spectral")(R)
    umap_x, umap_y, umap_z = data[:, 0], data[:, 1], data[:, 2]
    # umap_x, umap_y= data[:, 0], data[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)
    for g in np.unique(assignments):
        idx = np.where(np.array(assignments) == g)
        ax.scatter(umap_x[idx], umap_y[idx], umap_z[idx], c=cmap[g],
                   label=g, s=0.5, marker='o', alpha=0.8)
        # ax.scatter(umap_x[idx], umap_y[idx], c=cmap[g],
        #            label=g, s=0.5, marker='o', alpha=0.8)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    # plt.title('UMAP enhanced clustering')
    plt.legend(ncol=3)
    # plt.show()
    return fig, plt
def plot_classes_bsoidpy(data, assignments) -> None:
    """ Plot trained_tsne for EM-GMM assignments
    :param data: 2D array, trained_tsne
    :param assignments: 1D array, EM-GMM assignments
    """
    uk = list(np.unique(assignments))
    R = np.linspace(0, 1, len(uk))
    cmap = plt.cm.get_cmap("Spectral")(R)
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for g in np.unique(assignments):
        idx = np.where(np.array(assignments) == g)
        ax.scatter(tsne_x[idx], tsne_y[idx], tsne_z[idx], c=cmap[g],
                   label=g, s=0.5, marker='o', alpha=0.8)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    ax.view_init(70, 135)
    plt.title('Assignments by GMM')
    plt.legend(ncol=3)
    plt.show()
    timestr = time.strftime("_%Y%m%d_%H%M")
    my_file = 'train_assignments'
    fig.savefig(os.path.join(OUTPUT_PATH, my_file+timestr+'.svg'))
def plot_classes_bsoidumap(data, assignments) -> object:
    """ Plot umap_embeddings for HDBSCAN assignments
    :param data: 2D array, umap_embeddings
    :param assignments: 1D array, HDBSCAN assignments
    """
    uk = list(np.unique(assignments))
    R = np.linspace(0, 1, len(uk))
    cmap = plt.cm.get_cmap("Spectral")(R)
    umap_x, umap_y, umap_z = data[:, 0], data[:, 1], data[:, 2]
    # umap_x, umap_y= data[:, 0], data[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)
    for g in np.unique(assignments):
        idx = np.where(np.array(assignments) == g)
        ax.scatter(umap_x[idx], umap_y[idx], umap_z[idx], c=cmap[g],
                   label=g, s=0.5, marker='o', alpha=0.8)
        # ax.scatter(umap_x[idx], umap_y[idx], c=cmap[g],
        #            label=g, s=0.5, marker='o', alpha=0.8)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    plt.title('UMAP enhanced clustering')
    plt.legend(ncol=3)
    plt.show()
    return fig
def plot_classes_bsoidvoc(data, assignments) -> None:
    """ Plot trained_tsne for EM-GMM assignments
    :param data: 2D array, trained_tsne
    :param assignments: 1D array, EM-GMM assignments
    """
    uk = list(np.unique(assignments))
    R = np.linspace(0, 1, len(uk))
    cmap = plt.cm.get_cmap("Spectral")(R)
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for g in np.unique(assignments):
        idx = np.where(np.array(assignments) == g)
        ax.scatter(tsne_x[idx], tsne_y[idx], tsne_z[idx], c=cmap[g],
                   label=g, s=0.5, marker='o', alpha=0.8)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    ax.view_init(70, 135)
    plt.title('Assignments by GMM')
    plt.legend(ncol=3)
    plt.show()
    timestr = time.strftime("_%Y%m%d_%H%M")
    my_file = 'train_assignments'  # TODO: high: magic variable
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))

def plot_accuracy_bsoidapp(scores):
    """
    :param scores: 1D array, cross-validated accuracies for MLP classifier.
    """
    fig = plt.figure(facecolor='w', edgecolor='k')
    fig.suptitle(f"Performance on {HLDOUT * 100} % data")
    ax = fig.add_subplot(111)
    ax.boxplot(scores, notch=None)
    x = np.random.normal(1, 0.04, size=len(scores))
    plt.scatter(x, scores, s=40, c='r', alpha=0.5)
    ax.set_xlabel('MLP classifier')
    ax.set_ylabel('Accuracy')
    # plt.show()
    return fig, plt
def plot_accuracy_bsoidumap(scores):
    """
    :param scores: 1D array, cross-validated accuracies for MLP classifier.
    """
    fig = plt.figure(facecolor='w', edgecolor='k')
    fig.suptitle(f"Performance on {HLDOUT * 100} % data")
    ax = fig.add_subplot(111)
    ax.boxplot(scores, notch=None)
    x = np.random.normal(1, 0.04, size=len(scores))
    plt.scatter(x, scores, s=40, c='r', alpha=0.5)
    ax.set_xlabel('MLP classifier')
    ax.set_ylabel('Accuracy')
    plt.show()
    timestr = time.strftime("_%Y%m%d_%H%M")
    my_file = 'clf_scores'
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
def plot_accuracy_bsoidpy(scores):
    """
    :param scores: 1D array, cross-validated accuracies for SVM classifier.
    """
    fig = plt.figure(facecolor='w', edgecolor='k')
    fig.suptitle(f"Performance on {HLDOUT * 100} % data")
    ax = fig.add_subplot(111)
    ax.boxplot(scores, notch=None)
    x = np.random.normal(1, 0.04, size=len(scores))
    plt.scatter(x, scores, s=40, c='r', alpha=0.5)
    ax.set_xlabel('SVM classifier')
    ax.set_ylabel('Accuracy')
    plt.show()
    timestr = time.strftime("_%Y%m%d_%H%M")
    my_file = 'clf_scores'
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))

def plot_accuracy_bsoidvoc(scores):
    """
    :param scores: 1D array, cross-validated accuracies for MLP classifier.
    """
    fig = plt.figure(facecolor='w', edgecolor='k')
    fig.suptitle(f"Performance on {HLDOUT * 100} % data")
    ax = fig.add_subplot(111)
    ax.boxplot(scores, notch=None)
    x = np.random.normal(1, 0.04, size=len(scores))
    plt.scatter(x, scores, s=40, c='r', alpha=0.5)
    ax.set_xlabel('MLP classifier')
    ax.set_ylabel('Accuracy')
    plt.show()
    timestr = time.strftime("_%Y%m%d_%H%M")
    my_file = 'clf_scores'
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))


def plot_durhist_bsoidapp(lengths, grp):
    """
    :param lengths: 1D array, run lengths of each bout.
    :param grp: 1D array, corresponding label.
    """
    timestr = time.strftime("_%Y%m%d_%H%M")
    fig, ax = plt.subplots()
    R = np.linspace(0, 1, len(np.unique(grp)))
    cmap = plt.cm.get_cmap("Spectral")(R)
    for i in range(0, len(np.unique(grp))):
        fig.suptitle("Duration histogram of {} behaviors".format(len(np.unique(TM))))
        x = lengths[np.where(grp == i)]
        ax.hist(x, density=True, color=cmap[i], alpha=0.3, label='Group {}'.format(i))
    plt.legend(loc='upper right')
    plt.show()
    my_file = 'duration_hist_100msbins'
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
    return
def plot_durhist_bsoidpy(lengths, grp):
    """
    :param lengths: 1D array, run lengths of each bout.
    :param grp: 1D array, corresponding label.
    """
    timestr = time.strftime("_%Y%m%d_%H%M")
    fig, ax = plt.subplots()
    R = np.linspace(0, 1, len(np.unique(grp)))
    cmap = plt.cm.get_cmap("Spectral")(R)
    for i in range(0, len(np.unique(grp))):
        fig.suptitle("Duration histogram of {} behaviors".format(len(np.unique(TM))))
        x = lengths[np.where(grp == i)]
        ax.hist(x, density=True, color=cmap[i], alpha=0.3, label='Group {}'.format(i))
    plt.legend(loc='upper right')
    plt.show()
    my_file = 'duration_hist_100msbins'
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
    return
def plot_durhist_bsoidvoc(lengths, grp):
    """
    :param lengths: 1D array, run lengths of each bout.
    :param grp: 1D array, corresponding label.
    """
    timestr = time.strftime("_%Y%m%d_%H%M")
    fig, ax = plt.subplots()
    R = np.linspace(0, 1, len(np.unique(grp)))
    cmap = plt.cm.get_cmap("Spectral")(R)
    for i in range(0, len(np.unique(grp))):
        fig.suptitle("Duration histogram of {} behaviors".format(len(np.unique(TM))))
        x = lengths[np.where(grp == i)]
        ax.hist(x, density=True, color=cmap[i], alpha=0.3, label='Group {}'.format(i))
    plt.legend(loc='upper right')
    plt.show()
    my_file = 'duration_hist_100msbins'
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
    return
def plot_durhist_bsoidumap(lengths, grp):
    """
    :param lengths: 1D array, run lengths of each bout.
    :param grp: 1D array, corresponding label.
    """
    timestr = time.strftime("_%Y%m%d_%H%M")
    fig, ax = plt.subplots()
    R = np.linspace(0, 1, len(np.unique(grp)))
    cmap = plt.cm.get_cmap("Spectral")(R)
    for i in range(0, len(np.unique(grp))):
        fig.suptitle("Duration histogram of {} behaviors".format(len(np.unique(TM))))  # TODO: <--
        x = lengths[np.where(grp == i)]
        ax.hist(x, density=True, color=cmap[i], alpha=0.3, label='Group {}'.format(i))
    plt.legend(loc='upper right')
    plt.show()
    my_file = 'duration_hist_100msbins'
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
    return

def plot_feats_bsoidumap(feats: list, labels: list):
    """
    :param feats: list, features for multiple sessions
    :param labels: list, labels for multiple sessions
    """
    result = isinstance(labels, list)
    timestr = time.strftime("_%Y%m%d_%H%M")
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
                        fig.suptitle("{} pixels".format(feat_ls[j]))
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
                        fig.suptitle("{} pixels".format(feat_ls[j]))
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                my_file = 'sess{}_feat{}_hist'.format(k + 1, j + 1)
                fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
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
                    fig.suptitle("{} pixels".format(feat_ls[j]))
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
                    fig.suptitle("{} pixels".format(feat_ls[j]))
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            my_file = 'feat{}_hist'.format(j + 1)
            fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
        plt.show()
def plot_feats_bsoidpy(feats: list, labels: list):
    """
    :param feats: list, features for multiple sessions
    :param labels: list, labels for multiple sessions
    """
    result = isinstance(labels, list)
    timestr = time.strftime("_%Y%m%d_%H%M")
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
                for i in range(0, len(np.unique(labels_k))):
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
                fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
            plt.show()
    else:
        R = np.linspace(0, 1, len(np.unique(labels)))
        color = plt.cm.get_cmap("Spectral")(R)
        feat_ls = ("Relative snout to forepaws placement", "Relative snout to hind paws placement",
                   "Inter-forepaw distance", "Body length", "Body angle",
                   "Snout displacement", "Tail-base displacement")
        for j in range(0, feats.shape[0]):
            fig = plt.figure(facecolor='w', edgecolor='k')
            for i in range(0, len(np.unique(labels))):
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
            my_file = 'feat{}_hist'.format(j + 1)
            fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
        plt.show()
def plot_feats_bsoidapp(feats: list, labels: list):
    """
    :param feats: list, features for multiple sessions
    :param labels: list, labels for multiple sessions
    """
    # TODO: refactor based on `labels` type. Too much repetition, not enough DRY
    result = isinstance(labels, list)
    timestr = time.strftime("_%Y%m%d_%H%M")
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
                        fig.suptitle("{} pixels".format(feat_ls[j]))
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
                        fig.suptitle("{} pixels".format(feat_ls[j]))
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                my_file = 'sess{}_feat{}_hist'.format(k + 1, j + 1)
                fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
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
                    fig.suptitle("{} pixels".format(feat_ls[j]))
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
                    fig.suptitle("{} pixels".format(feat_ls[j]))
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            my_file = 'feat{}_hist'.format(j + 1)
            fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
        plt.show()
def plot_feats_bsoidvoc(feats: list, labels: list):
    """
    :param feats: list, features for multiple sessions
    :param labels: list, labels for multiple sessions
    """
    result = isinstance(labels, list)
    timestr = time.strftime("_%Y%m%d_%H%M")
    if result:
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
                        fig.suptitle("{} pixels".format(feat_ls[j]))
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
                        fig.suptitle("{} pixels".format(feat_ls[j]))
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                my_file = 'sess{}_feat{}_hist'.format(k + 1, j + 1)
                fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
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
                    fig.suptitle("{} pixels".format(feat_ls[j]))
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
                    fig.suptitle("{} pixels".format(feat_ls[j]))
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            my_file = 'feat{}_hist'.format(j + 1)
            fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
        plt.show()


def plot_tmat_bsoidUMAPAPP(tm: np.ndarray) -> object:
    """
    :param tm: object, transition matrix data frame
    """
    fig = plt.figure()
    fig.suptitle(f"Transition matrix of {tm.shape[0]} behaviors")
    sn.heatmap(tm, annot=True)
    plt.xlabel("Next frame behavior")
    plt.ylabel("Current frame behavior")
    plt.show()
    return fig
def plot_tmat_bsoidPYVOC(tm: np.ndarray, fps: int, save_fig_to_file=True) -> None:
    """
    :param tm: object, transition matrix data frame
    :param fps: scalar, camera frame-rate
    """
    fig = plt.figure()
    fig.suptitle(f"Transition matrix of {tm.shape[0]} behaviors")
    sn.heatmap(tm, annot=True)
    plt.xlabel("Next frame behavior")
    plt.ylabel("Current frame behavior")
    plt.show()
    fig_file_name = 'transition_matrix'
    if save_fig_to_file:
        fig.savefig(os.path.join(OUTPUT_PATH, fig_file_name+str(fps)+timestr+'.svg'))
    return
