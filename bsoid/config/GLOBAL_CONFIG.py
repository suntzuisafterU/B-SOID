"""
TODO: purpose
"""


#################

#BSOIDAPP

import logging
import sys

HLDOUT = 0.2  # Test partition ratio to validate clustering separation.
holdout_percent: float = HLDOUT
CV_IT = 10  # Number of iterations for cross-validation to show it's not over-fitting.
kfold_crossvalidation: int = CV_IT


# TODO: med: instantiate logger explicit object instead of setting global implicit logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level='INFO',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout)

# BSOID UMAP params, nonlinear transform
UMAP_PARAMS = {
    'n_components': 3,
    'min_dist': 0.0,  # small value
    'random_state': 23,
}

# HDBSCAN params, density based clustering
HDBSCAN_PARAMS = {
    'min_samples': 10  # small value
}

# Feedforward neural network (MLP) params
MLP_PARAMS = {
    'hidden_layer_sizes': (100, 10),  # 100 units, 10 layers
    'activation': 'logistic',  # logistics appears to outperform tanh and relu
    'solver': 'adam',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,  # learning rate not too high
    'alpha': 0.0001,  # regularization default is better than higher values.
    'max_iter': 1000,
    'early_stopping': False,
    'verbose': 0  # set to 1 for tuning your feedforward neural network
}


####### BSOIDPY

# EM_GMM parameters
EMGMM_PARAMS = {
    'n_components': 30,
    'covariance_type': 'full',  # t-sne structure means nothing.
    'tol': 0.001,
    'reg_covar': 1e-06,
    'max_iter': 100,
    'n_init': 20,  # 20 iterations to escape poor initialization
    'init_params': 'random',  # random initialization
    'random_state': 23,
    'verbose': 1  # set this to 0 if you don't want to show progress for em-gmm.
}
# Multi-class support vector machine classifier params
SVM_PARAMS = {
    'C': 10,  # 100 units, 10 layers
    'gamma': 0.5,  # logistics appears to outperform tanh and relu
    'probability': True,
    'random_state': 0,  # adaptive or constant, not too much of a diff
    'verbose': 0  # set to 1 for tuning your feedforward neural network
}


### BSOID UMAP
# (nothing original)

### BSOID VOC
# TSNE parameters, can tweak if you are getting undersplit/oversplit behaviors
# the missing perplexity is scaled with data size (1% of data for nearest neighbors)
TSNE_PARAMS = {
    'n_components': 3,  # 3 is good, 2 will not create unique pockets, 4 will screw GMM up (curse of dimensionality)
    'random_state': 23,
    'n_jobs': -1,  # all cores being used, set to -2 for all cores but one.
    'verbose': 2  # shows check points
}
