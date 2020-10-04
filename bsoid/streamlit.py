"""

streamlit api: https://docs.streamlit.io/en/stable/api.html
"""
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from mpl_toolkits.mplot3d import Axes3D  # Despite being "unused", this import MUST stay for 3d plotting to work. PLO!

import joblib
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import streamlit as st
import time
import umap


import bsoid
from bsoid import config
logger = config.initialize_logger(__file__)


###

# Instantiate names for buttons, options that can be changed on the fly but logic below stays the same
title = f'B-SOiD streamlit app'
valid_video_extensions = {'avi', 'mp4', }
# Variables for buttons, drop-down menus, and other things
start_new_project, load_existing_project = 'Start new', 'Load existing'

########################################################################################################################


def line_break():
    st.markdown('---')


# @st.cache(allow_output_mutation=True, persist=True)
def plot_GM_assignments_in_3d(data: np.ndarray, assignments, save_fig_to_file: bool, fig_file_prefix='train_assignments', show_now=True, azim_elev = (70,135)) -> object:
    """
    100% copied from bsoid/util/visuals.py....don't keep this functions long term.

    Plot trained TSNE for EM-GMM assignments
    :param data: 2D array, trained_tsne array (3 columns)
    :param assignments: 1D array, EM-GMM assignments
    :param save_fig_to_file:
    :param fig_file_prefix:
    :param show_later: use draw() instead of show()
    """
    # Arg checking
    if not isinstance(data, np.ndarray):
        err = f'Expected `data` to be of type numpy.ndarray but instead found: {type(data)} (value = {data}).'
        logger.error(err)
        raise TypeError(err)
    # Parse kwargs
    s = 's'
    marker = 'o'
    alpha = 0.8
    title = 'Assignments by GMM'
    # Plot graph
    uk = list(np.unique(assignments))
    R = np.linspace(0, 1, len(uk))
    colormap = plt.cm.get_cmap("Spectral")(R)
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Loop over assignments
    for i, g in enumerate(np.unique(assignments)):
        # Select data for only assignment i
        idx = np.where(np.array(assignments) == g)
        # Assign to colour and plot
        ax.scatter(tsne_x[idx], tsne_y[idx], tsne_z[idx], c=colormap[i], label=g, s=s, marker=marker, alpha=alpha)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    ax.view_init(*azim_elev)
    plt.title(title)
    plt.legend(ncol=3)
    if show_now:
        plt.show()
    else:
        plt.draw()

    return fig


###

def get_example_vid(path):  # TODO: rename
    with open(path, 'rb') as video_file:
        video_bytes = video_file.read()
    return video_bytes


def home(*args, **kwargs):
    """
    Designated home page when streamlit is run for BSOID
    """
    # Set up current function variables
    is_pipeline_loaded = False

    ### Sidebar ###
    is_project_info_submitted_empty = st.sidebar.empty()
    current_pipeline_name_empty = st.sidebar.empty()
    st.sidebar.markdown('----')

    ### Main Page ###
    st.markdown(f'# {title}')
    line_break()

    # Open project
    st.markdown('## Open project')
    start_new_opt = st.selectbox(
        label='Start a new project or load an existing one?',
        options=('', start_new_project, load_existing_project),
        key='StartProjectSelectBox',
    )
    # Initialize project
    try:
        # Start new project
        if start_new_opt == start_new_project:
            st.markdown(f'## Create new project pipeline')
            new_project_name = st.text_input(
                'Enter a new project name. Please only use letters, numbers, and underscores. Press Enter when done.')
            path_to_project_dir = st.text_input(
                'Enter a path to a folder where the new project pipeline will be stored. Press Enter when done.')

            is_project_info_submitted = st.button('Submit', key='SubmitNewProjectInfo')

            if is_project_info_submitted:
                # Error checking first
                if bsoid.io.has_invalid_chars_in_name_for_a_file(new_project_name):
                    raise ValueError(
                        f'Project name has invalid characters present. Re-submit project pipeline name. {new_project_name}')
                if not os.path.isdir(path_to_project_dir):
                    raise NotADirectoryError(f'The following (in double quotes) is not a valid directory: "{path_to_project_dir}". TODO: elaborate on error')
                # If OK: create default pipeline, save, continue

                p = bsoid.pipeline.TestPipeline1(name=new_project_name).save(path_to_project_dir)
                st.success('New project pipeline saved to disk')
                is_pipeline_loaded = True
            is_project_info_submitted_empty.write(f'is_project_info_submitted = {is_project_info_submitted}')
        # Load existing
        elif start_new_opt == load_existing_project:
            st.write('Load existing project pipeline')
            path_to_project_file = st.text_input('Enter full path to existing project pipeline file')
            # is_project_info_submitted = st.button('Submit file', key='SubmitExistingProjectInfo')

            # Do checks
            if path_to_project_file:
                # Error checking first
                if not os.path.isfile(path_to_project_file) or not path_to_project_file.endswith('.pipeline'):
                    raise FileNotFoundError(f'Path to valid BSOID pipeline file was not found. '
                                            f'User submitted: {path_to_project_file}')
                # If OK: load project, continue
                logger.debug(f'Attempting to open: {path_to_project_file}')
                p = bsoid.util.io.read_pipeline(path_to_project_file)
                logger.debug(f'Successfully opened: {path_to_project_file}')
                st.success('Pipeline successfully loaded.')
                is_pipeline_loaded = True
                is_project_info_submitted_empty.write('is_project_info_submitted = %s' % str(bool(path_to_project_file)))
        else: return
    except Exception as e:
        # In case of error, show error and do not continue
        st.error(f'{repr(e)}')
        return
    finally:
        st.markdown(f'---')

    if is_pipeline_loaded:
        current_pipeline_name_empty.markdown(f'Pipeline name: __{p.name}__')
        show_pipeline_info(p)

    return


def show_pipeline_info(p: bsoid.pipeline.Pipeline):
    """"""
    st.markdown(f'## Pipeline basic information')
    st.markdown(f'- Pipeline name: {p.name}')
    st.markdown(f'- Pipeline description: {p.description}')
    st.markdown(f'- Pipeline file location: {p.file_path}')
    st.markdown(f'- Data sources:')
    for loc in p.train_data_files_paths:
        st.markdown(f'- - {loc}')
    st.markdown(f'- Are the classifiers built: {p.is_built}')
    st.markdown(f'Number of data points in df_features: {len(p.df_features)}')
    line_break()

    return show_actions(p)


def show_actions(p: bsoid.pipeline.Pipeline):
    # Rebuild classifiers TODO?

    # Sidebar
    azim = st.sidebar.slider("azimuth (camera angle rotation on Y-axis)", 0, 90, 20, 5)
    elev = st.sidebar.slider("elevation (camera rotation about the Z-axis)", 0, 180, 110, 5)
    st.sidebar.markdown(f'---')

    # Main
    st.markdown(f'## Stuff to do? TODO')
    st.markdown('')

    # rebuild_classifiers_button = st.button('Rebuild classifiers', key='RebuildClassifiersButton')
    # if rebuild_classifiers_button:
    #     st.markdown(f'### Rebuilding classifiers')
    #     pass
    # view_analytics_button = st.button('View Analytics', key='ViewAnalyticsButton')
    # if view_analytics_button:

    # st.markdown('### Available analytics')

    # view_EMGMM_distributions = st.button('Show EM/GMM distributions', key='ViewEMGMMDistributionsButton')
    # a, b = generate_data(p)

    st.markdown(f'### EM/GMM distributions')
    # fig = p.plot_assignments_in_3d(show_now=True, azim_elev=(azim, elev))
    # df_dims_and_assignment = p.df_post_tsne[p.dims_cols_names+[p.gmm_assignment_col_name, ]]
    # fig = plot_GM_assignments_in_3d(df_dims_and_assignment[p.dims_cols_names].values,
    #                                 df_dims_and_assignment[p.gmm_assignment_col_name].values, False,
    #                                 azim_elev=(azim, elev))

    fig = p.plot_assignments_in_3d(azim_elev=(azim, elev))

    st.pyplot(fig)

    # fig = generate_3d_emgmm_plot(p, azim, elev)

    # ax = fig.gca(projection='3d')
    # ax.view_init(azim, elev)

    # st.pyplot(fig)


