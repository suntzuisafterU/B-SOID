"""

streamlit api: https://docs.streamlit.io/en/stable/api.html
"""
import joblib
import glob
import numpy as np
import os
import pandas as pd
import streamlit as st
import time

import bsoid


# Instantiate names for buttons, options that can be changed on the fly but logic below stays the same
title = f'B-SOiD streamlit app'
p = None
valid_video_extensions = {'avi', 'mp4', }

# Variables for buttons, drop-down menus, and other things
start_new_project, load_existing_project = 'Start new', 'Load existing'

# Add a slider to the sidebar:
# random_slider = st.sidebar.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0)
# )


########################################################################################################################


# @st.cache(persist=True)
def get_example_vid(path):  # TODO: rename
    with open(path, 'rb') as video_file:
        video_bytes = video_file.read()
    return video_bytes


def main(*args, **kwargs):
    """
    Designated home page when streamlit is run for BSOID
    """
    # Set up current_func() variables
    is_pipeline_loaded = False

    ### Sidebar ###
    is_project_info_submitted_empty = st.sidebar.empty()
    current_pipeline_name_empty = st.sidebar.empty()

    # # Add a selectbox to the sidebar:
    # some_selection = st.sidebar.selectbox(
    #     label='How would you like to be contacted?',
    #     options=('Email', 'Home phone', 'Mobile phone', )
    # )

    st.markdown('---')

    ### Main Page ###
    st.markdown(f'# {title}')
    st.markdown('---')

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
                p = bsoid.io.read_pipeline(path_to_project_file)
                st.success('Pipeline successfully loaded.')
                is_pipeline_loaded = True
                is_project_info_submitted_empty.write('is_project_info_submitted = %s' % str(bool(path_to_project_file)))
        else: return
    except Exception as e:
        # In case of error, show error and do not continue
        st.error(e)
        return
    finally:
        st.markdown(f'---')

    if is_pipeline_loaded:
        show_pipeline_info(p)

    return


def show_pipeline_info(p: bsoid.pipeline.Pipeline):
    """"""
    st.markdown(f'## Pipeline basic information')
    st.markdown(f'- Pipeline name: {p.name}')
    st.markdown(f'- Pipeline description: {p.description}')
    st.markdown(f'- Pipeline file location: {p.file_path}')
    st.markdown('---')


    st.markdown('---')
    return
