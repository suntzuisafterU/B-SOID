"""

streamlit api: https://docs.streamlit.io/en/stable/api.html
"""
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from mpl_toolkits.mplot3d import Axes3D  # Despite being "unused", this import MUST stay for 3d plotting to work. PLO!
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import streamlit as st
import sys
import time
import tkinter
import traceback

from bsoid import app, check_arg, config, io, pipeline, streamlit_session_state

logger = config.initialize_logger(__file__)


# Streamlit hack. TODO: low: organize
try:
    import streamlit.ReportThread as ReportThread
    from streamlit.server.Server import Server
except ImportError:
    # Streamlit >= 0.65.0
    import streamlit.report_thread as ReportThread
    from streamlit.server.server import Server

# __all__ = ['get']


### Instantiate names for buttons, options that can be changed on the fly but logic below stays the same

title = f'B-SOiD Streamlit app'
valid_video_extensions = {'avi', 'mp4', }
# Variables for buttons, drop-down menus, and other things
start_new_project_option_text, load_existing_project_option_text = 'Start new', 'Load existing'
# Set keys for objects
key_button_see_rebuild_options = 'key_button_see_model_options'
key_button_change_info = 'key_button_change_info'
key_button_rebuild_model = 'key_button_rebuild_model'
key_button_rebuild_model_confirmation = 'key_button_rebuild_model_confirmation'
key_button_add_new_data = 'key_button_add_new_data'
key_button_update_description = 'key_key_button_update_description'
key_button_add_train_data_source = 'key_button_add_train_data_source'
key_button_add_predict_data_source = 'key_button_add_predict_data_source'
key_button_update_assignments = 'key_button_update_assignments'
# key_button_review_behaviours = 'key_button_review_behaviours'
current_assignment = 'current_assignment'
TestButton1, testbutton2 = 'TestButton1', 'Testbutton2'
### Page variables data ###

streamlit_variables_dict = {  # Instantiate default variable values here
    key_button_see_rebuild_options: False,
    key_button_change_info: False,
    key_button_rebuild_model: False,
    key_button_rebuild_model_confirmation: False,
    key_button_add_new_data: False,
    key_button_add_train_data_source: False,
    key_button_add_predict_data_source: False,
    key_button_update_description: False,
    key_button_update_assignments: False,
    current_assignment: '',
    TestButton1: False, testbutton2: False,
}


# Page layout

def home(*args, **kwargs):
    """
    The designated home page/entry point when Streamlit is used for B-SOiD.

    -------------
    kwargs

        pipeline : str
        A path to an existing pipeline file which will be loaded by default
        on page load. If the pipeline kwarg is not specified, the config.ini
        value will be checked (via bsoid.config), and that file path, if
        present, will be used. If that config.ini key/value pair is not in
        use, then no default path will be specified and it will be entirely
        up to the user to fill out.

    """
    ### Set up initial variables
    global file_session
    file_session = streamlit_session_state.get(**streamlit_variables_dict)
    matplotlib.use('TkAgg')  # For allowing graphs to pop out as separate windows
    is_pipeline_loaded = False

    # Load up pipeline if specified on command line
    pipeline_file_path = kwargs.get('pipeline', '')
    if not pipeline_file_path:  # If not specified on command line, use config.ini path as default if possible.
        if config.default_pipeline_file_path and os.path.isfile(config.default_pipeline_file_path):
            pipeline_file_path = config.default_pipeline_file_path
        # If no config.ini path, then let user choose on page

    ### Sidebar ###

    ### Main Page ###
    st.markdown(f'# {title}')
    st.markdown('------------------------------------------------------------------------------------------------')

    # Start/open project using drop-down menu
    st.markdown('## Open project')
    start_new_opt = st.selectbox(
        label='Start a new project or load an existing one?',
        options=('', start_new_project_option_text, load_existing_project_option_text),
        key='StartProjectSelectBox',
    )

    try:
        # Option: Start new project
        if start_new_opt == start_new_project_option_text:
            st.markdown(f'## Create new project pipeline')
            new_project_name = st.text_input(
                'Enter a new project name. Please only use letters, numbers, and underscores. Press Enter when done.')
            path_to_project_dir = st.text_input(
                'Enter a path to a folder where the new project pipeline will be stored. Press Enter when done.')
            is_project_info_submitted = st.button('Submit', key='SubmitNewProjectInfo')
            if is_project_info_submitted:
                # Error checking first
                if check_arg.has_invalid_chars_in_name_for_a_file(new_project_name):
                    raise ValueError(
                        f'Project name has invalid characters present. Re-submit project '
                        f'pipeline name. {new_project_name}')
                if not os.path.isdir(path_to_project_dir):
                    raise NotADirectoryError(f'The following (in double quotes) is not a '
                                             f'valid directory: "{path_to_project_dir}". TODO: elaborate on error')
                # If OK: create default pipeline, save, continue
                p: pipeline.BasePipeline = pipeline.PipelinePrime(name=new_project_name).save(path_to_project_dir)
                st.success('New project pipeline saved to disk')
                is_pipeline_loaded = True
        # Option: Load existing project
        elif start_new_opt == load_existing_project_option_text:
            st.write('Load existing project pipeline')
            path_to_project_file = st.text_input(
                'Enter full path to existing project pipeline file',
                value=pipeline_file_path,  # TODO: remove this line later, or change to a config default?
                key='text_input_load_existing_pipeline'
            )
            # Do checks
            if path_to_project_file:
                # Error checking first
                if not os.path.isfile(path_to_project_file) or not path_to_project_file.endswith('.pipeline'):
                    raise FileNotFoundError(f'Path to valid BSOID pipeline file was not found. '
                                            f'User submitted path: {path_to_project_file}')
                # If OK: load project, continue
                logger.debug(f'Attempting to open: {path_to_project_file}')
                p: pipeline.BasePipeline = io.read_pipeline(path_to_project_file)
                logger.debug(f'Successfully opened: {path_to_project_file}')
                st.success('Pipeline loaded successfully.')
                is_pipeline_loaded = True
        # Option: no selection made. Wait for user.
        else:
            return
    except Exception as e:
        # In case of error, show error and do not continue
        st.markdown('')
        st.error(e)
        # st.markdown(f'Stack trace for error: {traceback.extract_stack()}')
        return

    if is_pipeline_loaded:
        st.markdown('------------------------------------------------------------------------------------------------')
        show_pipeline_info(p)

    return


def show_pipeline_info(p: pipeline.PipelinePrime, *args, **kwargs):
    """"""
    ### SIDEBAR INFORMATION
    # st.sidebar.markdown(f'gmm_n_components = {p.gmm_n_components}')
    # st.sidebar.markdown(f'gmm_covariance_type = {p.gmm_covariance_type}')
    # st.sidebar.markdown(f'gmm_tol = {p.gmm_tol}')
    # st.sidebar.markdown(f'gmm_reg_covar = {p.gmm_reg_covar}')
    # st.sidebar.markdown(f'gmm_n_init = {p.gmm_n_init}')
    # st.sidebar.markdown(f'gmm_init_params = {p.gmm_init_params}')

    # st.sidebar.slider('TODO: label', 1, 50, p.gmm_n_components, 1)

    ### MAIN PAGE INFORMATION ###
    st.markdown(f'## Pipeline basic information')
    st.markdown(f'- Name: **{p.name}**')
    st.markdown(f'- Description: **{p.description}**')
    st.markdown(f'- Local file location: **{p.file_path}**')
    st.markdown(f'- Training data sources list: '
                f'{None if len(p.training_data_sources) == 0 else ",".join(p.training_data_sources)}')
    st.markdown(f'- Test data sources: ')
    st.markdown(f'- Is the model built: **{p.is_built}**')
    st.markdown(f'- - Number of data points in training data set: '
                f'**{len(p.df_features_train) if p.df_features_train is not None else None}**')
    st.markdown(f'- - Total unique behaviours clusters: **{len(p.unique_assignments)}**')
    if p.cross_val_scores is not None:
        cross_val_score_text = f'- Median cross validation score: **{round(np.median(p.cross_val_scores), 2)}** ' \
                               f'(scores: {[round(x, 3) for x in list(p.cross_val_scores)]})'
    else:
        cross_val_score_text = f'- Cross validation score not available'
    st.markdown(f'{cross_val_score_text}')
    # st.markdown(f'- Raw assignment values: **{p.unique_assignments}**')

    st.markdown('------------------------------------------------------------------------------------------------')

    return show_actions(p)


def show_actions(p: pipeline.PipelinePrime):
    """"""

    ### Sidebar
    for a in p.unique_assignments:
        behaviour = p.get_assignment_label(a)
        st.sidebar.markdown(f'Assignment {a}: Behaviour = {behaviour}')

    ### Main
    st.markdown(f'## Actions')

    ## Modify basic pipeline info here ##
    st.markdown(f'### Modify pipeline info')
    button_update_description = st.button(f'TODO: Change description', key_button_update_description)
    if button_update_description:
        file_session[key_button_update_description] = not file_session[key_button_update_description]
    if file_session[key_button_update_description]:
        text_input_change_desc = st.text_input(f'WORK IN PROGRESS, not yet functional: Change project description here')
        if text_input_change_desc:
            p.set_description(text_input_change_desc).save()
            st.success(f'Pipeline description was changed! Refresh the page (or press "R") to see changes.')
    # TODO: low: add a "change save location" option?

    ########################################### MODEL BUILDING ##########################################
    st.markdown(f'### Model building')
    button_add_new_data = st.button('Add new data source to model (WIP)', key_button_add_new_data)
    if button_add_new_data:  # Click button, flip state
        file_session[key_button_add_new_data] = not file_session[key_button_add_new_data]
    if file_session[key_button_add_new_data]:  # Now check on value and display accordingly
        st.markdown(f'### Do you want to add data that will be used to train the model, or '
                    f'data that the model will evaluate?')
        # 1/2: Button for adding data to training data set
        button_add_train_data_source = st.button('-> Add new data for model training',
                                                 key=key_button_add_train_data_source)  # TODO: low: review button text
        if button_add_train_data_source:
            file_session[key_button_add_train_data_source] = not file_session[key_button_add_train_data_source]
            file_session[key_button_add_predict_data_source] = False
        if file_session[key_button_add_train_data_source]:
            st.markdown(f'TODO: add in new train data')
            input_new_data_source = st.text_input("TODO: Add new data source for training the model")
            if input_new_data_source:
                # Check if file exists
                if not os.path.isfile(input_new_data_source):
                    st.error(FileNotFoundError(f'TODO: expand: File not found: {input_new_data_source}. '
                                               f'Data not added to pipeline.'))
                # Add to pipeline, save
                else:
                    p = p.add_predict_data_source(input_new_data_source).save()
                    p = p.save()
                    st.success(f'TODO: New prediction data added to pipeline successfully! Pipeline has been saved.')
        # 2/2: Button for adding data to prediction set
        button_add_predict_data_source = st.button('-> Add new data to be evaluated by the model',
                                                   key=key_button_add_predict_data_source)
        if button_add_predict_data_source:
            file_session[key_button_add_predict_data_source] = not file_session[key_button_add_predict_data_source]
            file_session[key_button_add_train_data_source] = False
        if file_session[key_button_add_predict_data_source]:
            st.markdown(f'TODO: add in new predict data')
            input_new_predict_data_source = st.text(f'TODO: add new data source to be predicted by model')
            if input_new_predict_data_source:
                # Check if file exists
                if not os.path.isfile(input_new_predict_data_source):
                    st.error(FileNotFoundError(f'TODO: expand: File not found: {input_new_predict_data_source}. '
                                               f'Data not added to pipeline.'))
                else:
                    p = p.add_predict_data_source(input_new_predict_data_source)
                    p = p.save()
                    st.success(f'TODO: New prediction data added to pipeline successfully! Pipeline has been saved.')
        st.markdown('')
        st.markdown('')

    # Menu: rebuilding classifier
    button_see_rebuild_options = st.button('Rebuild model (WIP)', key_button_see_rebuild_options)
    if button_see_rebuild_options:  # Click button, flip state
        file_session[key_button_see_rebuild_options] = not file_session[key_button_see_rebuild_options]
    if file_session[key_button_see_rebuild_options]:  # Now check on value and display accordingly
        st.markdown('')
        slider_gmm_n_components = st.slider(f'GMM Components (currently set at: {p.gmm_n_components})', 2, 40, p.gmm_n_components)
        input_gmm_tolerance = st.number_input(f'gmm tolerance (currently set at: {p.gmm_tol}): TODO: clean up this line', 1e-10, 50., p.gmm_tol, 0.1, format='%f')
        st.markdown('')
        button_rebuild_model = st.button('Re-build model (WIP)', key_button_rebuild_model)
        if button_rebuild_model:
            file_session[key_button_rebuild_model] = not file_session[key_button_rebuild_model]
        if file_session[key_button_rebuild_model]:
            button_confirmation_of_rebuild = st.button('Confirm', key_button_rebuild_model_confirmation)
            if button_confirmation_of_rebuild:
                file_session[key_button_rebuild_model_confirmation] = True
            if file_session[key_button_rebuild_model_confirmation]:
                with st.spinner('Rebuilding model...'):
                    # app.sample_runtime_function()
                    p = p.build(True, True).save()
                st.success(f'Model was successfully re-built!')
                file_session[key_button_rebuild_model_confirmation] = False
        st.markdown('----------------------------------------------------------------------------------------------')
    st.markdown('--------------------------------------------------------------------------------------------------')

    ######################################### MODEL DIAGNOSTICS ########################################################
    st.markdown(f'### Model Diagnostics')
    st.markdown(f'See GMM distributions according to TSNE-reduced feature dimensions // TODO: make this shorter.')
    st.markdown(f'View distribution of assignments')
    view = st.button(f'View assignment distribution')
    if view:
        fig, ax = p.get_plot_svm_assignments_distribution()
        st.pyplot(fig)
    gmm_button = st.button('Pop out window of GMM distribution')  # TODO: low: phrase this button better?
    if gmm_button:
        p.plot_assignments_in_3d(show_now=True)

    st.markdown('-----------------------------------------------------------------------------------------------------')

    ############################## VIEWING SAMPLE VIDEOS OF BEHAVIOURS #################################################
    st.markdown('')
    st.markdown(f'### Reviewing example videos of behaviours')
    # button_review_behaviours = st.button('Review assignments labels', key_button_review_behaviours)
    # if button_review_behaviours:  # Click button, flip state
    #     file_session[key_button_review_behaviours] = not file_session[key_button_review_behaviours]
    # if file_session[key_button_review_behaviours]:  # Now check on value and display accordingly
    #     for a in p.unique_assignments:
    #         behaviour = p.get_assignment_label(a)
    #         st.markdown(f'Assignment {a}: Behaviour = {behaviour}')

    example_videos_dir_file_list = [x for x in os.listdir(config.EXAMPLE_VIDEOS_OUTPUT_PATH)  # TODO: add user intervention on default path to check?
                                    if x.split('.')[-1] in valid_video_extensions]
    videos_dict = {k: os.path.join(config.EXAMPLE_VIDEOS_OUTPUT_PATH, k)
                   for k in example_videos_dir_file_list}

    vid = st.selectbox(label=f"Total videos found: {len(videos_dict)}",
                       options=list(videos_dict.keys()))  # TODO: low: add key?
    try:
        st.video(get_example_vid(videos_dict[vid]))
    except FileNotFoundError:
        st.error(FileNotFoundError(f'No example behaviour videos were found at this time. '
                                   f'Try generating them at check back again after. '
                 f'// DEBUG INFO: path checked: {config.EXAMPLE_VIDEOS_OUTPUT_PATH}'))

    # Line up names to behaviours
    st.markdown('')
    button_label_assignments = st.button('Review assignments labels', key_button_update_assignments)
    if button_label_assignments:  # Click button, flip state
        file_session[key_button_update_assignments] = not file_session[key_button_update_assignments]
    if file_session[key_button_update_assignments]:  # Now check on value and display accordingly
        # st.markdown('')
        assignment = st.selectbox(label='Choose an assignment:',
                                  options=['', ]+list(p.unique_assignments))

        text_input_behaviour = st.text_input(f'WIP: Add label to assignment #{assignment}')
        if text_input_behaviour and str(assignment):
            # pass
            p = p.update_assignment_label(assignment, text_input_behaviour).save()
            # st.success(f'Added new label')
            # file_session[key_button_update_assignments] = False


def review_videos(p):




    return


# Accessory functions #

def line_break():
    st.markdown('---')


def get_3d_plot(p, **kwargs):
    return p.plot_assignments_in_3d(**kwargs)


# @st.cache  # TODO: will st.cache benefit this part?
def get_example_vid(path):  # TODO: rename
    """"""
    check_arg.ensure_is_file(path)
    with open(path, 'rb') as video_file:
        video_bytes = video_file.read()
    return video_bytes


def flip_button_state(button_key: str):
    # NOTE: LIKELY DOES NOT WORK!
    file_session[key_button_see_rebuild_options] = not file_session[key_button_see_rebuild_options]
    pass


# Misc.

def example_of_value_saving():
    session_state = streamlit_session_state.get(**streamlit_variables_dict)

    st.markdown("# [Title]")
    button1 = st.button('Test Button 1', 'TestButton1')
    st.markdown(f'Pre button1: Button 1 session state: {session_state["TestButton1"]}')
    if button1:
        session_state['TestButton1'] = not session_state['TestButton1']
    if session_state['TestButton1']:
        line_break()
        # session_state['TestButton1'] = not session_state['TestButton1']

        st.markdown(f'In button1: Button 1 session state: {session_state["TestButton1"]}')

        button2 = st.button('Test Button 2', 'TestButton2')
        if button2:
            session_state['TestButton2'] = not session_state['TestButton2']
        if session_state['TestButton2']:
            line_break()
            session_state['TestButton2'] = True
            st.markdown('button2 pressed')

    return


# Main: likely to be deleted later.

if __name__ == '__main__':
    # Note: this import only necessary when running streamlit onto this file specifically rather than
    #   calling `streamlit run main.py streamlit`
    BSOID_project_path = os.path.dirname(os.path.dirname(__file__))
    if BSOID_project_path not in sys.path:
        sys.path.insert(0, BSOID_project_path)
    # home()
    example_of_value_saving()


