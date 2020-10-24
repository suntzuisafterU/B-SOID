"""

streamlit api: https://docs.streamlit.io/en/stable/api.html
"""
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from mpl_toolkits.mplot3d import Axes3D  # Despite being "unused", this import MUST stay for 3d plotting to work. PLO!
import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
import os
# import pandas as pd
import streamlit as st
import sys
# import time
# import tkinter
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
key_button_see_rebuild_options = 'key_button_see_classifier_options'
key_button_change_info = 'key_button_change_info'
key_button_rebuild_model = 'key_button_rebuild_model'
key_button_rebuild_model_confirmation = 'key_button_rebuild_model_confirmation'
key_button_add_new_data = 'key_button_add_new_data'
key_button_update_description = 'key_key_button_update_description'
key_button_add_train_data_source = 'key_button_add_train_data_source'
key_button_add_predict_data_source = 'key_button_add_predict_data_source'
testbutton1, testbutton2 = 'Testbutton1', 'Testbutton2'
### Page variables data ###

streamlit_variables_dict = {  # Instantiate default variable values here
    key_button_see_rebuild_options: False,
    key_button_change_info: False,
    key_button_rebuild_model: False,
    testbutton1: False,
    testbutton2: False,
    key_button_rebuild_model_confirmation: False,
    key_button_add_new_data: False,
    key_button_add_train_data_source: False,
    key_button_add_predict_data_source: False,
    key_button_update_description: False,
}

file_session = streamlit_session_state.get(**streamlit_variables_dict)


# Page layout

def home(*args, **kwargs):
    """
    Designated home page/entry point when Streamlit is used for B-SOiD
    """
    # Set up initial variables
    matplotlib.use('TkAgg')  # For allowing graphs to pop out as separate windows
    # global streamlit_variables_dict
    # session_state = streamlit_variables_dict.get(**streamlit_variables_dict)
    is_pipeline_loaded = False

    ### Sidebar ###

    # st.sidebar.markdown(f'Settings')
    # is_project_info_submitted_empty = st.sidebar.empty()
    # current_pipeline_name_empty = st.sidebar.empty()
    # st.sidebar.markdown('----')

    ### Main Page ###
    st.markdown(f'# {title}')
    line_break()

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
            # is_project_info_submitted_empty.write(f'is_project_info_submitted = {is_project_info_submitted}')
        # Option: Load existing project
        elif start_new_opt == load_existing_project_option_text:
            st.write('Load existing project pipeline')
            path_to_project_file = st.text_input(
                'Enter full path to existing project pipeline file',
                # "C:\\videoTest1.pipeline",
                key='text_input_load_existing_pipeline'
            )
            # is_project_info_submitted = st.button('Submit file', key='SubmitExistingProjectInfo')

            # Do checks
            if path_to_project_file:
                # Error checking first
                if not os.path.isfile(path_to_project_file) or not path_to_project_file.endswith('.pipeline'):
                    raise FileNotFoundError(f'Path to valid BSOID pipeline file was not found. '
                                            f'User submitted: {path_to_project_file}')
                # If OK: load project, continue
                logger.debug(f'Attempting to open: {path_to_project_file}')
                p: pipeline.BasePipeline = io.read_pipeline(path_to_project_file)
                logger.debug(f'Successfully opened: {path_to_project_file}')
                st.success('Pipeline loaded successfully.')
                is_pipeline_loaded = True
                # is_project_info_submitted_empty.write('is_project_info_submitted = %s' % str(bool(path_to_project_file)))
        # Option: no selection made. Wait for user.
        else:
            return
    except Exception as e:
        # In case of error, show error and do not continue
        st.error(f'{repr(e)}')
        st.markdown(f'Stack trace for error: {traceback.extract_stack()}')
        return
    finally:
        st.markdown(f'---')

    if is_pipeline_loaded:
        # current_pipeline_name_empty.markdown(f'Pipeline name: __{p.name}__')
        show_pipeline_info(p)

    return


def show_pipeline_info(p: pipeline.PipelinePrime, session=None, **kwargs):
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
    st.markdown(f'- Name: {p.name}')
    st.markdown(f'- Description: {p.description}')
    st.markdown(f'- Local file location: {p.file_path}')
    # st.markdown(f'- TODO: FIX THIS: Data sources: {len(p.train_data_files_paths)}') #?
    # for loc in p.train_data_files_paths:
    #     st.markdown(f'- - {loc}')
    st.markdown(f'- Are the classifiers built: {p.is_built}')
    st.markdown(f'- Number of data points in df_features: '
                f'{len(p.df_features_train) if p.df_features_train is not None else None}')

    st.markdown('TODO: Model information')
    # st.markdown('TODO: SVM Information')

    line_break()

    return show_actions(p)


def show_actions(p: pipeline.PipelinePrime):
    # Rebuild classifiers TODO?

    ### Sidebar
    # azim = st.sidebar.slider("azimuth (camera angle rotation about the Y-axis)", 0, 90, 20, 5)
    # elev = st.sidebar.slider("elevation (camera rotation about the Z-axis)", 0, 180, 110, 5)
    # st.sidebar.markdown(f'---')

    ### Main
    st.markdown(f'## Actions')

    # Modify basic pipeline info here
    st.markdown(f'### Modify pipeline info')
    button_update_description = st.button(f'TODO: Change description', key_button_update_description)
    if button_update_description:
        file_session[key_button_update_description] = not file_session[key_button_update_description]
    if file_session[key_button_update_description]:
        x = st.text_input(f'WORK IN PROGRESS: Change project description here')
    # TODO: low: add a "change save location" option?

    # Menu: adding new training data or data to be predicted
    if not file_session[key_button_add_new_data]:
        line_break()
    st.markdown(f'### Model building')
    button_add_new_data = st.button('Add new data source to model', key_button_add_new_data)
    if button_add_new_data:  # Click button, flip state
        file_session[key_button_add_new_data] = not file_session[key_button_add_new_data]
    if file_session[key_button_add_new_data]:  # Now check on value and display accordingly
        st.markdown(f'### Do you want to add data that will be used to train the model, or data that the model will evaluate?')
        # 1/2: Button for adding data to training data set
        button_add_train_data_source = st.button('-> Add new data for model training', key=key_button_add_train_data_source)  # TODO: low: review button text
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
        # line_break()
        st.markdown('')
        st.markdown('')

    # Menu: rebuilding classifier
    button_see_rebuild_options = st.button('Rebuild classifier', key_button_see_rebuild_options)
    if button_see_rebuild_options:  # Click button, flip state
        file_session[key_button_see_rebuild_options] = not file_session[key_button_see_rebuild_options]
    if file_session[key_button_see_rebuild_options]:  # Now check on value and display accordingly
        # st.markdown(f'In button1: Button 1 session state: {session[key_button_rebuild]}')
        st.markdown(f'')
        slider_gmm_n_components = st.slider('GMM Components', 2, 40, p.gmm_n_components)
        input_gmm_tolerance = st.number_input('gmm tol: TODO: replace this with somethign else', 1e-10, 50., p.gmm_tol, 0.1)
        st.markdown(f'')

        button_rebuild_model = st.button('Re-build model', key_button_rebuild_model)
        if button_rebuild_model:
            file_session[key_button_rebuild_model] = not file_session[key_button_rebuild_model]
        if file_session[key_button_rebuild_model]:
            button_confirmation_of_rebuild = st.button('Confirm', key_button_rebuild_model_confirmation)
            if button_confirmation_of_rebuild:
                file_session[key_button_rebuild_model_confirmation] = True
            if file_session[key_button_rebuild_model_confirmation]:
                with st.spinner('Rebuilding model...'):
                    app.sample_runtime_function()
                st.success(f'Model was successfully re-built!')
                file_session[key_button_rebuild_model_confirmation] = False

    line_break()

    ### MODEL DIAGNOSTICS ###
    st.markdown(f'### Model Diagnostics')
    st.markdown(f'See GMM distributions according to TSNE-reduced feature dimensions // TODO: make this shorter.')
    gmm_button = st.button('Pop out window of GMM distrib. // TODO: phrase this better')
    if gmm_button:
        p.plot_assignments_in_3d(show_now=True)


    ### VIEWING SAMPLE VIDEOS OF BEHAVIOURS
    line_break()
    st.markdown(f'### Reviewing example videos of behaviours')
    example_vids_dir_file_list = [x for x in os.listdir(config.EXAMPLE_VIDEOS_OUTPUT_PATH)  # TODO: add user intervention on default path to check?
                                  if x.split('.')[-1] in valid_video_extensions]
    videos_dict = {k: os.path.join(config.EXAMPLE_VIDEOS_OUTPUT_PATH, k)
                   for k in example_vids_dir_file_list}
    # st.markdown(f'')
    vid = st.selectbox(f"Total videos found: {len(videos_dict)}", list(videos_dict.keys()))  # TODO: low: add key?
    try:
        st.video(get_example_vid(videos_dict[vid]))
        # with open(videos_dict[vid], 'rb') as video_file:
        #     video_bytes = video_file.read()
        #     st.video(video_bytes)
    except FileNotFoundError:
        st.error(f'No example behaviour videos were found at this time. Try generating them at check back again after. '
                 f'// DEBUG INFO: path checked: {config.EXAMPLE_VIDEOS_OUTPUT_PATH}')


# Accessory functions #

def get_3d_plot(p, **kwargs):
    return p.plot_assignments_in_3d(**kwargs)


def line_break():
    st.markdown('---')


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


# # SessionState: attempting to use the hack for saving session state.
#
# class SessionState(object):
#     def __init__(self, **kwargs):
#         """A new SessionState object.
#
#         Parameters
#         ----------
#         **kwargs : any
#             Default values for the session state.
#
#         Example
#         -------
#         >>> session_state = SessionState(user_name='', favorite_color='black')
#         >>> session_state.user_name = 'Mary'
#         ''
#         >>> session_state.favorite_color
#         'black'
#
#         """
#         for key, val in kwargs.items():
#             setattr(self, key, val)
#
#     def __getitem__(self, item):
#         try:
#             return getattr(self, item)
#         except Exception as e:
#             logger.error(f'Unexpected error: {repr(e)}')
#             raise e
#
#     def __setitem__(self, key, value):
#         setattr(self, key, value)
#
#
# def get(**kwargs):
#     """Gets a SessionState object for the current session.
#
#     Creates a new object if necessary.
#
#     Parameters
#     ----------
#     **kwargs : any
#         Default values you want to add to the session state, if we're creating a
#         new one.
#
#     Example
#     -------
#     >>> session_state = get(user_name='', favorite_color='black')
#     >>> session_state.user_name
#     ''
#     >>> session_state.user_name = 'Mary'
#     >>> session_state.favorite_color
#     'black'
#
#     Since you set user_name above, next time your script runs this will be the
#     result:
#     >>> session_state = get(user_name='', favorite_color='black')
#     >>> session_state.user_name
#     'Mary'
#
#     """
#     # Hack to get the session object from Streamlit.
#
#     ctx = ReportThread.get_report_ctx()
#
#     this_session = None
#
#     current_server = Server.get_current()
#     if hasattr(current_server, '_session_infos'):
#         # Streamlit < 0.56
#         session_infos = Server.get_current().session_infos.values()
#     else:
#         session_infos = Server.get_current()._session_info_by_id.values()
#
#     for session_info in session_infos:
#         s = session_info.session
#         if (
#             # Streamlit < 0.54.0
#             (hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg)
#             or
#             # Streamlit >= 0.54.0
#             (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue)
#             or
#             # Streamlit >= 0.65.2
#             (not hasattr(s, '_main_dg') and s._uploaded_file_mgr == ctx.uploaded_file_mgr)
#         ):
#             this_session = s
#
#     if this_session is None:
#         raise RuntimeError(
#             "Oh noes. Couldn't get your Streamlit Session object. "
#             'Are you doing something fancy with threads?')
#
#     # Got the session object! Now let's attach some state into it.
#
#     if not hasattr(this_session, '_custom_session_state'):
#         this_session._custom_session_state = SessionState(**kwargs)
#
#     return this_session._custom_session_state


# Misc

def example_of_value_saving():
    session_state = streamlit_session_state.get(**streamlit_variables_dict)

    st.markdown("# [Title]")
    button1 = st.button('Test Button 1', 'Testbutton1')
    st.markdown(f'Pre button1: Button 1 session state: {session_state["Testbutton1"]}')
    if button1:
        session_state['Testbutton1'] = not session_state['Testbutton1']
    if session_state['Testbutton1']:
        line_break()
        # session_state['Testbutton1'] = not session_state['Testbutton1']

        st.markdown(f'In button1: Button 1 session state: {session_state["Testbutton1"]}')

        button2 = st.button('Test Button 2', 'Testbutton2')
        if button2:
            session_state['Testbutton2'] = not session_state['Testbutton2']
        if session_state['Testbutton2']:
            line_break()
            session_state['Testbutton2'] = True
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


