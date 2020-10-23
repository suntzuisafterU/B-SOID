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

from bsoid import check_arg, config, io, pipeline

logger = config.initialize_logger(__file__)



### Instantiate names for buttons, options that can be changed on the fly but logic below stays the same
title = f'B-SOiD Streamlit app'
valid_video_extensions = {'avi', 'mp4', }
# Variables for buttons, drop-down menus, and other things
start_new_project_option_text, load_existing_project_option_text = 'Start new', 'Load existing'

### Page data ###

streamlit_variables_dict = {  # Instantiate default variable values here
    # 'Testbutton1': False,
    # 'Testbutton2': False,

}


def home(*args, **kwargs):
    """
    Designated home page/entry point when Streamlit is used for B-SOiD
    """
    matplotlib.use('TkAgg')
    # # global streamlit_variables_dict
    # session_state = get(**streamlit_variables_dict)
    # # session_state = get()
    #
    # st.markdown("Stuff at top!")
    # button1 = st.button('Test Button 1', 'Testbutton1')
    # st.markdown(f'Pre button1: Button 1 session state: {session_state["Testbutton1"]}')
    # if button1:
    #     session_state['Testbutton1'] = not session_state['Testbutton1']
    # if session_state['Testbutton1']:
    #     line_break()
    #     # session_state['Testbutton1'] = not session_state['Testbutton1']
    #
    #     st.markdown(f'In button1: Button 1 session state: {session_state["Testbutton1"]}')
    #
    #     button2 = st.button('Test Button 2', 'Testbutton2')
    #     if button2:
    #         session_state['Testbutton2'] = not session_state['Testbutton2']
    #     if session_state['Testbutton2']:
    #         line_break()
    #         session_state['Testbutton2'] = True
    #         st.markdown('button2 pressed')

    line_break()
    # Set up current function variables
    is_pipeline_loaded = False

    ### Sidebar ###
    st.sidebar.markdown(f'Settings')
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
        options=('', start_new_project_option_text, load_existing_project_option_text),
        key='StartProjectSelectBox',
    )
    # Initialize project
    try:
        # Start new project
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
            is_project_info_submitted_empty.write(f'is_project_info_submitted = {is_project_info_submitted}')
        # Load existing
        elif start_new_opt == load_existing_project_option_text:
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
                p: pipeline.BasePipeline = io.read_pipeline(path_to_project_file)
                logger.debug(f'Successfully opened: {path_to_project_file}')
                st.success('Pipeline successfully loaded.')
                is_pipeline_loaded = True
                is_project_info_submitted_empty.write('is_project_info_submitted = %s' % str(bool(path_to_project_file)))
        else: return
    except Exception as e:
        # In case of error, show error and do not continue
        st.markdown(f'{traceback.extract_stack()}')
        st.error(f'{repr(e)}')
        return
    finally:
        st.markdown(f'---')

    if is_pipeline_loaded:
        current_pipeline_name_empty.markdown(f'Pipeline name: __{p.name}__')
        show_pipeline_info(p)

    return


def show_pipeline_info(p: pipeline.PipelinePrime, **kwargs):
    """"""
    st.markdown(f'## Pipeline basic information')
    st.markdown(f'- Pipeline name: {p.name}')
    st.markdown(f'- Pipeline description: {p.description}')
    st.markdown(f'- Pipeline file location: {p.file_path}')
    # st.markdown(f'- TODO: FIX THIS: Data sources: {len(p.train_data_files_paths)}') #?
    # for loc in p.train_data_files_paths:
    #     st.markdown(f'- - {loc}')
    st.markdown(f'- Are the classifiers built: {p.is_built}')
    st.markdown(f'- Number of data points in df_features: '
                f'{len(p.df_features_train) if p.df_features_train is not None else None}')
    line_break()

    return show_actions(p)


def show_actions(p: pipeline.PipelinePrime):
    # Rebuild classifiers TODO?

    # Sidebar
    azim = st.sidebar.slider("azimuth (camera angle rotation about the Y-axis)", 0, 90, 20, 5)
    elev = st.sidebar.slider("elevation (camera rotation about the Z-axis)", 0, 180, 110, 5)
    st.sidebar.markdown(f'---')

    # Main
    st.markdown(f'### (Re-)build classifiers?')
    st.markdown(f'TODO: button goes here :)')

    st.markdown('### Add new data sources')
    new_file_button = st.button('Enter new file (TODO)')  # TODO
    if new_file_button:
        st.markdown(f'Work In Progress!')
        # new_filepath = st.text_input('Insert new data file:')
        # if new_filepath:
        #     if not os.path.isfile(new_filepath) or not os.path.isdir(new_filepath):
        #         st.error(ValueError(f'Invalid path to data file: {new_filepath}'))
        #         return
        #     p.add_train_data_source(new_filepath).save()

    line_break()
    st.markdown(f'### Viewing GMM distributions')

    # GMM diagnostics
    gmm_button = st.button('Pop up GMM distributions')
    if gmm_button:
        p.plot_assignments_in_3d(show_now=True)

    #
    line_break()
    st.markdown(f'### Reviewing example videos of behaviours')
    example_vids_dir_file_list = [x for x in os.listdir(config.EXAMPLE_VIDEOS_OUTPUT_PATH)
                                  if x.split('.')[-1] in valid_video_extensions]
    demo_videos = {k: os.path.join(config.EXAMPLE_VIDEOS_OUTPUT_PATH, k)
                   for k in example_vids_dir_file_list}

    vid = st.selectbox("TODO: rename text for video select box", list(demo_videos.keys()), 0)
    with open(demo_videos[vid], 'rb') as video_file:
        # video_file = open(demo_vids[vid], 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)


    # button_create_example_vids = st.button('', key='asdfasdf')


# Accessory functions #

# @st.cache
def get_3d_plot(p, **kwargs):
    return p.plot_assignments_in_3d(**kwargs)


def line_break():
    st.markdown('---')


def get_example_vid(path):  # TODO: rename
    """"""
    check_arg.ensure_is_file(path)
    with open(path, 'rb') as video_file:
        video_bytes = video_file.read()
    return video_bytes


# @st.cache
def plot_GM_assignments_in_3d(data: np.ndarray, assignments, show_now=True, azim_elev = (70,135)) -> object:
    """
    100% copied from bsoid/util/visuals.py....don't keep this functions long term.

    Plot trained TSNE for EM-GMM assignments
    :param data: 2D array, trained_tsne array (3 columns)
    :param assignments: 1D array, EM-GMM assignments
    """
    # Arg checking
    if not isinstance(data, np.ndarray):
        err = f'Expected `data` to be of type numpy.ndarray but instead found: {type(data)} (value = {data}).'
        logger.error(err)
        raise TypeError(err)
    # Parse kwargs
    s = 0.5
    marker = 'o'
    alpha = 0.8
    title = 'Assignments by GMM'
    # Plot graph
    # uk = list(np.unique(assignments))
    R = np.linspace(0, 1, len(np.unique(assignments)))
    colormap = plt.cm.get_cmap("Spectral")(R)
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Loop over assignments
    for i, g in enumerate(np.unique(assignments)):
        try:
            # Select data for only assignment i
            idx = np.where(np.array(assignments) == g)
            # Assign to colour and plot
            ax.scatter(tsne_x[idx], tsne_y[idx], tsne_z[idx], c=colormap[i],
                       label=g,
                       s=s,
                       marker=marker,
                       alpha=alpha
                       )
        except TypeError as te:
            logger.error(f'type error: {te} // i = {i} / g = {g}')
            raise te
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

    return fig, ax


# SessionState: attempting to use the hack for saving session state.
try:
    import streamlit.ReportThread as ReportThread
    from streamlit.server.Server import Server
except ImportError:
    # Streamlit >= 0.65.0
    import streamlit.report_thread as ReportThread
    from streamlit.server.server import Server


class SessionState(object):
    def __init__(self, **kwargs):
        """A new SessionState object.

        Parameters
        ----------
        **kwargs : any
            Default values for the session state.

        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'

        """
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except Exception as e:
            logger.error(f'Unexpected error: {repr(e)}')
            raise e

    def __setitem__(self, key, value):
        setattr(self, key, value)


def get(**kwargs):
    """Gets a SessionState object for the current session.

    Creates a new object if necessary.

    Parameters
    ----------
    **kwargs : any
        Default values you want to add to the session state, if we're creating a
        new one.

    Example
    -------
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    ''
    >>> session_state.user_name = 'Mary'
    >>> session_state.favorite_color
    'black'

    Since you set user_name above, next time your script runs this will be the
    result:
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    'Mary'

    """
    # Hack to get the session object from Streamlit.

    ctx = ReportThread.get_report_ctx()

    this_session = None

    current_server = Server.get_current()
    if hasattr(current_server, '_session_infos'):
        # Streamlit < 0.56
        session_infos = Server.get_current().session_infos.values()
    else:
        session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        if (
            # Streamlit < 0.54.0
            (hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg)
            or
            # Streamlit >= 0.54.0
            (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue)
            or
            # Streamlit >= 0.65.2
            (not hasattr(s, '_main_dg') and s._uploaded_file_mgr == ctx.uploaded_file_mgr)
        ):
            this_session = s

    if this_session is None:
        raise RuntimeError(
            "Oh noes. Couldn't get your Streamlit Session object. "
            'Are you doing something fancy with threads?')

    # Got the session object! Now let's attach some state into it.

    if not hasattr(this_session, '_custom_session_state'):
        this_session._custom_session_state = SessionState(**kwargs)

    return this_session._custom_session_state


__all__ = ['get']


# Misc
def example_of_value_saving():
    session_state = get(**streamlit_variables_dict)

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


# Main

if __name__ == '__main__':
    # Note: this import only necessary when running streamlit onto this file specifically rather than
    #   calling `streamlit run main.py streamlit`
    BSOID_project_path = os.path.dirname(os.path.dirname(__file__))
    if BSOID_project_path not in sys.path:
        sys.path.insert(0, BSOID_project_path)
    # home()
    example_of_value_saving()


