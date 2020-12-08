"""

streamlit api: https://docs.streamlit.io/en/stable/api.html
Number formatting: https://python-reference.readthedocs.io/en/latest/docs/str/formatting.html
    Valid formatters: %d %e %f %g %i
More on formatting: https://pyformat.info/
"""
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from mpl_toolkits.mplot3d import Axes3D  # Despite being "unused", this import MUST stay for 3d plotting to work. PLO!
from typing import Dict, List
import easygui
import matplotlib
import numpy as np
import os
import streamlit as st
import sys
import traceback
# import matplotlib.pyplot as plt
# import pandas as pd
# import time
# import tkinter
# import cPickle as pickle


from bsoid import check_arg, config, io, pipeline, streamlit_session_state

logger = config.initialize_logger(__file__)


##### Instantiate names for buttons, options that can be changed on the fly but logic below stays the same #####

title = f'B-SOiD Streamlit app'
valid_video_extensions = {'avi', 'mp4', }
# Variables for buttons, drop-down menus, and other things
start_new_project_option_text, load_existing_project_option_text = 'Create new', 'Load existing'
pipeline_prime_name, pipeline_epm_name, pipelineTimName = 'PipelinePrime', 'pipeline_epm_name', 'PipelineTim'
training_data_option, predict_data_option = 'Training Data', 'Predict Data'
key_iteration_page_refresh_count = 'key_iteration_page_refresh_count'

# Set keys for objects (mostly buttons) for streamlit components that need some form of persistence.
key_button_show_adv_pipeline_information = 'key_button_show_more_pipeline_information'
key_button_see_rebuild_options = 'key_button_see_model_options'
key_button_see_advanced_options = 'key_button_see_advanced_options'
key_button_change_info = 'key_button_change_info'
key_button_rebuild_model = 'key_button_rebuild_model'
key_button_rebuild_model_confirmation = 'key_button_rebuild_model_confirmation'
key_button_add_new_data = 'key_button_add_new_data'
key_button_menu_remove_data = 'key_button_menu_remove_data'
key_button_update_description = 'key_button_update_description'
key_button_add_train_data_source = 'key_button_add_train_data_source'
key_button_add_predict_data_source = 'key_button_add_predict_data_source'
key_button_review_assignments = 'key_button_update_assignments'
key_button_view_assignments_distribution = 'key_button_view_assignments_distribution'
key_button_save_assignment = 'key_button_save_assignment'
key_button_show_example_videos_options = 'key_button_show_example_videos_options'
key_button_create_new_example_videos = 'key_button_create_new_example_videos'
key_button_menu_label_entire_video = 'key_button_menu_label_entire_video'
### Page variables data ###
streamlit_persitency_variables = {  # Instantiate default variable values here
    key_iteration_page_refresh_count: 0,
    key_button_show_adv_pipeline_information: False,
    key_button_see_rebuild_options: False,
    key_button_see_advanced_options: False,
    key_button_change_info: False,
    key_button_rebuild_model: False,
    key_button_rebuild_model_confirmation: False,
    key_button_add_new_data: False,
    key_button_add_train_data_source: False,
    key_button_add_predict_data_source: False,
    key_button_menu_remove_data: False,
    key_button_update_description: False,
    key_button_review_assignments: False,
    key_button_view_assignments_distribution: False,
    key_button_save_assignment: False,
    key_button_show_example_videos_options: False,
    key_button_create_new_example_videos: False,
    key_button_menu_label_entire_video: False,
}
# TODO: propagate file path thru session var?


##### Page layout #####

def home(**kwargs):
    """
    The designated home page/entry point when Streamlit is used with B-SOiD.
    -------------
    kwargs

        pipeline_path : str
        A path to an existing pipeline file which will be loaded by default
        on page load. If this kwarg is not specified, the config.ini
        value will be checked (via bsoid.config), and that file path, if
        present, will be used. If that config.ini key/value pair is not in
        use, then no default path will be specified and it will be entirely
        up to the user to fill out.

    """
    ### Set up initial variables
    global file_session
    file_session = streamlit_session_state.get(**streamlit_persitency_variables)
    matplotlib.use('TkAgg')  # For allowing graphs to pop out as separate windows
    file_session[key_iteration_page_refresh_count] = file_session[key_iteration_page_refresh_count] + 1
    is_pipeline_loaded = False

    ### SIDEBAR ###
    # st.sidebar.markdown(f'### Iteration: {file_session[key_iteration_page_refresh_count]}')
    # st.sidebar.markdown('------')

    ### MAIN ###
    st.markdown(f'# {title}')
    st.markdown('------------------------------------------------------------------------------------------')
    try:
        # Load up pipeline if specified on command line or config.ini
        pipeline_file_path = kwargs.get('pipeline_path', '')
        if not pipeline_file_path:  # If not specified on command line, use config.ini path as default if possible.
            if config.default_pipeline_file_path and os.path.isfile(config.default_pipeline_file_path):
                pipeline_file_path = config.default_pipeline_file_path
            # If no config.ini path, then let user choose on page

        ## Start/open project using drop-down menu ##
        st.markdown('## Open project')
        start_new_opt = st.selectbox(
            label='Start a new project or load an existing one?',
            options=('', start_new_project_option_text, load_existing_project_option_text),
            key='StartProjectSelectBox',
            index=2 if os.path.isfile(pipeline_file_path) else 0
        )
        st.markdown('')
        # Option: Start new project
        if start_new_opt == start_new_project_option_text:
            st.markdown(f'## Create new project pipeline')
            select_pipe_type = st.selectbox('Select a pipeline implementation', options=('', pipeline_prime_name, pipeline_epm_name, pipelineTimName))
            if select_pipe_type:
                text_input_new_project_name = st.text_input(
                    'Enter a name for your project pipeline. Please only use letters, numbers, and underscores.')
                path_to_pipeline_dir = st.text_input(
                    'Enter a path to a folder where the new project pipeline will be stored. Press Enter when done.')
                button_project_info_submitted_is_clicked = st.button('Submit', key='SubmitNewProjectInfo')
                if button_project_info_submitted_is_clicked:
                    # Error checking first
                    if check_arg.has_invalid_chars_in_name_for_a_file(text_input_new_project_name):
                        char_err = ValueError(f'Project name has invalid characters present. '
                                              f'Re-submit project pipeline name. {text_input_new_project_name}')
                        st.error(char_err)
                        st.stop()
                    if not os.path.isdir(path_to_pipeline_dir):
                        dir_err = NotADirectoryError(f'The following (in double quotes) is not a valid '
                                                     f'directory: "{path_to_pipeline_dir}". TODO: elaborate on error')
                        st.error(dir_err)
                        st.stop()
                    # If OK: create default pipeline, save, continue
                    if select_pipe_type == pipeline_prime_name:
                        p = pipeline.PipelinePrime(text_input_new_project_name).save(path_to_pipeline_dir)
                    elif select_pipe_type == pipeline_epm_name:
                        p = pipeline.PipelineEPM(text_input_new_project_name).save(path_to_pipeline_dir)
                    elif select_pipe_type == pipelineTimName:
                        p = pipeline.PipelineTim(text_input_new_project_name).save(path_to_pipeline_dir)
                    else:
                        st.error(RuntimeError('Something unexpected happened'))
                        st.markdown(f'traceback: {traceback.format_exc()}')
                        st.stop()
                    st.success(f"""
Success! Your new project pipeline has been saved to disk. 

It is recommended that you copy the following path to your clipboard:

{os.path.join(path_to_pipeline_dir, f'{text_input_new_project_name}.pipeline')}

Refresh the page and load up your new pipeline file!
""")
        # Option: Load existing project
        elif start_new_opt == load_existing_project_option_text:
            st.write('Load existing project pipeline')
            path_to_pipeline_file = st.text_input(
                'Enter full path to existing project pipeline file',
                value=pipeline_file_path,  # TODO: remove this line later, or change to a config default?
                key='text_input_load_existing_pipeline'
            )
            # Do checks
            if path_to_pipeline_file:
                # Error checking first
                if not os.path.isfile(path_to_pipeline_file) or not path_to_pipeline_file.endswith('.pipeline'):
                    raise FileNotFoundError(f'Path to valid BSOID pipeline file was not found. '
                                            f'User submitted path: {path_to_pipeline_file}')
                # If OK: load project, continue
                logger.debug(f'Attempting to open: {path_to_pipeline_file}')
                p = io.read_pipeline(path_to_pipeline_file)
                logger.info(f'Streamlit: successfully opened: {path_to_pipeline_file}')
                st.success('Pipeline loaded successfully.')
                is_pipeline_loaded = True
        # Option: no (valid) selection made. Wait for user to select differently.
        else:
            return
    except Exception as e:
        # In case of error, show error and do not continue
        st.markdown('An unexpected error occurred. See below:')
        from traceback import format_exc as get_traceback_string

        st.info(f'Traceback: {get_traceback_string()}')
        st.error(e)
        st.markdown(f'Stack trace for error: {str(traceback.extract_stack())}')
        logger.error(str(traceback.extract_stack()))
        return

    if is_pipeline_loaded:
        start_new_opt = load_existing_project_option_text
        # path_to_project_file = p._source_folder
        st.markdown('----------------------------------------------------------------------------------------------')
        show_pipeline_info(p, pipeline_path=pipeline_file_path)


def show_pipeline_info(p: pipeline.PipelinePrime, pipeline_path, **kwargs):
    """  """
    ### SIDEBAR ###

    ### MAIN PAGE ###
    st.markdown(f'## Pipeline basic information')
    st.markdown(f'- Name: **{p.name}**')
    st.markdown(f'- Description: **{p.description}**')
    st.markdown(f'- Local file location: **{pipeline_path}**')
    st.markdown(f'- Is the model built: **{p.is_built}**')

    ### Menu button: show more info
    button_show_advanced_pipeline_information = st.button(
        f'Expand/collapse advanced info', key=key_button_show_adv_pipeline_information)
    if button_show_advanced_pipeline_information:
        file_session[key_button_show_adv_pipeline_information] = not file_session[key_button_show_adv_pipeline_information]
    if file_session[key_button_show_adv_pipeline_information]:
        st.markdown(f'- Training data sources:')
        if len(p.training_data_sources) > 0:
            for s in p.training_data_sources: st.markdown(f'- - **{s}**')
        else:
            st.markdown(f'- - **None**')
        st.markdown(f'- Predict data sources:')
        if len(p.predict_data_sources) > 0:
            for s in p.predict_data_sources:
                st.markdown(f'- - **{s}**')
        else:
            st.markdown(f'- - **None**')

        st.markdown(f'- Number of data points in training data set: '
                    f'**{len(p.df_features_train_scaled) if p.df_features_train_scaled is not None else None}**')
        st.markdown(f' - Total unique behaviours clusters: **{len(p.unique_assignments)}**')
        if len(p.cross_val_scores) > 0:
            decimals_round = 3
            cross_val_score_text = f'- - Median cross validation score: **{round(np.median(p.cross_val_scores), decimals_round)}** (literal scores: {sorted([round(x, decimals_round) for x in list(p.cross_val_scores)])})'
        else:
            cross_val_score_text = f'- Cross validation score not available'
        st.markdown(f'{cross_val_score_text}')
        st.markdown(f'- Raw assignment values: **{p.unique_assignments}**')
        st.markdown(f'- Features set: {p.all_features}')
    ###

    # Model check before displaying actions that could further change pipeline state.
    if p.is_in_inconsistent_state:
        st.markdown('')
        st.info("""
The pipeline is detected to be in an inconsistent state. 

Some common causes include adding/deleting training data or changing model 
parameters without subsequently rebuilding the model.

We recommend that you rebuild the model to avoid future problems. """.strip())

    # # TODO: for below commented-out: add a CONFIRM button to confirm model re-build, then re-instate

    st.markdown('------------------------------------------------------------------------------------------------')

    return show_actions(p, pipeline_path)


def show_actions(p: pipeline.PipelinePrime, pipeline_path):
    """ Show basic actions that we can perform on the model """
    ### SIDEBAR ###

    ### MAIN PAGE ###
    st.markdown(f'## Actions')

    ################################# CHANGE PIPELINE INFORMATION ###############################################
    st.markdown(f'### Pipeline information')
    button_update_description = st.button(f'Expand/collapse: Change project description', key_button_update_description)
    if button_update_description:
        file_session[key_button_update_description] = not file_session[key_button_update_description]
    if file_session[key_button_update_description]:
        text_input_change_desc = st.text_input(f'WORK IN PROGRESS, not yet functional: Change project description here')
        if text_input_change_desc:
            p.set_description(text_input_change_desc).save(os.path.dirname(pipeline_path))
            st.success(f'Pipeline description was changed! Refresh the '
                       f'page (by clicking the page and pressing "R") to see changes.')

    # TODO: low: add a "change save location" option?

    ####################################### MODEL BUILDING #############################################
    st.markdown(f'## Model building & information')

    ### Menu button: adding new data ###
    button_add_new_data = st.button('Expand/collapse: Add new data to model', key_button_add_new_data)
    if button_add_new_data:  # Click button, flip state
        file_session[key_button_add_new_data] = not file_session[key_button_add_new_data]
    if file_session[key_button_add_new_data]:  # Now check on value and display accordingly
        st.markdown(f'### Do you want to add data that will be used to train the model, or '
                    f'data that the model will evaluate?')
        # 1/2: Button for adding data to training data set
        button_add_train_data_source = st.button('-> Add new data for training the model', key=key_button_add_train_data_source)
        if button_add_train_data_source:
            file_session[key_button_add_train_data_source] = not file_session[key_button_add_train_data_source]
            file_session[key_button_add_predict_data_source] = False  # Close the menu for adding prediction data
        if file_session[key_button_add_train_data_source]:
            st.markdown('')
            input_new_data_source = st.text_input("Input a file path below to data which will be used to train the model")
            if input_new_data_source:
                # Check if file exists
                if not os.path.isfile(input_new_data_source):
                    st.error(FileNotFoundError(f'TODO: expand: File not found: {input_new_data_source}. Data not added to pipeline.'))
                # Add to pipeline, save
                else:
                    p = p.add_train_data_source(input_new_data_source).save(os.path.dirname(pipeline_path))
                    st.success(f'TODO: New training data added to pipeline successfully! Pipeline has been saved to: "{pipeline_path}". Refresh the page to see changes.')  # TODO: finish statement. Add in suggestion to refresh page.
                    file_session[key_button_add_train_data_source] = False  # Reset menu to collapsed state
            st.markdown('')
        # 2/2: Button for adding data to prediction set
        button_add_predict_data_source = st.button('-> Add data to be evaluated by the model', key=key_button_add_predict_data_source)
        if button_add_predict_data_source:
            file_session[key_button_add_predict_data_source] = not file_session[key_button_add_predict_data_source]
            file_session[key_button_add_train_data_source] = False  # Close the menu for adding training data
        if file_session[key_button_add_predict_data_source]:
            st.markdown(f'TODO: add in new predict data')
            input_new_predict_data_source = st.text_input(f'Input a file path below to a new data source which will be analyzed by the model.')
            if input_new_predict_data_source:
                # Check if file exists
                if not os.path.isfile(input_new_predict_data_source):
                    st.error(FileNotFoundError(f'File not found: {input_new_predict_data_source}. '
                                               f'No data was added to pipeline prediction data set.'))

                else:
                    p = p.add_predict_data_source(input_new_predict_data_source).save(os.path.dirname(pipeline_path))
                    st.success(f'New prediction data added to pipeline successfully! Pipeline has been saved. Refresh the page to see cehanges.')
                    file_session[key_button_add_predict_data_source] = False  # Reset menu to collapsed state
        st.markdown('')
        st.markdown('')

    ###

    ### Menu button: removing data ###
    button_remove_data = st.button('Expand/collapse: remove data from model', key_button_menu_remove_data)
    if button_remove_data:
        file_session[key_button_menu_remove_data] = not file_session[key_button_menu_remove_data]
    if file_session[key_button_menu_remove_data]:

        select_train_or_predict_remove = st.selectbox('Select which data you want to remove', options=['', training_data_option, predict_data_option])

        if select_train_or_predict_remove == training_data_option:
            select_train_data_to_remove = st.selectbox('Select a source of data to be removed', options=['']+p.training_data_sources)
            if select_train_data_to_remove:
                st.markdown(f'Are you sure you want to remove the following data from the training data set: {select_train_data_to_remove}')
                st.markdown(f'NOTE: upon removing the data, the model will need to be rebuilt.')
                confirm = st.button('Confirm')
                if confirm:
                    with st.spinner(f'Removing {select_train_data_to_remove} from training data set...'):
                        p = p.remove_train_data_source(select_train_data_to_remove).save(os.path.dirname(pipeline_path))
                    file_session[key_button_menu_remove_data] = False
                    st.success(f'{select_train_data_to_remove} data successfully removed!')

        if select_train_or_predict_remove == predict_data_option:
            select_predict_option_to_remove = st.selectbox('Select a source of data to be removed', options=['']+p.predict_data_sources)
            if select_predict_option_to_remove:
                st.markdown(f'Are you sure you want to remove the following data from the predicted/analyzed data set: {select_predict_option_to_remove}')
                confirm = st.button('Confirm')
                if confirm:
                    with st.spinner(f'Removing {select_predict_option_to_remove} from predict data set'):
                        p.remove_predict_data_source(select_predict_option_to_remove).save(os.path.dirname(pipeline_path))
                    st.success(f'{select_predict_option_to_remove} data was successfully removed!')
                    file_session[key_button_menu_remove_data] = False
                st.markdown('')

        st.markdown('')

    ###

    ### Menu button: rebuilding model ###
    button_see_rebuild_options = st.button('Expand/Collapse: Review Model Parameters & Rebuild Model', key_button_see_rebuild_options)
    if button_see_rebuild_options:  # Click button, flip state
        file_session[key_button_see_rebuild_options] = not file_session[key_button_see_rebuild_options]
    if file_session[key_button_see_rebuild_options]:  # Now check on value and display accordingly
        st.markdown('------------------------------------------------------------------------------------')
        st.markdown('## Model Parameters')
        # TODO: Low/Med: implement variable feature selection
        # st.markdown(f'### Select features')
        # st.multiselect('select features', p.all_features, default=p.all_features)  # TODO: develop this feature selection tool!
        # st.markdown('---')

        st.markdown('### Gaussian Mixture Model Parameters')
        slider_gmm_n_components = st.slider(f'GMM Components (clusters)', value=p.gmm_n_components, min_value=2, max_value=40, step=1)
        # TODO: low: add GMM: probability = True
        # TODO: low: add: GMM: n_jobs = -2

        ### Other model info ###
        st.markdown('### Other model information')
        input_k_fold_cross_val = st.number_input(f'Set K for K-fold cross validation', value=int(p.cross_validation_k), min_value=2, format='%i')  # TODO: low: add max_value= number of data points (for k=n)?
        # TODO: med/high: add number input for % holdout for test/train split

        # Hack solution: specify params here so that the variable exists even though advanced params section not opened.

        st.markdown('')
        ### Advanced Parameters ###
        button_see_advanced_options = st.button('Expand: advanced parameters')
        if button_see_advanced_options:
            file_session[key_button_see_advanced_options] = not file_session[key_button_see_advanced_options]
        if file_session[key_button_see_advanced_options]:
            st.markdown('### Advanced model options. Do not change things here unless you know what you are doing!')
            st.markdown('If you collapse the advanced options menu, all changes made will revert unless you rebuild the model.')
            # See advanced options for model
            st.markdown('### Advanced TSNE Parameters')
            input_tsne_early_exaggeration = st.number_input(f'TSNE: early exaggeration', min_value=0., max_value=100., value=p.tsne_early_exaggeration, step=0.1, format='%.2f')
            input_tsne_n_components = st.slider(f'TSNE: n components/dimensions', value=p.tsne_n_components, min_value=1, max_value=10, step=1, format='%i')
            input_tsne_n_iter = st.number_input(label=f'TSNE n iterations', value=p.tsne_n_iter, min_value=250, max_value=5_000)
            # TODO: n_jobs: n_jobs=-1: all cores being used, set to -2 for all cores but one.
            st.markdown(f'### Advanced GMM parameters')
            input_gmm_reg_covar = st.number_input(f'GMM "reg. covariance" ', value=p.gmm_reg_covar, format='%f')
            input_gmm_tolerance = st.number_input(f'GMM tolerance', value=p.gmm_tol, min_value=1e-10, max_value=50., step=0.1, format='%.2f')
            input_gmm_max_iter = st.number_input(f'GMM max iterations', min_value=1, max_value=100_000, value=p.gmm_max_iter, step=1, format='%f')
            input_gmm_n_init = st.number_input(f'GMM "n_init" ("Number of initializations to perform. the best results is kept")  . It is recommended that you use a value of 20', value=p.gmm_n_init, step=1, format="%i")
            st.markdown('### Advanced SVM Parameters')
            ### SVM ###
            input_svm_c = st.number_input(f'SVM C', value=p.svm_c, format='%.2f')
            input_svm_gamma = st.number_input(f'SVM gamma', value=p.svm_gamma, format='%.2f')
        else:
            input_tsne_early_exaggeration, input_tsne_n_components = p.tsne_early_exaggeration, p.tsne_n_components
            input_tsne_n_iter, input_gmm_reg_covar, input_gmm_tolerance = p.tsne_n_iter, p.gmm_reg_covar, p.gmm_tol
            input_gmm_max_iter, input_gmm_n_init = p.gmm_max_iter, p.gmm_n_init
            input_svm_c, input_svm_gamma = p.svm_c, p.svm_gamma

        ###

        st.markdown('')

        st.markdown(f'Note: changing the above parameters without rebuilding the model will have no effect.')

        # Save above info + rebuild model
        st.markdown('## Rebuild model with new parameters above?')
        button_rebuild_model = st.button('I want to rebuild model with new parameters', key_button_rebuild_model)
        if button_rebuild_model: file_session[key_button_rebuild_model] = not file_session[key_button_rebuild_model]
        if file_session[key_button_rebuild_model]:  # Rebuild model button was clicked
            st.markdown('Are you sure?')
            button_confirmation_of_rebuild = st.button('Confirm', key_button_rebuild_model_confirmation)
            if button_confirmation_of_rebuild:
                file_session[key_button_rebuild_model_confirmation] = True
            if file_session[key_button_rebuild_model_confirmation]:  # Rebuild model confirmed.
                with st.spinner('Rebuilding model...'):
                    model_vars = {
                        'gmm_n_components': slider_gmm_n_components,
                        'cross_validation_k': input_k_fold_cross_val,
                        # Advanced opts
                        'tsne_early_exaggeration': input_tsne_early_exaggeration,
                        'tsne_n_components': input_tsne_n_components,
                        'tsne_n_iter': input_tsne_n_iter,

                        'gmm_reg_covar': input_gmm_reg_covar,
                        'gmm_tol': input_gmm_tolerance,
                        'gmm_max_iter': input_gmm_max_iter,
                        'gmm_n_init': input_gmm_n_init,

                        'svm_c': input_svm_c,
                        'svm_gamma': input_svm_gamma,

                    }

                    # TODO: HIGH: make sure that model parameters are put into Pipeline before build() is called.
                    p = p.set_params(**model_vars)
                    p = p.build(True, True).save(os.path.dirname(pipeline_path))
                st.success(f'Model was successfully re-built! Refresh the page to see changes.')
                file_session[key_button_rebuild_model_confirmation] = False
        st.markdown('----------------------------------------------------------------------------------------------')

    ###


    ###

    st.markdown('--------------------------------------------------------------------------------------------------')

    return see_model_diagnostics(p, pipeline_path)


def see_model_diagnostics(p, pipeline_file_path):
    ######################################### MODEL DIAGNOSTICS ########################################################
    st.markdown(f'## Model Diagnostics')
    st.markdown(f'See GMM distributions according to TSNE-reduced feature dimensions // TODO: make this shorter.')
    ### View Histogram for assignment distribution
    st.markdown(f'This section is a work-in-progress. Opening a graph in this section is very volatile and there is '
            f'high chance that by opening a graph the streamlit will crash. This is being worked on!')
    st.markdown(f'View distribution of assignments')
    button_view_assignments_distribution = st.button(f'View assignment distribution')
    if button_view_assignments_distribution:
        file_session[key_button_view_assignments_distribution] = not file_session[key_button_view_assignments_distribution]
    if file_session[key_button_view_assignments_distribution]:
        if p.is_built:
            fig, ax = p.get_plot_svm_assignments_distribution()
            st.pyplot(fig)
        else:
            st.info('There are no assignment distributions available for display because '
                    'the model is not currently built.')

    ###

    # View 3d Plot
    gmm_button = st.button('Pop out window of cluster/assignment distribution')  # TODO: low: phrase this button better?
    if gmm_button:
        if p.is_built:
            try:
                p.plot_assignments_in_3d(show_now=True)
            except ValueError:
                st.error('Cannot plot cluster distribution since the model is not currently built.')
        else:
            st.info('A 3d plot of the cluster distributions could not be created because '
                    'the model is not built. ')
    ###
    st.markdown('--------------------------------------------------------------------------------------------------')

    return review_behaviours(p, pipeline_file_path)


def review_behaviours(p, pipeline_file_path):
    """"""
    ### SIDEBAR

    ### MAIN

    ## Review Behaviour Example Videos ##
    st.markdown(f'## Behaviour clustering review')

    example_videos_file_list: List[str] = [video_file_name for video_file_name in os.listdir(config.EXAMPLE_VIDEOS_OUTPUT_PATH) if video_file_name.split('.')[-1] in valid_video_extensions]  # # TODO: low/med: add user intervention on default path to check?
    videos_dict: Dict[str: str] = {**{'': ''}, **{video_file_name: os.path.join(config.EXAMPLE_VIDEOS_OUTPUT_PATH, video_file_name) for video_file_name in example_videos_file_list}}

    video_selection: str = st.selectbox(label=f"Select video to view. Total videos found in folder '{config.EXAMPLE_VIDEOS_OUTPUT_PATH}': {len(videos_dict)-1}", options=list(videos_dict.keys()))
    if video_selection:
        try:
            st.video(get_video_bytes(videos_dict[video_selection]))
        except FileNotFoundError as fe:
            st.error(FileNotFoundError(f'No example behaviour videos were found at this time. Try '
                                       f'generating them at check back again after. // '
                                       f'DEBUG INFO: path checked: {config.EXAMPLE_VIDEOS_OUTPUT_PATH} // {repr(fe)}'))
    ###

    st.markdown('')

    ###

    st.markdown('')

    ### Create new example videos ###
    button_create_new_ex_videos = st.button(f'Expand/collapse: Create new example videos // TODO: elaborate', key=key_button_show_example_videos_options)
    if button_create_new_ex_videos:
        file_session[key_button_show_example_videos_options] = not file_session[key_button_show_example_videos_options]
    if file_session[key_button_show_example_videos_options]:
        st.markdown(f'Fill in variables for making new example videos of behaviours')
        select_data_source = st.selectbox('Select a data source', options=['']+p.training_data_sources)
        input_video = st.text_input(f'Input path to corresponding video relative to selected data source', value=config.BSOID_BASE_PROJECT_PATH)
        file_name_prefix = st.text_input(f'File name prefix. This helps us differentiate between example videos. OK to leave blank. ')
        number_input_output_fps = st.number_input(f'Output FPS for example videos', value=8, min_value=1)
        number_input_max_examples_of_each_behaviour = st.number_input(f'Maximum number of videos created for each behaviour', value=5, min_value=1)
        number_input_min_rows = st.number_input(f'Min # of data rows required for a detection to occur', value=1, min_value=1, max_value=10_000)
        number_input_frames_leadup = st.number_input(f'min # of rows of data after/before behaviour has occurred that lead up // todo: precision', value=0, min_value=0)

        st.markdown('')

        ### Create new example videos button
        st.markdown('#### When the variables above are filled out, press the "Confirm" button below to create new example videos')
        st.markdown('')
        button_create_new_ex_videos = st.button('Confirm', key=key_button_create_new_example_videos)
        if button_create_new_ex_videos:
            is_error_detected = False
            ### Check for errors (display as many errors as necessary for redress)
            # File name prefix check
            if check_arg.has_invalid_chars_in_name_for_a_file(file_name_prefix):
                is_error_detected = True
                invalid_name_err_msg = f'Invalid file name submitted. Has invalid char. Prefix="{file_name_prefix}"'
                st.error(ValueError(invalid_name_err_msg))
            # Input video check
            if not os.path.isfile(input_video):
                is_error_detected = True
                err_msg = f'Video file not found at path "{input_video}" '
                st.error(FileNotFoundError(err_msg))
            # Continue if good.
            if not is_error_detected:
                with st.spinner('Creating new videos...'):
                    p = p.make_behaviour_example_videos(
                        select_data_source,
                        input_video,
                        file_name_prefix,
                        min_rows_of_behaviour=number_input_min_rows,
                        max_examples=number_input_max_examples_of_each_behaviour,
                        output_fps=number_input_output_fps,
                        num_frames_leadup=number_input_frames_leadup,
                    )
                st.success(f'Example videos created!')  # TODO: low: improve message
        st.markdown('--------------------------------------------------------------------------------------')

    ###

    ### Review labels for behaviours ###
    button_review_assignments_is_clicked = st.button('Expand/collapse: review behaviour/assignments labels', key_button_review_assignments)
    if button_review_assignments_is_clicked:  # Click button, flip state
        file_session[key_button_review_assignments] = not file_session[key_button_review_assignments]
    if file_session[key_button_review_assignments]:  # Depending on state, set behaviours to assignments
        if not p.is_built:
            st.info('The model has not been built yet, so there are no labeling options available.')
        else:
            ### View all assignments
            st.markdown(f'#### All changes entered save automatically. After all changes, refresh page to see changes.')
            for a in p.unique_assignments:
                file_session[str(a)] = p.get_assignment_label(a)
                existing_behaviour_label = p.get_assignment_label(a)
                existing_behaviour_label = existing_behaviour_label if existing_behaviour_label is not None else '(No behaviour label assigned yet)'
                text_input_new_label = st.text_input(f'Add behaviour label to assignment # {a}', value=existing_behaviour_label, key=f'key_new_behaviour_label_{a}')
                if text_input_new_label != existing_behaviour_label:
                    p = p.set_label(a, text_input_new_label).save(os.path.dirname(pipeline_file_path))

    ###
    # ### Review labels for behaviours ###
    # button_review_assignments_is_clicked = st.button('Expand/collapse: review behaviour/assignments labels', key_button_review_assignments)
    # if button_review_assignments_is_clicked:  # Click button, flip state
    #     file_session[key_button_review_assignments] = not file_session[key_button_review_assignments]
    # if file_session[key_button_review_assignments]:  # Depending on state, set behaviours to assignments
    #     if not p.is_built:
    #         st.info('The model has not been built yet, so there are no labeling options available.')
    #     else:
    #         ### View all assignments
    #         st.markdown(f'#### All changes entered save automatically. After all changes, refresh page to see changes.')
    #         for a in p.unique_assignments:
    #             file_session[str(a)] = p.get_assignment_label(a)
    #             existing_behaviour_label = p.get_assignment_label(a)
    #             existing_behaviour_label = existing_behaviour_label if existing_behaviour_label is not None else '(No behaviour label assigned yet)'
    #             text_input_new_label = st.text_input(f'Add behaviour label to assignment # {a}', value=existing_behaviour_label, key=f'key_new_behaviour_label_{a}')
    #             if text_input_new_label != existing_behaviour_label:
    #                 p = p.set_label(a, text_input_new_label).save(os.path.dirname(pipeline_file_path))

    ###

    st.markdown('')

    ### Label an entire video ###
    button_menu_label_entire_video = st.button('Expand/collapse: Use model to label to entire video', key=key_button_menu_label_entire_video)
    if button_menu_label_entire_video:
        file_session[key_button_menu_label_entire_video] = not file_session[key_button_menu_label_entire_video]
    if file_session[key_button_menu_label_entire_video]:
        st.markdown('')
        st.markdown(f'(WIP) Menu to label entire video')
        input_video_to_label = st.text_input('Input path to video which is to be labeled')
        selected_data_source = st.selectbox('Select a data source to use as the label set for '
                                            'specified video (WIP: rewrite this line better :) )',
                                            options=['']+p.training_data_sources+p.predict_data_sources)

        button_create_labeled_video = st.button('Create labeled video')
        if button_create_labeled_video:
            if not selected_data_source:
                st.error('An invalid data source was selected. Please change the data source and try again.')
                st.stop()
            with st.spinner('(WIP: Video creation efficiency still being worked on) Creating labeled video now. This could take several minutes...'):  # TODO: High
                # app.sample_runtime_function(3)
                p.make_video(input_video_to_label, 'magicvariable_in_streamlit')  # TODO: med: add other options like adding output path and output fps?
            st.success('Success! Video was created at: TODO: get video out path')
        st.markdown('---------------------------------------------------------------------------------------')

    ###

    return display_footer(p, pipeline_file_path)


#

def display_footer(p, *args, **kwargs):
    """ Footer of Streamlit page """

    return p


# Accessory functions #

def line_break():
    """ Displays a horizontal line-break on the Streamlit page. """
    st.markdown('---')


def get_video_bytes(path_to_video):
    """  """
    check_arg.ensure_is_file(path_to_video)
    with open(path_to_video, 'rb') as video_file:
        video_bytes = video_file.read()
    return video_bytes


# Misc.

def example_of_value_saving():
    session_state = streamlit_session_state.get(**{'TestButton1': False, 'TestButton2': False})
    st.markdown("# [Title]")
    button1_is_clicked = st.button('Test Button 1', 'TestButton1')
    st.markdown(f'Pre button1: Button 1 session state: {session_state["TestButton1"]}')
    if button1_is_clicked:
        session_state['TestButton1'] = not session_state['TestButton1']
    if session_state['TestButton1']:
        st.markdown(f'In button1: Button 1 session state: {session_state["TestButton1"]}')
        button2_is_clicked = st.button('Test Button 2', 'TestButton2')
        if button2_is_clicked:
            session_state['TestButton2'] = not session_state['TestButton2']
        if session_state['TestButton2']:
            line_break()
            st.markdown('Button 2 pressed')


# Main: likely to be deleted later.

if __name__ == '__main__':
    # Note: this import only necessary when running streamlit onto this file specifically rather than
    #   calling `streamlit run main.py streamlit`
    BSOID_project_path = os.path.dirname(os.path.dirname(__file__))
    if BSOID_project_path not in sys.path:
        sys.path.insert(0, BSOID_project_path)
    # home()
    example_of_value_saving()
