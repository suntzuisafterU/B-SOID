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


DLC_PROJECT_PATH = 'test'
app_model_data_filename = 'appmodeldatafiilename'
valid_video_extensions = {'avi', 'mp4', }

########################################################################################################################


@st.cache(persist=True)
def get_example_vids(path):
    with open(path, 'rb') as video_file:
        video_bytes = video_file.read()
    return video_bytes


def home(*args, **kwargs):
    """
    Designated home page when streamlit is run for BSOID
    """


    st.markdown('---')

    ### Sidebar ###
    # Add a selectbox to the sidebar:
    some_selection = st.sidebar.selectbox(
        label='How would you like to be contacted?',
        options=('Email', 'Home phone', 'Mobile phone', )
    )

    # Add a slider to the sidebar:
    random_slider = st.sidebar.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0)
    )





    st.markdown('---')

    ### Main

    st.markdown('# B-SOiD streamlit app')
    st.markdown('---')
    st.selectbox(
        label='Start a new project or load an existing one?',
        options=('', 'Start new', 'Load existing'),
        key='StartProject',
    )


    # Load/create project
    st.markdown('## Load/create new project')
    st.markdown('### Further explanation of what is going on')
    st.markdown('---')

    # Do analysis
    st.markdown('')
    with st.spinner('Waiting on analysis to be done...'):
        # Do computations
        time.sleep(1)
        pass
    st.success('Done!')
    st.exception(RuntimeError('st.exception(): Test runtime error for fun'))




    st.markdown('---')

    # Do other stuff
    st.help('?')
    st.markdown('---')
    st.info('st.info(): You can change params or you can ...')

    #
    st.markdown('---')

    # Extra section for fun
    ### Random extra section ###
    st.markdown('# Random extra section')

    BASE_PATH = st.text_input('Enter a DLC project "BASE PATH":', DLC_PROJECT_PATH)
    try:
        os.listdir(BASE_PATH)
        st.markdown(f'You have selected **{BASE_PATH}** as your root directory for training/testing sub-directories.')
    except FileNotFoundError:
        st.error('No such directory')
        # return


    res = st.button('Push button', key='RandomPushButton')
    st.write(f'Button value = {res}')  # PUsh to change to true

    text_input = st.text_input('test', 'default_text')
    submit_text = st.button('Submit', key='SubmitTextAtTop')
    if submit_text:
        st.write(f'Text input value: {text_input}')

    st.markdown('# Sample videos')
    demo_videos = {x: os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'examples', x) for x in
                   os.listdir(os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'examples'))
                   if x.split('.')[-1] in valid_video_extensions}
    vid = st.selectbox("Notable examples, please contribute!", list(demo_videos.keys()), 0)
    st.video(get_example_vids(demo_videos[vid]))
    st.markdown('---')

    return
