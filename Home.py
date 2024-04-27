import os

from geopy.geocoders import Nominatim
import pydeck as pdk
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st
import sklearn as sk
import joblib
from sklearn.preprocessing import LabelEncoder
import time
import numpy as np
import math
import base64
import joblib
from sklearn import preprocessing
import pickle
import pydeck as pdk

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")
# importing geopy library and Nominatim class

st.title("Projected impacts of climate change on your local temprature and take action today! ")



with st.sidebar:
    st.write("Author Information:")

    with st.spinner("Loading..."):
        time.sleep(5)
    st.success(
        "Quang-Loc Nguyen*, and other researchers in SP Jain School")
    st.success(
        "Corresponding author: Quang-Loc Nguyen. Email: loc.bs21don038@spjain.org")
_LOREM_IPSUM = """
It's time to take action on one of the most pressing issues of our time: ***climate change***.

Climate change affects us all, from the air we breathe to the food we eat, and its impacts are becoming increasingly evident in our everyday lives. But there is hope. By raising awareness and taking collective action, we can mitigate its effects and build a more sustainable future for generations to come.

At **SP Jain School of Global Management**, we're dedicated to promoting knowledge and awareness about climate change and empowering individuals like you
to make a difference. Our interactive platform offers a wealth of resources, tools, and opportunities for you to get involved:

- Explore educational content to deepen your understanding of climate science and its impacts.
- Calculate temprature rise, affected crops productions, and discover practical tips for adapting it in your local community.
- Connect with a community of like-minded individuals, share ideas, and join forces to tackle climate challenges together.
- Stay informed with the latest news, updates, and success stories from the front lines of climate action.
- Take meaningful action through volunteer opportunities, advocacy campaigns, and sustainable initiatives.
Together, we can be part of the solution. Join us in the fight against climate change and together, let's create a more resilient and sustainable future for all!
"""


def stream_data():
    for word in _LOREM_IPSUM.split(" "):
        yield word + " "
        time.sleep(0.02)


x = st.text_input("How can I call you? An Inspiring Environmental Activist!")
if x:  # This will check if 'x' is not empty
    st.markdown(
        f''':green[Dear ***{x}***, Are you concerned about the future of our planet? Do you want to make a positive impact on the world around you?]'''
    )
    container = st.container(border=True)
    container.write_stream(stream_data)
# Author: Quang-Loc Nguyen
st.image('https://climate.nasa.gov/internal_resources/2710/Effects_page_triptych.jpeg',
         caption='Climate Change Consequences')


# audio_file = open('En-Climate_change.ogg', 'rb')
# audio_bytes = audio_file.read()
# st.audio(audio_bytes, format='audio/ogg')

sample_rate = 44100  # 44100 samples per second
seconds = 2  # Note duration of 2 seconds
frequency_la = 440  # Our played note will be 440 Hz
# Generate array with seconds*sample_rate steps, ranging between 0 and seconds
t = np.linspace(0, seconds, seconds * sample_rate, False)
# Generate a 440 Hz sine wave
note_la = np.sin(frequency_la * t * 2 * np.pi)


st.subheader("What are current research articles about the topic?")
# Your original DataFrame
data_df = pd.DataFrame({
    "Articles": [
        "Financial Assessment under Climate Change",
        "Urban Planning",
        "Mitigation due to Climate Change",
        "Climate Change Finance for Business",
    ],
    "Official Link": [
        "https://www.mdpi.com/1911-8074/15/11/542",
        "https://www.mdpi.com/2413-8851/7/2/46",
        "https://www.mdpi.com/1660-4601/19/19/12233",
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4351342",
    ],
})

# Convert author URLs to HTML links
data_df['Official Link'] = data_df['Official Link'].apply(
    lambda x: f"<a href='{x}' target='_blank'>Open Reading</a>")

# Display the DataFrame as HTML
st.write(data_df.to_html(escape=False), unsafe_allow_html=True)
