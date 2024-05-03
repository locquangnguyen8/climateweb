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

from geopy.geocoders import Nominatim
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import pickle
from matplotlib import pyplot as plt
import sklearn as sk
import time
import math
import joblib
from sklearn import preprocessing
import pydeck as pdk

# importing geopy library and Nominatim class

st.title("Projected impacts of climate change on your local temprature and take action today! ")



with st.sidebar:
    st.write("Author Information:")

    with st.spinner("Loading..."):
        time.sleep(5)
    st.success(
        "Quang-Loc Nguyen*, and other researchers in SP Jain Global")
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

st.subheader("How to use this website:")
st.write("In the left pannel of the website, you can visite different pages to explore information. The **Visualizations** page is designed to examine the dataset and features. You also can input your local temperature and location into the **Predictions** page, and the Random Forest model generates predictions for climate change in your area. The website goes beyond mere prediction, offering practical recommendations for adapting to climate change in **Recommendations** page. The **Appendix** is aimed to provide additional information about the author and the dataset.")
st.subheader("This website is designed for:")
st.write("- Predict Temperature Changes: Utilizing the random forest model, the project predicts temperature fluctuations in various regions based on user inputs, such as local temperature and location.")
st.write("- Assess Impact on Crops: The project examines how these temperature changes affect major crops like maize, rice, soybean, and wheat, offering detailed analysis of potential yield reductions or other crop-specific impacts.")
st.write("- Calculate Economic Loss: Beyond crop impacts, the project estimates the economic loss that may result from the changing climate, providing an essential context for understanding broader societal impacts.")
st.write("- Provide Adaptation Recommendations: The project provides tailored recommendations to help local communities adapt to climate change, such as altering farming practices, implementing water management strategies, or diversifying crops.")
st.write("- Raise Awareness and Influence Policy: By making this tool available to the public, the project aims to raise awareness about climate change, encouraging local government agencies to create policies that address climate-related issues affecting their communities.")
st.subheader("Some articles about the climate change phenomena:")
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
