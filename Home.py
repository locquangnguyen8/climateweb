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



st.header("Local Area Temprature Prediction")
st.write("In this part, the robust machine learning will help you project the local temprature rise.")

Global_delta_T_from_2005 = st.number_input(
    'Insert Global Temperature Rise from 2005: ')
# calling the Nominatim tool and create Nominatim class
geolocator = Nominatim(user_agent="App_name")
p = st.text_input("Enter your address")

t = p.split(' ')[-1]
# st.write(t)
# st.link_button("Find Latitude and Longitude",
# url="https://www.latlong.net/")
# st.markdown("""
# <a style='display: block; text-align: center; background-color: #4CAF50; color: white; padding: 14px 20px; margin: 10px 0; border: none; cursor: pointer; width: 100%;'
# href='https://www.latlong.net/' target='_blank'>Find Latitude and Longitude</a>
# """, unsafe_allow_html=True)

Climate_scenario = st.selectbox('Pick a Climate Scenario: ', ['RCP8.5', 'RCP4.5', 'RCP6.0', 'RCP2.6', 'A2', 'B2',
                                                              'Others', 'A1B', 'B1', 'A1FI, B1'])
_multi2 = '''This is climate-scenario-based simulations: “RCP”, “RCP2.6”, “RCP6.0”, “RCP4.5”, “RCP8.5”, “CMIP5”, and “CMIP6”. 
RCP stands for the Representative Concentration Pathways, and each RCP corresponds to a greenhouse gas concentration trajectory describing different 
future greenhouse gas emission levels. The number followed by RCP is the level of radiative forcing (Wm−2) 
reached at the end of the 21st century, 
which increases with the volume of greenhouse gas emitted to the atmosphere.
'''


def stream_data2():
    for word in _multi2.split(" "):
        yield word + " "
        time.sleep(0.02)


# if st.button("Climate Scenario(s) explaination"):
    # st.write_stream(stream_data2)
expander = st.expander("Climate Scenario explaination")
expander.write_stream(stream_data2)
# st.link_button("Find out more Pepresentative Concentration Pathways",
# url="https://link.springer.com/article/10.1007/s10584-011-0148-z")
st.markdown("""
    <a style='display: block; text-align: center; background-color: #4CAF50; color: white; padding: 14px 20px; margin: 10px 0; border: none; cursor: pointer; width: 100%;' 
    href='https://link.springer.com/article/10.1007/s10584-011-0148-z' target='_blank'>Representative Concentration Pathways</a>
    """, unsafe_allow_html=True)
Future_Mid_point = st.slider('Choose Future Year: ', 2000, 2110, 2024)
Current_Average_Temperature_area_weighted = st.number_input(
    'Insert Current Average Temperature: ')
Country = t
Time_slice = st.selectbox('Select Time Slice: ', ['EC', 'MC', 'NF'])
_multi1 = '''This is the climate change adaptation for crops production sections. If the local government implements these options as ways to adapt crops to climate change, Please choose below adaptation options, which are categorised into fertiliser, irrigation, cultivar, soil organic matter management, planting time, tillage, and others. 
Specifically, in the fertiliser option, if the amount and timing of fertiliser application are changed from the current conventional method, it is considered as adaptation. 
In the irrigation option, if the simulation program determines the irrigation scheduling based on the crop growth, 
climatic and soil moisture conditions, it can be deemed as adaptation because the management is adjusted to future climatic conditions. 
If rainfed and irrigated conditions are simulated separately, it should not consider irrigation as an adaptation. 
'''


def stream_data1():
    for word in _multi1.split(" "):
        yield word + " "
        time.sleep(0.02)


# if st.button("Find out Crop Mitigation"):
    # st.write_stream(stream_data1)
expander = st.expander("Crop Mitigation")
expander.write_stream(stream_data1)

Fertiliser = st.radio(
    'Has Fertiliser Applicable in the Local Community?', ['Yes', 'No'])
Irrigation = st.radio(
    'Has Irrigation Deployed in the Local Community?', ['Yes', 'No'])
Cultivar = st.radio(
    'Has Cutltivar Deleloped in the Local Community?', ['Yes', 'No'])
Adaptation_type = st.selectbox('Define Adaption Type: ', [
                               'Combined', 'Cultivar', 'Fertiliser', 'Irrigation', 'No', 'Others'])
location = geolocator.geocode(p)
# st.write((location.latitude, location.longitude))
latitude = location.latitude
longitude = location.longitude
columns = [
    'Global_delta_T_from_2005',
    'latitude',
    'longitude',
    'Climate_scenario',
    'Future_Mid-point',
    'Current_Average_Temperature_(dC)_area_weighted',
    'Country',
    'Time_slice',
    'Fertiliser',
    'Irrigation',
    'Cultivar',
    'Adaptation_type']

data = [
    Global_delta_T_from_2005,
    latitude,
    longitude,
    Climate_scenario,
    Future_Mid_point,
    Current_Average_Temperature_area_weighted,
    Country,
    Time_slice,
    Fertiliser,
    Irrigation,
    Cultivar,
    Adaptation_type]

if all(data):
    df = pd.DataFrame([data], columns=columns)
    st.write(df)  # Display the DataFrame

    # Load the trained model
    model = joblib.load("model.joblib")
    # Apply Label Encoder on specific columns
    label_encoder_columns = ['Country', 'Time_slice', 'Fertiliser',
                             'Irrigation', 'Cultivar', 'Adaptation_type', 'Climate_scenario']
    le = {column: LabelEncoder().fit(df[column])
          for column in label_encoder_columns}

    # Transform user inputs using label encoder
    transformed_data = [le[column].transform(
        [value])[0] if column in le else value for column, value in zip(columns, data)]
    transformed_data = pd.DataFrame([transformed_data], columns=columns)

    # Make predictions using the loaded model
    prediction = model.predict(transformed_data)
    x = round(prediction[0], 2)
    # Display the prediction
    st.write(
        f"**The {p}'s temprature in {Future_Mid_point} will be increased by:**", x)
# calculate prediction interval
interval = 1.96 * 0.00462
lower, upper = round(x - interval, 3), round(x + interval, 3)
st.write(
    f"*The 95% confidence interval of the predicted temperature range from {lower} to {upper}*")
st.markdown('''
    :red["**The local temprature rise directly affects crop growth and yield!**".]''')

st.write("Summary statistics of climate change impacts (average) on four major crops expresses as ")
st.write("**Per Decade Impact**:")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Maize", "−3.9")
col2.metric("Rice", "-1.4")
col3.metric("Soybean", "-2.6")
col4.metric("Wheat", "-1.8")

st.write("**Per Degree Impact (% °C−1)**:")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Maize", "−13.5")
col2.metric("Rice", "-2.6")
col3.metric("Soybean", "-8.8")
col4.metric("Wheat", "-5.6")

st.write('Interpretation in Mathematic Equation:')
st.latex(r'''
    (-13.5*Maize - 2.6*Rice - 8.8*Soybean - 5.6*Wheat) * Degree Rise
    ''')

st.image('https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41597-022-01150-7/MediaObjects/41597_2022_1150_Fig4_HTML.png?as=webp',
         caption='Climate change impacts (% of yield change from the baseline period) on four crops without adaptation under RCP8.5')

st.write('***Crop production lost in your local community:***')
col1, col2, col3, col4 = st.columns(4)
maize = col1.number_input("Current Maize Production")
rice = col2.number_input("Current Rice Production")
soybean = col3.number_input("Soybean Production")
wheat = col4.number_input("Current Wheat Production")

# Calculate the loss
lostmaize = round(x * (-13.5) * maize * 0.01, 2)
lostrice = round(x * (-2.6) * rice * 0.01, 2)
lostsoy = round(x * (-8.8) * soybean * 0.01, 2)
lostwheat = round(x * (-5.6) * wheat * 0.01, 2)

# st.write("Projected Lost due to Climate Change")
# col1, col2, col3, col4 = st.columns(4)
# col1.metric(label="The Maize Loss:", value=f"{lostmaize:,}")
# col2.metric(label="The Rice Loss:", value=f"{lostrice:,}")
# col3.metric(label="The Soybean Loss:", value=f"{lostsoy:,}")
# col4.metric(label="The Wheat Loss:", value=f"{lostwheat:,}")

# Calculate the remaining
projectmaize = round(maize + lostmaize, 2)
projectrice = round(rice + lostrice, 2)
projectsoy = round(soybean + lostsoy, 2)
projectwheat = round(wheat + lostwheat, 2)

# st.markdown(''':red[Projected Crops Production]''')
# col1, col2, col3, col4 = st.columns(4)
# col1.metric(label="The Maize Projected:", value=f"{projectmaize:,}")
# col2.metric(label="The Rice Projected:", value=f"{projectrice:,}")
# col3.metric(label="The Soybean Projected:", value=f"{projectsoy:,}")
# col4.metric(label="The Wheat Projected:", value=f"{projectwheat:,}")

st.write(
    f"**The {p}'s Crops Production in {Future_Mid_point} Prediction and Loss:**")


col1, col2, col3, col4 = st.columns(4)
col1.metric("Maize", projectmaize, lostmaize)
col2.metric("Rice", projectrice, lostrice)
col3.metric("Soybean", projectsoy, lostsoy)
col4.metric("Wheat", projectwheat, lostwheat)

st.write('**Economy Loss Projection (the U.S dollar):**')
col1, col2, col3, col4 = st.columns(4)
h = col1.number_input("Price of Maize per unit")
j = col2.number_input("Price of Rice per unit")
k = col3.number_input("Price of Soybean per unit")
l = col4.number_input("Price of Wheat per unit")


# Calculate the remaining
mmaize = h * lostmaize
mrice = j * lostrice
msoy = k * lostsoy
mwheat = l * lostwheat

ttlost = mmaize + mrice + msoy + mwheat
ttotal = round(-ttlost, 2)
st.markdown(f'''
    ## :red[**The {p}'s Economy in {Future_Mid_point} will loss around ${ttotal} !**]''')


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
df9 = pd.read_csv("Projected_impacts_datasheet_11.24.2021.csv")
tab1, tab2 = st.tabs(["🗃 Data", "📈 Author"])
tab1.subheader("Predictive Modelling")
tab1.write(
    "**In this study, Random Forest Regressor is employed to predict the data.**")
tab1.write(
    "500 trees are used as a parameter that specifies the number of decision trees. The seed for the random number generator used by the algorithm is 42")
tab1.write("The Mean Square Error of the algorithm is 0.0057")

tab1.subheader("Reference Data")

tab1.write("**Data that has been used to train the predictive model:**")
tab1.dataframe(df9)
# Using markdown to create a link that looks like a button
tab2.subheader("Author Profile")
tab2.write("**Quang-Loc Nguyen** is a Research Assistant in the SP Jain School of Global Management. His focuses are, but not limited to, Climate Change, Economics, Social Science, and Data Science. **Loc** has published widely in top-tier journals such as Nature: *npj Climate Action*, SPRINGER: *Journal of Environmental Studies and Sciences*, the ABDC list: *Journal of Risk and Financial Management*, and Q1, Q2 ranked journals.")
tab2.write("")
tab2.markdown("""
    <a style='display: block; text-align: center; background-color: #4CAF50; color: white; padding: 14px 20px; margin: 10px 0; border: none; cursor: pointer; width: 100%;' 
    href='https://scholar.google.com/citations?user=y-vi274AAAAJ&hl=en' target='_blank'>Author Profile</a>
    """, unsafe_allow_html=True)
st.subheader("Reference")
container = st.container(border=True)
container.write("Hasegawa, T., Wakatsuki, H., Ju, H. et al. A global dataset for the projected impacts of climate change on four major crops. Sci Data 9, 58 (2022). https://doi.org/10.1038/s41597-022-01150-7")
container.write("Minh-Hoang Nguyen, Minh-Phuong Thi Duong, **Quang-Loc Nguyen**, Viet-Phuong La, Vuong-Quan Hoang. In search of value: the intricate impacts of benefit perception, knowledge, and emotion about climate change on marine protection support. Journal of Environmental Studies and Sciences (2024). https://doi.org/10.1007/s13412-024-00902-8")
container.write("**Nguyen, Quang Loc** and Nguyen, Minh-Hoang and La, Viet-Phuong and Bhatti, Ishaq and Vuong, Quan Hoang, Enterprise's Strategies to Improve Financial Capital under Climate Change Scenario – Evidence of the Leading Country. Nature, *npj Climate Action* (2024).")
