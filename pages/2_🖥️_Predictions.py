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


st.set_page_config(page_title="ML Predictions", page_icon="🧠", layout="wide")

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
