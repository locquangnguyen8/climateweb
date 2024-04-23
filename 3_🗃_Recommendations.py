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


_multi0 = '''Adapting agriculture to ensure crop production in the face of climate change requires a combination of strategies that enhance resilience, conserve resources, and optimize productivity. Here are some key strategies:

**Crop Diversification:** Planting a variety of crops helps spread risk. Different crops have different tolerances to temperature, precipitation, and pests, so diversification can help buffer against the impacts of climate variability.

**Breeding Resilient Varieties:** Developing and using crop varieties that are more resilient to climate stressors such as drought, heat, floods, and pests. This involves traditional breeding techniques as well as emerging technologies like genetic engineering.

**Water Management:** Implementing efficient irrigation systems, water conservation practices, and water harvesting techniques to ensure crops have adequate water during changing climate conditions, including both droughts and floods.

**Soil Health Management:** Practices such as conservation tillage, cover cropping, and adding organic matter to soil help improve soil structure, fertility, and water retention, making it more resilient to extreme weather events.

**Agroforestry and Agroecology:** Integrating trees and shrubs into agricultural landscapes (agroforestry) and adopting agroecological practices help increase biodiversity, improve soil health, enhance water retention, and provide natural pest control.

**Precision Agriculture:** Utilizing technology such as remote sensing, drones, and GPS-guided machinery to optimize resource use, minimize waste, and tailor management practices to specific environmental conditions.

**Integrated Pest Management (IPM):** Employing a combination of techniques such as crop rotation, biological control, and use of resistant varieties to manage pests and diseases in a sustainable manner.

**Climate-Smart Crop Management:** Implementing practices like adjusted planting dates, modified fertilizer and nutrient management, and adopting new cropping systems suited to changing climate conditions.

**Investment in Research and Innovation:** Supporting research into climate-resilient crop varieties, sustainable agricultural practices, and innovative technologies to address emerging challenges posed by climate change.

**Policy Support and Capacity Building:** Governments can provide incentives, subsidies, and support for farmers to adopt climate-smart practices. Extension services and farmer education programs can help build capacity and facilitate knowledge transfer.

**Risk Management and Insurance:** Developing and promoting risk management tools such as crop insurance, weather-indexed insurance, and contingency planning to help farmers mitigate losses during climate-related disasters.

**International Collaboration and Knowledge Sharing:** Encouraging collaboration among countries, institutions, and stakeholders to share best practices, technologies, and resources for adapting agriculture to climate change on a global scale.

By integrating these strategies into agricultural systems, policymaker, local community, and farmers can better adapt to the challenges posed by climate change while ensuring food security and sustainable production for future generations. 
'''

st.subheader("Recommendations for Adaptation and Mitigation")


def stream_data0():
    for word in _multi0.split(" "):
        yield word + " "
        time.sleep(0.02)


# if st.button("What are strategies to adapt and mitigate the Climate Change?"):
    # st.write_stream(stream_data0)
st.image('https://www.wur.nl/upload_mm/0/b/b/3bc88b1f-acd4-4627-927d-fbb41ec3e490_shutterstock_1572914245_3a50c691_750x400.jpg',
         caption='Adapting Agriculture in the Changing World')
on = st.toggle('**Strategies to Adapt and Mitigate**')

if on:
    st.write(stream_data0)
