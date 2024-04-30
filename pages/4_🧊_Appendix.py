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


tab1, tab2 = st.tabs(["🗃 Data", "📈 Author"])
tab1.subheader("Predictive Modelling")
tab1.write(
    "**In this study, Random Forest Regressor is employed to predict the data.**")
tab1.write(
    "500 trees are used as a parameter that specifies the number of decision trees. The seed for the random number generator used by the algorithm is 42")
tab1.write("The Mean Square Error of the algorithm is 0.0057")

tab1.subheader("Reference Data")

tab1.write("**Data that has been used to train the predictive model:**")
df9 = pd.read_csv("Projected_impacts_datasheet_11.24.2021.csv")
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
