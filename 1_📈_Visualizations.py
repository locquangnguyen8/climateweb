import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
st.set_page_config(page_title="Data Visualization", layout="wide")
# Function to load data



# UI for file upload and data manipulation
st.header("Data Visualization")
st.write("Sample Figures from the Data")
st.image('https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41597-022-01150-7/MediaObjects/41597_2022_1150_Fig2_HTML.png?as=webp',
         caption='(a) By global temperature rise, (b) By projected time periods (midpoint years) and climate scenarios, (c) Regions and crop species, and (d) adaptation options and crop species. (d) Crops and adaptation options')
st.image('https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs10584-011-0156-z/MediaObjects/10584_2011_156_Fig3_HTML.gif?as=webp',
         caption='Emission Concentrations based on different Senarios')

# st.write("The Local Temprature Change Frequency Visualisation:")
# chart_data = df['Local Temperature Rise from 2005']
# st.bar_chart(chart_data)
# st.line_chart(chart_data)

# Assuming `file_details` shows some details about the data
# Uncomment and modify according to your app's context
# file_details = "Some details about the data file"
# st.write(file_details)

# st.write("First five data rows")
# st.write(df.head())

# st.write("Missing values table")
# Calculate missing values and convert to DataFrame
# missing_values_count = df.isna().sum().to_frame().T

# Optionally, you can rename the index to something more descriptive
# missing_values_count.index = ['Missing Values Count']

# Display the transposed DataFrame
# st.write(missing_values_count)

# st.write("Data Shape")
# st.write(df.shape)

# st.write("Basic Statistical summary")
# st.write(df.describe())

st.header("Interactive Visualization")
def load_data():
    # Modify the path according to your file location
    df1 = pd.read_excel('Projected_impacts_datasheet_11.24.2021.xlsx')
    df1.columns = df1.columns.str.replace(' ', '_')
    # Dictionary with new column names
    new_column_names = {
        'Ref_No_': 'Ref_No',
        'Current_Annual_Precipitation_(mm)__area_weighted_': 'Current_Annual_Precipitation_(mm)__area_weighted',
        'Global_delta_T_from_pre-industrial_period_': 'Global_delta_T_from_pre-industrial_period',
        '_Annual_Precipitation_change__from_2005_(mm)': 'Annual_Precipitation_change__from_2005_(mm)',
        'Others_': 'Others',
        '_Methods': 'Method', 'Local_delta_T_from_2005': 'Local Temperature Rise from 2005', 'Global_delta_T_from_2005': 'Global Temperature Rise from 2005',

    }

    # Rename columns
    df1.rename(columns=new_column_names, inplace=True)
    df1.drop(columns=['Reference', 'Publication_year', 'Ref_No',
                      'Method', 'Scenario_source', 'doi', 'ID'], inplace=True)
    df1 = df1[['Local Temperature Rise from 2005', 'Global Temperature Rise from 2005', 'latitude', 'longitude', 'Climate_scenario', 'Future_Mid-point',
               'Current_Average_Temperature_(dC)_area_weighted', 'Country', 'Time_slice', 'Fertiliser', 'Irrigation', 'Cultivar', 'Adaptation_type']]
    return df1


def load_data1():
    # Modify the path according to your file location
    df = df1.copy()
    new_column_names1 = {
        'Current_Average_Temperature_(dC)_area_weighted': 'Current Average Temperature',
        'Climate_scenario': 'Climate Scenario',
        'Adaptation_type': 'Adaptation Type', 'Future_Mid-point': 'Future Mid Point', 'Time_slice': 'Time Slice', }
    df.rename(columns=new_column_names1, inplace=True)
    return df

# Function to encode categorical columns


def encode_columns(df, columns):
    label_encoder = LabelEncoder()
    for column in columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df

# Streamlit app starts here


# Load and preprocess data
df1 = load_data()
df = load_data1()
categorical_columns = df.select_dtypes(
    include=['object', 'category', 'bool']).columns
categorical_cols = st.multiselect(
    'Select Categorical Variables to Encode', categorical_columns, default=[])

df = encode_columns(df, categorical_cols)
numeric_columns = df.select_dtypes(include=[np.number]).columns
selected_columns = st.multiselect("Select Columns to Plot", numeric_columns)

# Define your columns
col1, col2, col3, col4 = st.columns(4)

# Create a button in each column
hist_button = col1.button("Histogram")
scatter_button = col2.button("Scatter Plot")
box_button = col3.button("Box Plot")
heat_map_button = col4.button("Corr. Heatmap")

bins = st.slider("Number of Bins for Histogram", 5, 100, 20)

# bins = st.slider("Number of bins for histogram", 5, 100, 20)
if hist_button:
    # Min: 5, Max: 100, Default: 20

    if len(selected_columns) > 0:
        df_melt = pd.melt(df[selected_columns])
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.histplot(data=df_melt, x='value', hue='variable', bins=bins,
                     element="step", stat="density", common_norm=False, ax=ax)
        plt.xticks(rotation=90)  # rotate x-axis labels for better visibility
        st.pyplot(fig)


elif scatter_button:
    if len(selected_columns) == 2:
        fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1])
        st.plotly_chart(fig)
elif box_button:
    if len(selected_columns) > 0:
        fig = go.Figure()
        for column in selected_columns:
            fig.add_trace(go.Box(y=df[column], name=column))
        st.plotly_chart(fig)
elif heat_map_button:
    if len(selected_columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 10))
        correlation_matrix = df[selected_columns].corr().round(2)
        sns.heatmap(data=correlation_matrix,
                    annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
