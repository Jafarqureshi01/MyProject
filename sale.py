import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Streamlit UI
st.title("Big Mart Sales Prediction")

# Load dataset
st.sidebar.header('Upload Dataset')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    big_mart_data = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded Successfully!")
    st.dataframe(big_mart_data.head())  # Display the first few rows of the dataset

    # Checking for missing values
    st.subheader("Missing Values")
    missing_values = big_mart_data.isnull().sum()
    st.write(missing_values)

    # Fill missing values for 'Item_Weight' column
    if big_mart_data['Item_Weight'].isnull().sum() > 0:
        big_mart_data['Item_Weight'] = big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean())
        st.write("Missing values in 'Item_Weight' have been filled with the mean.")

    # Handle 'Outlet_Size' missing values based on mode for each 'Outlet_Type'
    mode_of_Outlet_Size = big_mart_data.pivot_table(
        values='Outlet_Size', 
        columns='Outlet_Type', 
        aggfunc=lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
    missing_outlet_size = big_mart_data['Outlet_Size'].isna()
    big_mart_data.loc[missing_outlet_size, 'Outlet_Size'] = big_mart_data.loc[missing_outlet_size, 'Outlet_Type'].apply(
        lambda x: mode_of_Outlet_Size[x] if x in mode_of_Outlet_Size else None
    )
    st.write("Missing values in 'Outlet_Size' have been filled based on the mode of corresponding 'Outlet_Type'.")

    # Clean column names
    big_mart_data.columns = big_mart_data.columns.str.strip()

    # Data Visualizations
    st.subheader("Item Weight Distribution")
    plt.figure(figsize=(6, 6))
    sns.histplot(big_mart_data['Item_Weight'], kde=True)
    st.pyplot()

    st.subheader("Item Visibility Distribution")
    plt.figure(figsize=(6, 6))
    sns.histplot(big_mart_data['Item_Visibility'], kde=True)
    st.pyplot()

    st.subheader("Item MRP Distribution")
    plt.figure(figsize=(6, 6))
    sns.histplot(big_mart_data['Item_MRP'], kde=True)
    st.pyplot()

    st.subheader("Outlet Establishment Year Distribution")
    plt.figure(figsize=(6, 6))
    sns.histplot(big_mart_data['Outlet_Establishment_Year'], kde=True)
    st.pyplot()

    # Countplots
    st.subheader("Outlet Establishment Year Count")
    plt.figure(figsize=(6, 6))
    sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
    st.pyplot()

    st.subheader("Item Fat Content Count")
    plt.figure(figsize=(6, 6))
    sns.countplot(x='Item_Fat_Content', data=big_mart_data)
    st.pyplot()

    st.subheader("Item Type Count")
    plt.figure(figsize=(30, 6))
    sns.countplot(x='Item_Type', data=big_mart_data)
    st.pyplot()

    st.subheader("Outlet Size Count")
    plt.figure(figsize=(6, 6))
    sns.countplot(x='Outlet_Size', data=big_mart_data)
    st.pyplot()

    # Data Preprocessing
    big_mart_data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)
    encoder = LabelEncoder()
    big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
    big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
    big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
    big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
    big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])
    big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
    big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

    # Define features and target
    x = big_mart_data.drop(columns='Item_MRP', axis=1)
    y = big_mart_data['Item_MRP']

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    # Train the model
    regressor = XGBRegressor()
    regressor.fit(x_train, y_train)

    # Predict and evaluate on training data
    training_data_prediction = regressor.predict(x_train)
    r2_train = metrics.r2_score(y_train, training_data_prediction)

    # Predict and evaluate on test data
    test_data_prediction = regressor.predict(x_test)
    r2_test = metrics.r2_score(y_test, test_data_prediction)

    # Display R-squared values
    st.subheader("Model Evaluation")
    st.write(f"R-squared value on Training Data: {r2_train}")
    st.write(f"R-squared value on Test Data: {r2_test}")

else:
    st.write("Please upload a CSV file to get started.")
