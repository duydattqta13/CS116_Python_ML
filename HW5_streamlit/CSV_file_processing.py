# Bài tập: Kéo file csv, cho user tích chọn cột (checkbox),
# box chọn phương pháp (LinearRegression, RandomForest), tỉ lệ train/test (0.5,0.3,0.2)
# Chọn phương pháp normalize (option MinMax, Standard, Onehot)

# Chuẩn bị
# B1: Activate CS116
# B2: pip install streamlit
# B3: 

import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

# 1.Label
# st.title("chào thế giới!")
st.markdown("""
# Giới thiệu Streamlit
## I. Giới thiệu
## II. Các thành phần cơ bản
""")

title = st.text_input('Course', 'CS116')
st.write('Practice', title)
# 2.Text box
# 3.Drop down/ combo box
# 4.Check box
# 5.Radio button
# 6.Button
# 8. Group/Tab control
# 9.File uploader
uploaded_file = st.file_uploader("1. Kéo file csv vào đây:")
if uploaded_file is not None:

    # Read csv file
    df = pd.read_csv(uploaded_file)
    st.title('50_Startups')
    st.write(df)

    # Process csv file
    st.write('2. Select Columns')
    for i in range(len(df.columns)):
        columns_name = st.checkbox(df.columns[i])

    # Select Model
    model = st.selectbox("3. Select Models", 
    ["LinearRegression", "RandomForest"])
    is_pressed = st.button("Select model")
    if is_pressed:
        if model == "LinearRegression":
            st.write(model, " is selected!")
        elif model == "RandomForest":
            st.write(model," is selected!")

    # Select train/test ratio
    rate = st.selectbox("4. Select train/test ratio", 
    ["0.5", "0.3", "0.2"])
    is_pressed = st.button("Select ratio")
    if is_pressed:
        if rate == "0.5":
            st.write(rate, " is selected!")
        elif rate == "0.3":
            st.write(rate," is selected!")
        elif rate == "0.2":
            st.write(rate," is selected!")
    
    # Select normalization method
    normalize_method = st.selectbox("5. Select normalization method", 
    ["MinMaxScaler", "StandardScaler"])
    is_pressed = st.button("Select normalization method")
    if is_pressed:
        if normalize_method == "MinMaxScaler":
            st.write(normalize_method, " is selected!")
        elif normalize_method == "StandardScaler":
            st.write(normalize_method," is selected!")

    




