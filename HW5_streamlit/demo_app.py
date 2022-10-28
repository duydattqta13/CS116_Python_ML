# Bài tập: Kéo file csv, cho user tích chọn cột (checkbox),
# box chọn phương pháp (LinearRegression, RandomForest), tỉ lệ train/test (0.5,0.3,0.6)
# Chọn phương pháp normalize (option MinMax, Standard, Onehot)

# Chuẩn bị
# B1: Activate CS116
# B2: pip install streamlit
# B3: 

import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
# 1.Label
# st.title("chào thế giới!")
st.markdown("""
# Giới thiệu Streamlit
## 1. Giới thiệu
## 2. Các thành phần cơ bản
""")

title = st.text_input('Movie title', 'Life of Brian')
st.write('The current movie title is', title)
# 2.Text box
# 3.Drop down/ combo box
# 4.Check box
# 5.Radio button
# 6.Button
# 8. Group/Tab control
tab1, tab2 = st.tabs(["Tính toán", "Hình ảnh"])

# 7.Image
with tab2:
    st.image(Image.open("./data/abc.jpg"))
with tab1:
    a = st.text_input("Nhập a:")
    b = st.text_input("Nhập b:")

    # Có thể thay combo box (selectbox) bằng radiobox, checkbox ở dưới
    operator = st.selectbox("chọn phép toán", 
        ["Cộng", "Trừ", "Nhân", "Chia"])
    is_pressed = st.button("Tính")
    if is_pressed:
        if operator == "Cộng":
            st.write("Kết quả: ", int(a)+int(b))
        elif operator == "Trừ":
            st.write("Kết quả: ", int(a)-int(b))
        elif operator == "Nhân":
            st.write("Kết quả: ", int(a)*int(b))
        elif operator == "Chia":
            st.write("Kết quả: ", float(a)/float(b))
# 9.File uploader
uploaded_file = st.file_uploader("Chọn file thật lồng lộn:")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    img_path = 'data/' + uploaded_file.name
    with open(img_path, 'wb') as f:
        f.write(bytes_data)
    
    # Read image
    img = cv2.imread(img_path, 0)
    # Process image
    filter = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    result = cv2.filter2D(img, -1, filter)
    result = cv2.add(result, cv2.filter2D(img, -1, filter.T))

    filter = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    result = cv2.filter2D(img, -1, filter)
    result = cv2.add(result, cv2.filter2D(img, -1, filter.T))

    # Show image
    col_in, col_out = st.columns(2)
    with col_in:
        st.header("Ảnh gốc")
        st.image(Image.open(img_path))
    with col_out:
        st.header("Ảnh kết quả")
        st.image(result)

with tab2:
    files = os.scandir('./data/')
    col_left, col_right = st.columns(2)
    for i, file in enumerate(files):
        if i % 2 == 0:
            with col_left:
                st.image(Image.open(file))
        else:
            with col_right:
                st.image(Image.open(file))
            


