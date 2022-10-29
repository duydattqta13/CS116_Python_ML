"""
Sinh viên: Nguyễn Duy Đạt
MSSV: 20520435
Lớp: CS116.N11.KHTN
Giảng Viên: Nguyễn Vinh Tiệp
Bài tập 5: Đánh giá mô hình Hồi quy với giao diện Streamlit

Sử dụng Streamlit để làm giao diện ứng dụng theo gợi ý trên lớp lý thuyết.

Yêu cầu bao gồm:
Thiết kế giao diện với Streamlit để có thể:
- Upload file csv (sau này có thể thay bằng tập dữ liệu khác dễ dàng).
- Hiển thị bảng dữ liệu với file đã upload
- Chọn lựa input feature (các cột dữ liệu đầu vào)
- Chọn lựa hệ số cho train/test split: Ví dụ 0.8 có nghĩa là 80% để train và 20% để test
- Chọn lựa hệ số K cho K-Fold cross validation: Ví dụ K =4
- Nút "Run" để tiến hành chạy và đánh giá thuật toán

Output sẽ là biểu đồ cột hiển thị các kết quả sử dụng độ đo MAE và MSE. 
*Lưu ý: Train/Test split và K-Fold cross validation được thực hiện độc lập, 
chỉ chọn 1 trong hai phương pháp này.
"""

# Import Package - library
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Student Information
st.markdown("""
# HOMEWORK CS116
## Name: Duy Dat Nguyen
## Student ID: 20520435
""")

title = st.text_input('Course', 'CS116')
st.write('Practice', title)

# * Upload CSV file
uploaded_file = st.file_uploader("1. Kéo file csv vào đây:")
if uploaded_file is not None:

    # 1. Read csv file
    df2 = pd.read_csv(uploaded_file)
    st.title('50_Startups')
    st.write(df2)

    df = pd.DataFrame()

    # 2. Process csv file & select columns
    st.write('2. Select Columns')
    st.write("Input: ")
    name = []
    for i in range(len(df2.columns) - 1):
        columns_name = st.checkbox(df2.columns[i])
        if columns_name:
            name.append(str(df2.columns[i]))
    if len(name) == 0:
        st.write('Please choose at least one column!')
    else:
        df[name] = df2[name]
        df[df2.columns[-1]] = df2[df2.columns[-1]]

        st.write("DataFrame after selecting columns:",df)

        st.write("Output: ", df2.columns[-1])
        #--------------------------------------------------------
        X_df = df.drop(columns=df2.columns[-1])
        y_df = df[df2.columns[-1]]

        # 3. Encode categorical columns
        X_df = pd.get_dummies(X_df)

    # 4. Scale Data
    scaler = MinMaxScaler()
    
    for i in range(len(df.columns)):
        if df.columns[i] == 'R&D Spend':
            X_df[['R&D Spend']] = scaler.fit_transform(X_df[['R&D Spend']])
        elif df.columns[i] == 'Administration':
            X_df[['Administration']] = scaler.fit_transform(X_df[['Administration']])
        elif df.columns[i] == 'Marketing Spend':
            X_df[['Marketing Spend']] = scaler.fit_transform(X_df[['Marketing Spend']])

    # 5. Select Protocol
    #  5.1 Create radio protocol and set train/test split option & K-fold option
    protocol = st.radio("3. Select protocol", 
    ["Train/Test Split", "K-Fold"])
    if protocol == "Train/Test Split":
        ratio = st.text_input("Input Ratio:")
        
    elif protocol == "K-Fold":
        k = st.text_input("Input k:")
    
    model = LinearRegression()

    #  5.2 Create Evaluate Function
    def evaluate_train_test(X_train, y_train, X_test, y_test):  

        res = model.fit(X_train, y_train)
        y_train_pred = res.predict(X_train)
        y_test_pred = res.predict(X_test)

        # a) Evaluate with MSE
        train_error = round(mean_squared_error(y_train, y_train_pred), 3)
        test_error = round(mean_squared_error(y_test, y_test_pred), 3)
        st.write('MSE on train: ', train_error)
        st.write('MSE on test: ', test_error)

        chart_data = pd.DataFrame({
            'MSE': [train_error, test_error],
            'Evaluation dataset':["Train error", "Test error"]
        })

        bar_chart = alt.Chart(chart_data).mark_bar().encode(
            y='MSE',
            x=alt.X('Evaluation dataset', sort=None)
        )
        st.altair_chart(bar_chart, use_container_width=True)
        st.write('--------------------------------------------------------')

        # b) Evaluate with MAE
        train_error = round(mean_absolute_error(y_train, y_train_pred), 3)
        test_error = round(mean_absolute_error(y_test, y_test_pred), 3)
        st.write('MAE on train: ', train_error)
        st.write('MAE on test: ', test_error)
        
        chart_data = pd.DataFrame({
            'MAE': [train_error, test_error],
            'Evaluation dataset':["Train error", "Test error"]
        })

        bar_chart = alt.Chart(chart_data).mark_bar().encode(
            y='MAE',
            x=alt.X('Evaluation dataset', sort=None)
        )
        st.altair_chart(bar_chart, use_container_width=True)

    # ----------------------------------------------------------
    # 5.3. Create evaluating k fold function
    def evaluate_k_fold(k, model, X_df, y_df, scoring):
        if scoring == 'neg_mean_squared_error':
            scores = abs(cross_val_score(model, X_df, y_df, cv=k, scoring=scoring, n_jobs=-1))
            st.write("Mean MSE Score:", scores.mean())
            MSE = []
            Fold = []
            for i in range(k):
                MSE.append(scores[i])
                Fold.append("Fold " + str(i+1))
            chart_data = pd.DataFrame({
                'MSE': MSE,
                'Fold': Fold
            })
        
            bar_chart = alt.Chart(chart_data).mark_bar().encode(
                y='MSE',
                x= alt.X('Fold', sort=None)
            )
            st.altair_chart(bar_chart, use_container_width=True)

        elif scoring =='neg_mean_absolute_error':
            scores = abs(cross_val_score(model, X_df, y_df, scoring=scoring, n_jobs=-1))
            st.write("Mean MAE Score:", scores.mean())
            MAE = []
            Fold = []
            for i in range(k):
                MAE.append(scores[i])
                Fold.append("Fold " + str(i+1))
            
            chart_data = pd.DataFrame({
                'MAE': MAE,
                'Fold': Fold
            })
        
            bar_chart = alt.Chart(chart_data).mark_bar().encode(
                y='MAE',
                x= alt.X('Fold', sort=None)
            )
            st.altair_chart(bar_chart, use_container_width=True)

    # ----------------------------------------------------------
    # 6. Create "Run" button
    run_button = st.button("Run")
    if run_button:
        if protocol == "Train/Test Split":
            ratio = float(ratio)
            X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, y_df, 
                                                                        test_size= 1 - ratio, random_state=10, shuffle=True)
            evaluate_train_test(X_train_df, y_train_df, X_test_df, y_test_df)
            
        elif protocol == "K-Fold":
            k = int(k)
            # a) Evaluate with MSE
            evaluate_k_fold(k, model, X_df, y_df, scoring='neg_mean_squared_error')
            # b) Evaluate with MAE
            evaluate_k_fold(k, model, X_df, y_df, scoring='neg_mean_absolute_error')