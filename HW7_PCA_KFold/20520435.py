"""
Sinh viên: Nguyễn Duy Đạt
MSSV: 20520435
Lớp: CS116.N11.KHTN
Giảng Viên: Nguyễn Vinh Tiệp
Bài tập 6: Đánh giá mô hình Phân lớp với giao diện Streamlit + PCA

Tiến hành giảm chiều dữ liệu đặc trưng trước khi phân lớp, 
    sử dụng một mô hình bất kỳ: Ví dụ: SVM/Logistic Regression.

Xác định xem giảm bao nhiêu chiều thì cho độ chính xác cao nhất.
"""

# Import Package - library
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA

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
    st.title('iris_data')
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

        le = LabelEncoder()

        y_df = le.fit_transform(y_df)
    
    # 4.PCA
    option = st.radio("3. Select PCA", 
    ["Yes", "No"])
    if option == "Yes":
        n = st.text_input("Input max components:")
        n = int(n)
        if (n > 0) & (n <= X_df.shape[1]):
            X_df_list = []
            for i in range(1, n+1):
                pca = PCA(n_components=i)
                X_pca = pca.fit_transform(X_df)
                X_df_list.append(X_pca)
    # 5. Select Protocol
    #  5.1 Create radio protocol and set train/test split option & K-fold option
    protocol = st.radio("4. Select protocol", 
    ["Train/Test Split", "K-Fold"])
    if protocol == "Train/Test Split":
        ratio = st.text_input("Input Ratio:")
        
    elif protocol == "K-Fold":
        k = st.text_input("Input k:")
    
    model = SVC(kernel='rbf')

    #  5.2 Create Evaluate Function
    def evaluate_train_test(X_train, y_train, X_test, y_test):  

        res = model.fit(X_train, y_train)
        y_train_pred = res.predict(X_train)
        y_test_pred = res.predict(X_test)

        # Evaluate with F1
        train_error = round(f1_score(y_train, y_train_pred, average='micro'), 3)
        test_error = round(f1_score(y_test, y_test_pred, average='micro'), 3)
        st.write('F1 Score on train: ', train_error)
        st.write('F1 Score on test: ', test_error)

        chart_data = pd.DataFrame({
            'F1 Score': [train_error, test_error],
            'Evaluation dataset':["Train error", "Test error"]
        })

        bar_chart = alt.Chart(chart_data).mark_bar().encode(
            y='F1 Score',
            x=alt.X('Evaluation dataset', sort=None)
        )
        st.altair_chart(bar_chart, use_container_width=True)
        st.write('--------------------------------------------------------')

    # ----------------------------------------------------------
    # 5.3. Create evaluating k fold function
    def evaluate_k_fold(k, model, X_df_list, y_df):
        st.write('Number of Components: ',n)
        st.write('With k = ',k)
        scores_list = []
        components = []
        for i in range(0, len(X_df_list)):
            scores = round(cross_val_score(model, X_df_list[i], y_df, scoring='f1_micro', cv=k).mean(), 3)
            scores_list.append(scores)
            components.append(str(i+1))

        data = { 'Component': components, 'F1_Score': scores_list}
        
        df_summary = pd.DataFrame(data)

        st.write(df_summary)

        df_optimal = df_summary[df_summary['F1_Score'] == max(df_summary['F1_Score'])]
        st.write('Giá trị n_components và độ chính xác tối ưu:', df_optimal)

        chart_data = pd.DataFrame({
            'F1 Score': scores_list,
            'Component':components
        })

        bar_chart = alt.Chart(chart_data).mark_bar().encode(
            y='F1 Score',
            x=alt.X('Component', sort=None)
        )
        st.altair_chart(bar_chart, use_container_width=True)
        st.write('--------------------------------------------------------')
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
            evaluate_k_fold(k, model, X_df_list, y_df)