import streamlit as st
import base64

import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter
from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

# judul utama
st.title('APLIKASI SISTEM PREDIKSI CHURN PELANGGAN TELEKOMUNIKASI')
st.markdown('''---''')

# baca file gambar dan encode ke base64
file_path = "logoutm.png"
with open(file_path, "rb") as f:
    data = f.read()
    encoded = base64.b64encode(data).decode()

# tampilkan gambar dengan HTML + style center
st.sidebar.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{encoded}" width="100"><br>
        <p style="margin-top: 5px; font-weight: bold;"> Nabila Atira Qurratul Aini </p>
        <p style="margin-top: -10px;"> 210411100066 </p>
        <p style="margin-top: -10px;"> Teknik Informatika </p>
    </div>
    """,
    unsafe_allow_html=True
)

# membaca dataset telecommunication customer churn
dataset = pd.read_csv('https://raw.githubusercontent.com/NabilaAtiraQurratulAini/Dataset/refs/heads/main/Dataset%20TTC%20-%20Telecommunication%20Customer%20Churn.csv')

# fitur gender
modus_gender = dataset['gender'].mode()[0]
dataset['gender'].fillna(modus_gender, inplace=True)

# fitur tenure
mean_tenure = dataset['tenure'].mean()
dataset['tenure'].fillna(mean_tenure, inplace=True)

# fitur totalcharges
mean_totalcharges = dataset['TotalCharges'].mean()
dataset['TotalCharges'].fillna(mean_totalcharges, inplace=True)

# tentukan kolom numerik yang akan dianalisis
numeric_columns = dataset[['tenure', 'MonthlyCharges', 'TotalCharges']]

# hitung Q1, Q3, dan IQR
Q1 = numeric_columns.quantile(0.25)
Q3 = numeric_columns.quantile(0.75)
IQR = Q3 - Q1

# simpan Q1 dan Q3 dari data asli
q1_iqr = {}
q3_iqr = {}
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    q1_iqr[col] = dataset[col].quantile(0.25)
    q3_iqr[col] = dataset[col].quantile(0.75)

st.session_state.q1_iqr = q1_iqr
st.session_state.q3_iqr = q3_iqr

# ambil hanya kolom numerik yang diinginkan
numerical_features = dataset[['tenure', 'MonthlyCharges', 'TotalCharges']]

# terapkan LOF
lof = LocalOutlierFactor(n_neighbors=20)
outlier_scores = lof.fit_predict(numerical_features)

# buat dataframe hasil
dataset_filtered = numerical_features.copy()
dataset_filtered['LOF_Score'] = lof.negative_outlier_factor_
dataset_filtered['Outlier'] = ['Yes' if x == -1 else 'No' for x in outlier_scores]

# ambil data outlier saja
outliers_only = dataset_filtered[dataset_filtered['Outlier'] == 'Yes']

# hitung jumlah outlier per fitur
outlier_counts = {feature: outliers_only[feature].count() for feature in numerical_features}
outlier_table = pd.DataFrame(list(outlier_counts.items()), columns=['Fitur', 'Jumlah Outlier'])

# tangani langsung di dataset_filtered
fitur_asli = [col for col in numerical_features if col not in ['LOF_Score', 'Outlier']]

for feature in fitur_asli:
    mean_wajar = dataset_filtered[dataset_filtered['Outlier'] == 'No'][feature].mean()
    dataset_filtered[feature] = dataset_filtered.apply(lambda row: mean_wajar if row['Outlier'] == 'Yes' else row[feature], axis=1)

# salin dataset awal dan update nilai yang telah ditangani
dataset_preprocessing_lof = dataset.copy()
for feature in fitur_asli:
    dataset_preprocessing_lof[feature] = dataset_filtered[feature]
    
# menambahkan lof disini
st.session_state.lof_reference_data = dataset_preprocessing_lof[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()

st.title("🚀 DEPLOYMENT")

# pilih model
st.markdown('<p style="font-size:25px; font-weight:bold;">🔍 PILIH MODEL</p>', unsafe_allow_html=True)
model_option = st.selectbox("Pilih Model", ["IQR", "LOF"])
model_path = "model_mlp_iqr.h5" if model_option == "IQR" else "model_mlp_lof.h5"

st.markdown('<p style="font-size:20px; font-weight:bold;">📝 FORM INPUT DATA PELANGGAN</p>', unsafe_allow_html=True)

# input asli dari user
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.radio("SeniorCitizen", [0, 1])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (dalam bulan)", min_value=0.0)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["One year", "Two year", "Month-to-month"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", [
    "Credit card (automatic)", "Bank transfer (automatic)", "Mailed check", "Electronic check"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# masukkan data ke DataFrame
input_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([input_data])

if model_option == "IQR":
    input_df.drop(columns=["SeniorCitizen"], inplace=True)

# mapping encoding manual
encode_map = {
    'gender': {'Female': 0, 'Male': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'MultipleLines': {'No': 0, 'Yes': 1, 'No phone service': 2},
    'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
    'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'Contract': {'One year': 0, 'Two year': 1, 'Month-to-month': 2},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'PaymentMethod': {
        'Credit card (automatic)': 0,
        'Bank transfer (automatic)': 1,
        'Mailed check': 2,
        'Electronic check': 3
    }
}

for col, mapping in encode_map.items():
    input_df[col] = input_df[col].map(mapping)

# simpan nilai asli untuk outlier check
original_input = input_df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()

# status outlier
st.markdown('<p style="font-size:20px; font-weight:bold;">📊 STATUS OUTLIER</p>', unsafe_allow_html=True)

# iqr
iqr_status = []
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    Q1 = st.session_state.q1_iqr[col]
    Q3 = st.session_state.q3_iqr[col]
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    val = input_df[col].values[0]
    
    if val < lower or val > upper:
        iqr_status.append(col)

if not iqr_status:
    st.success("✔️ Data tidak termasuk outlier menurut IQR")
else:
    st.error(f"❌ Data termasuk outlier menurut IQR (fitur: {', '.join(iqr_status)})")

# lof
lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof_model.fit(st.session_state.lof_reference_data)
lof_result = lof_model.predict(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])[0]

if lof_result == 1:
    st.success("✔️ Data tidak termasuk outlier menurut LOF")
else:
    # identifikasi fitur mana yang memiliki nilai ekstrem terhadap distribusi referensi
    lof_outlier_features = []
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        min_val = st.session_state.lof_reference_data[col].min()
        max_val = st.session_state.lof_reference_data[col].max()
        val = input_df[col].values[0]
        if val < min_val or val > max_val:
            lof_outlier_features.append(col)

    if lof_outlier_features:
        st.error(f"❌ Data termasuk outlier menurut LOF (fitur: {', '.join(lof_outlier_features)})")
    else:
        st.error("❌ Data termasuk outlier menurut LOF")

# prediksi
if st.button("🔮 Prediksi"):
    try:
        model = load_model(model_path, compile=False)

        # pastikan fitur sesuai expected_features = 18
        expected_features = 18 if model_option == "IQR" else 19
        if input_df.shape[1] != expected_features:
            st.error(f"❌ Jumlah fitur tidak sesuai: model butuh {expected_features}, input punya {input_df.shape[1]}")
            st.stop()

        prob = model.predict(input_df)[0][0]
        label = "Churn" if prob > 0.5 else "Tidak Churn"

        st.success(f"🔣 Hasil Prediksi: **{label}**")
        st.info(f"🌟 Probabilitas Churn: **{prob:.2%}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
