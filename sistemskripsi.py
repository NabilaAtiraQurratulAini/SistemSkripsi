import streamlit as st
import base64

import pandas as pd
import pickle

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
        <h4 style="margin-bottom: 30px;">IMPELENTASI MULTI-LAYER PERCEPTRON (MLP) UNTUK KLASIFIKASI CUSTOMER CHURN TELECOMMUNICATION BERDASARKAN FITUR PALING BERPENGARUH</h4>
        <img src="data:image/png;base64,{encoded}" width="100" style="margin-bottom: 15px;"><br>
        <p style="margin-top: 10px; font-weight: bold;"> Nabila Atira Qurratul Aini </p>
        <p style="margin-top: -10px;"> 210411100066 </p>
        <h4 style="margin-top: 30px; margin-bottom: -20px;">PROGRAM STUDI TEKNIK INFORMATIKA</h4>
        <h4 style="margin-bottom: -20px;">FAKULTAS TEKNIK</h4>
        <h4 style="margin-bottom: -20px;">UNIVERSITAS TRUNOJOYO MADURA</h4>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("🚀 DEPLOYMENT")

# === load model dengan pickle ===
model_path = "mlp3_iqr.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.markdown('<p style="font-size:20px; font-weight:bold;">📝 FORM INPUT DATA PELANGGAN</p>', unsafe_allow_html=True)

# input asli dari user
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.radio("Senior Citizen", [0, 1])
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
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
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
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
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

# drop fitur yang tidak dipakai model MLP3_IQR
drop_features = ["Dependents", "MultipleLines"]
for col in drop_features:
    if col in input_df.columns:
        input_df.drop(columns=[col], inplace=True)

# prediksi dengan scikit-learn MLPClassifier
if st.button("🔮 Prediksi"):
    try:
        prob = model.predict_proba(input_df)[0][1]
        label = model.predict(input_df)[0]
        label_text = "Churn" if label == 1 else "Tidak Churn"

        st.success(f"🔣 Hasil Prediksi: **{label_text}**")
        st.metric(label="🌟 Probabilitas Churn: ", value=f"**{prob:.2%}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")

