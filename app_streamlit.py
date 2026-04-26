import streamlit as st
import pandas as pd
import joblib

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Prediksi Tingkat Hidrasi",
    page_icon="💧",
    layout="centered"
)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("model_tree.joblib")

# ===============================
# HEADER
# ===============================
st.markdown(
"""
<h1 style='text-align: center;'>💧 Prediksi Tingkat Hidrasi</h1>
<p style='text-align: center; font-size:18px;'>
Aplikasi Machine Learning untuk memperkirakan kondisi hidrasi tubuh
berdasarkan karakteristik individu dan lingkungan.
</p>
""",
unsafe_allow_html=True
)

st.divider()

# ===============================
# INPUT DATA
# ===============================
with st.container():

    st.subheader("📊 Masukkan Data")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 10, 80, 25)
        weight = st.slider("Weight (kg)", 30.0, 120.0, 60.0)
        gender = st.pills("Gender", ["Male", "Female"], default="Male")

    with col2:
        water_intake = st.slider("Daily Water Intake (liters)", 0.5, 5.0, 2.0)
        activity = st.pills(
            "Physical Activity Level",
            ["Low", "Moderate", "High"],
            default="Moderate"
        )
        weather = st.pills(
            "Weather",
            ["Hot", "Cold", "Normal"],
            default="Hot"
        )

st.divider()

# ===============================
# PREDIKSI
# ===============================
if st.button("🔎 Prediksi Tingkat Hidrasi", use_container_width=True):

    data_baru = pd.DataFrame(
        [[age, gender, weight, water_intake, activity, weather]],
        columns=[
            "Age",
            "Gender",
            "Weight (kg)",
            "Daily Water Intake (liters)",
            "Physical Activity Level",
            "Weather"
        ]
    )

    prediksi = model.predict(data_baru)[0]

    st.markdown("### 🧠 Hasil Prediksi")

    st.success(f"Tingkat hidrasi yang diprediksi oleh model adalah **{prediksi}**")

    st.balloons()

st.divider()

# ===============================
# FOOTER
# ===============================
st.markdown(
"""
<div style='text-align:center; color:gray;'>
Dibuat oleh <b>Alrifat</b> | Machine Learning Klasifikasi
</div>
""",
unsafe_allow_html=True
)