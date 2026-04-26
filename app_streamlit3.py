import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Prediksi Hidrasi",
    page_icon="💧",
    layout="wide"
)

# ===============================
# LOAD DATA & MODEL
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("Daily_Water_Intake.csv")

@st.cache_resource
def load_model():
    return joblib.load("model_tree.joblib")

df = load_data()
model = load_model()

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:

    try:
        st.image("logoml.png", width=200)
    except:
        st.warning("Logo tidak ditemukan")

    st.markdown("## 💧 Prediksi Hidrasi")

    st.info("""
Aplikasi ini digunakan untuk memprediksi tingkat hidrasi seseorang 
berdasarkan data individu dan kondisi lingkungan.
""")

    st.markdown("### 📌 Tema")
    st.write("Machine Learning - Kesehatan (Hidrasi)")

# ===============================
# TAB
# ===============================
tab1, tab2, tab3 = st.tabs([
    "🧠 Prediksi",
    "📊 Informasi",
    "👨‍💻 Developer"
])

# ===============================
# 🧠 TAB PREDIKSI
# ===============================
with tab1:

    st.title("🧠 Prediksi Tingkat Hidrasi")

    col1, col2 = st.columns(2)

    with col1:
        umur = st.slider("Umur", 10, 80, 25)
        berat = st.slider("Berat Badan (kg)", 30.0, 120.0, 60.0)
        gender = st.selectbox("Jenis Kelamin", ["Laki-Laki", "Perempuan"])

    with col2:
        air = st.slider("Konsumsi Air (liter)", 0.5, 5.0, 2.0)
        aktivitas = st.selectbox("Aktivitas", ["Rendah", "Sedang", "Tinggi"])
        cuaca = st.selectbox("Cuaca", ["Panas", "Dingin", "Normal"])

    if st.button("🔎 Prediksi", use_container_width=True):

        # ===============================
        # MAPPING (WAJIB)
        # ===============================
        gender_map = {
            "Laki-Laki": "Male",
            "Perempuan": "Female"
        }

        aktivitas_map = {
            "Rendah": "Low",
            "Sedang": "Moderate",
            "Tinggi": "High"
        }

        cuaca_map = {
            "Panas": "Hot",
            "Dingin": "Cold",
            "Normal": "Normal"
        }

        # ===============================
        # DATAFRAME
        # ===============================
        data = pd.DataFrame(
            [[
                umur,
                gender_map[gender],
                berat,
                air,
                aktivitas_map[aktivitas],
                cuaca_map[cuaca]
            ]],
            columns=[
                "Age",
                "Gender",
                "Weight (kg)",
                "Daily Water Intake (liters)",
                "Physical Activity Level",
                "Weather"
            ]
        )

        hasil = model.predict(data)[0]

        # ===============================
        # OUTPUT 
        # ===============================
        if hasil == "Good":
            st.success("✅ Hidrasi Baik (Good)")
            st.info("Kondisi tubuh Anda terhidrasi dengan baik. Pertahankan pola minum Anda 👍")

        elif hasil == "Poor":
            st.error("⚠️ Hidrasi Buruk (Poor)")
            st.info("Disarankan meningkatkan konsumsi air sekitar 2–3 liter per hari 💧")

        else:
            st.warning(f"Hasil tidak dikenali: {hasil}")

    st.divider()

    # ===============================
    # LINE CHART
    # ===============================
    st.subheader("📈 Line Chart Konsumsi Air Harian")

    fig, ax = plt.subplots()

    ax.plot(
        df["Daily Water Intake (liters)"],
        marker='o',
        linestyle='-',
        color='red'
    )

    ax.set_xlabel("Index")
    ax.set_ylabel("Daily Water Intake (liters)")
    ax.set_title("Trend Konsumsi Air Harian")

    ax.grid(True)

    st.pyplot(fig)

# ===============================
# 📊 TAB INFORMASI
# ===============================
with tab2:

    st.title("📊 Informasi")

    st.markdown("## 💧 Apa itu Hidrasi?")
    st.write("""
Hidrasi adalah kondisi keseimbangan cairan dalam tubuh yang penting untuk menjaga fungsi organ, suhu tubuh, dan metabolisme.
""")

    st.divider()

    st.markdown("## 📌 Faktor yang Mempengaruhi Hidrasi")

    st.markdown("### 👤 Umur")
    st.write("Semakin bertambah usia, kemampuan tubuh menjaga keseimbangan cairan dapat menurun.")

    st.markdown("### ⚖️ Berat Badan")
    st.write("Semakin tinggi berat badan, semakin banyak air yang dibutuhkan tubuh.")

    st.markdown("### 🥤 Konsumsi Air")
    st.write("""
- 1.5 – 2 liter: minimum  
- 2 – 3 liter: optimal  
- >3 liter: tinggi  
""")

    st.divider()

    st.markdown("## 🏃 Aktivitas Fisik")
    st.markdown("""
- **Rendah** → duduk, belajar  
- **Sedang** → jalan kaki, aktivitas rumah  
- **Tinggi** → olahraga, kerja berat  
""")

    st.divider()

    st.markdown("## 🚻 Jenis Kelamin")
    st.write("Laki-laki umumnya membutuhkan lebih banyak cairan dibanding perempuan.")

    st.divider()

    st.markdown("## 🌦️ Cuaca")
    st.markdown("""
- **Panas** (>30°C) → kebutuhan air meningkat  
- **Normal** (20–30°C) → kebutuhan standar  
- **Dingin** (<20°C) → rasa haus menurun  
""")

    st.divider()

    st.markdown("## 💡 Insight Model")
    st.info("""
Berat badan memiliki hubungan positif yang cukup kuat terhadap konsumsi air harian (0.65). 
Semakin besar berat badan, semakin tinggi kebutuhan air.
""")

# ===============================
# 👨‍💻 TAB DEVELOPER
# ===============================
with tab3:

    st.title("👨‍💻 Developer")

    st.markdown("""
**Nama:** Alrifat  
**Bidang:** Machine Learning  
**Project:** Prediksi Tingkat Hidrasi  
""")

    st.success("🚀 Terbuka untuk project Machine Learning")
