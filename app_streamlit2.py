import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Prediksi Tingkat Hidrasi",
    page_icon="💧",
    layout="centered"
)

# ===============================
# CUSTOM CSS (UI MODERN)
# ===============================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 16px;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA & MODEL
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("Daily_Water_Intake.csv")

df = load_data()
model = joblib.load("model_tree.joblib")

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:

    st.markdown("## 🚀 Menu Aplikasi")

    page = st.selectbox(
        "Pilih Halaman",
        [
            "🏠 Prediksi",
            "📊 Data",
            "📈 Insight Model"
        ]
    )

    st.divider()

    st.markdown("### 👨‍💻 Tentang Pembuat")
    st.markdown("""
    **Alrifat**  
    🎓 Machine Learning Enthusiast  
    """)

# ===============================
# 🏠 HALAMAN PREDIKSI
# ===============================
if page == "🏠 Prediksi":

    st.markdown("<h1 style='text-align:center;'>💧 Prediksi Tingkat Hidrasi</h1>", unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;'>Masukkan data untuk mengetahui kondisi hidrasi tubuh</p>", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        umur = st.slider("Umur", 10, 80, 25)
        berat = st.slider("Berat Badan (kg)", 30.0, 120.0, 60.0)
        gender = st.pills("Jenis Kelamin", ["Male", "Female"], default="Male")

    with col2:
        air = st.slider("Konsumsi Air Harian (liter)", 0.5, 5.0, 2.0)
        aktivitas = st.pills("Aktivitas Fisik", ["Low", "Moderate", "High"], default="Moderate")
        cuaca = st.pills("Cuaca", ["Hot", "Cold", "Normal"], default="Hot")

    st.divider()

    if st.button("🔎 Prediksi Tingkat Hidrasi", use_container_width=True):

        data = pd.DataFrame(
            [[umur, gender, berat, air, aktivitas, cuaca]],
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

        st.success(f"Hasil Prediksi: **{hasil}**")
        st.snow()

# ===============================
# 📊 HALAMAN DATA + HISTOGRAM
# ===============================
elif page == "📊 Data":

    st.title("📊 Data Overview")

    st.dataframe(df)

    st.subheader("📈 Histogram Konsumsi Air Harian")

    kolom = "Daily Water Intake (liters)"

    fig = plt.figure()
    plt.hist(df[kolom], bins=20)
    plt.xlabel("Daily Water Intake (liters)")
    plt.ylabel("Frekuensi")

    st.pyplot(fig)

    st.info("Histogram ini menunjukkan distribusi konsumsi air harian pada dataset.")

# ===============================
# 📈 HALAMAN INSIGHT MODEL
# ===============================
elif page == "📈 Insight Model":

    st.title("📈 Insight Model")

    # Tools
    st.markdown("## 🛠️ Tools yang Digunakan")
    st.write("""
    - Python  
    - Streamlit  
    - Scikit-learn  
    - Pandas  
    - Matplotlib  
    - Joblib  
    """)

    st.divider()

    # Insight Korelasi
    st.markdown("## 💡 Insight")

    st.info("""
Berdasarkan analisis korelasi, berat badan memiliki hubungan positif yang cukup kuat terhadap konsumsi air harian (0.65). 
Hal ini menunjukkan bahwa individu dengan berat badan lebih tinggi cenderung membutuhkan asupan air yang lebih banyak.

Oleh karena itu, dapat disimpulkan bahwa berat badan merupakan faktor yang paling berpengaruh dalam menentukan kebutuhan konsumsi air harian dibandingkan variabel lainnya.
""")