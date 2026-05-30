# =================================================================
# SECTION 1: IMPORT LIBRARIES
# =================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as nn  # Menggunakan st untuk Streamlit (standar)
import streamlit as st

# Mengatur konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Sleep Health & Stress Prediction",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan premium, modern, dan bersih
st.markdown("""
<style>
    /* Styling font dan background */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4A90E2, #50E3C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #7F8C8D;
        margin-bottom: 2rem;
    }
    
    /* Card design */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(224, 224, 224, 0.6);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2C3E50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7F8C8D;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Result card */
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .result-low {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4);
    }
    .result-medium {
        background: linear-gradient(135deg, #f2994a, #f2c94c);
        box-shadow: 0 4px 15px rgba(242, 153, 74, 0.4);
    }
    .result-high {
        background: linear-gradient(135deg, #eb5757, #000000);
        box-shadow: 0 4px 15px rgba(235, 87, 87, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# =================================================================
# SECTION 2: UTILITY FUNCTIONS (LOAD DATA & MODELS)
# =================================================================

@st.cache_resource
def load_model_and_preprocessor():
    """Memuat model AdaBoost dan objek preprocessing yang sudah dilatih."""
    model_path = 'models/adaboost_model.pkl'
    prep_path = 'models/preprocessor.pkl'
    
    if os.path.exists(model_path) and os.path.exists(prep_path):
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        return model, preprocessor
    return None, None

@st.cache_data
def load_datasets():
    """Memuat dataset asli dan dataset terproses."""
    raw_path = 'data/sleep_health.csv'
    processed_path = 'data/sleep_health_preprocessed.csv'
    
    raw_df = pd.read_csv(raw_path) if os.path.exists(raw_path) else None
    processed_df = pd.read_csv(processed_path) if os.path.exists(processed_path) else None
    return raw_df, processed_df

# Load Model dan Data
model, preprocessor = load_model_and_preprocessor()
df_raw, df_processed = load_datasets()


# =================================================================
# SIDEBAR NAVIGATION
# =================================================================

# Cek apakah model utama sudah ditraining
if model is None or preprocessor is None or df_raw is None:
    st.warning("⚠️ Model atau Dataset utama belum siap! Silakan jalankan `train_model.py` terlebih dahulu untuk melatih model.")

    if st.button("Jalankan Pelatihan Model Sekarang"):
        with st.spinner("Melatih model di background... Mohon tunggu..."):
            import subprocess
            subprocess.run(["python", "train_model.py"])
            st.success("🎉 Pelatihan model selesai! Silakan refresh halaman ini.")
            st.rerun()
    st.stop()

# Sidebar Styling & Logo
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h2 style="color: #2C3E50; font-weight: 800; margin-bottom: 0px;">🏥 HealthPredict</h2>
    <p style="color: #7F8C8D; font-size: 0.85rem;">Multi-Dataset ML System</p>
</div>
""", unsafe_allow_html=True)

# Pilihan Tipe Dashboard Utama
dashboard_type = st.sidebar.selectbox(
    "Pilih Tipe Dashboard",
    ["🌙 Dashboard Pribadi (Sleep Health)", "👥 Dashboard Kelompok (Multi-Dataset)"]
)

st.sidebar.markdown("---")

if dashboard_type == "🌙 Dashboard Pribadi (Sleep Health)":
    # Pilihan Menu Pribadi
    menu = st.sidebar.radio(
        "Navigasi Menu",
        ["📊 Dashboard Dataset (EDA)", "🔮 Prediksi Tingkat Stress", "📈 Evaluasi & Kinerja Model", "📁 Prediksi Massal (CSV)", "🧪 Ruang Eksperimen ML", "📚 Teori & Cara Kerja AdaBoost"]
    )
else:
    # Pilihan Menu Kelompok
    menu_kelompok = st.sidebar.radio(
        "Navigasi Menu Kelompok",
        ["📁 Kelola & Latih Dataset", "📊 Analisis Data (EDA)", "🔮 Prediksi Real-time", "📈 Evaluasi Model", "🧪 Ruang Eksperimen ML", "🏆 Perbandingan & Kesimpulan", "📚 Teori & Cara Kerja AdaBoost"]
    )

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Pengembang:**
- Parisan Apro (152023141)
- Machine Learning

**Teknologi:**
- Python & Streamlit
- Scikit-Learn
- AdaBoost Classifier
""")


# =================================================================
# SECTION 3: VISUALIZATION & PAGES
# =================================================================

# Custom Palette untuk Visualisasi
colors = ["#77A6F7", "#FFCC5C", "#FF6F69"]


def render_adaboost_theory():
    st.markdown('<div class="main-title">📚 Teori & Cara Kerja AdaBoost Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Penjelasan Konseptual, Alur Matematika, Keunggulan, dan Limitasi Algoritma AdaBoost</div>', unsafe_allow_html=True)
    
    # 1. Definisi & Konsep Utama
    st.subheader("💡 Apa itu AdaBoost?")
    st.markdown("""
    **AdaBoost** (kependekan dari **Adaptive Boosting**) adalah salah satu algoritma *Ensemble Learning* yang paling populer dan revolusioner. 
    Algoritma ini pertama kali dirumuskan oleh **Yoav Freund** dan **Robert Schapire** pada tahun 1996, yang memenangkan penghargaan *Gödel Prize* atas kontribusinya ini.
    
    Metode ini bekerja dengan prinsip **Boosting**, yaitu melatih serangkaian model sederhana secara berurutan (sekuensial). 
    Setiap model baru dalam rantai ini dirancang khusus untuk memperbaiki kesalahan yang dibuat oleh model-model sebelumnya. 
    Keputusan akhir didapatkan melalui skema pemungutan suara berbobot (*weighted voting*).
    """)
    
    # 3 Konsep Utama Card Layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card" style="text-align: left; height: 260px;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #4A90E2; margin-bottom: 10px;">🧩 Ensemble Learning</div>
            <p style="font-size: 0.9rem; color: #555; line-height: 1.5;">
                Pendekatan machine learning yang menggabungkan prediksi dari beberapa model dasar (weak learners) untuk menghasilkan model akhir yang lebih stabil, akurat, dan kuat (strong learner).
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: left; height: 260px;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #50E3C2; margin-bottom: 10px;">🌲 Decision Stump</div>
            <p style="font-size: 0.9rem; color: #555; line-height: 1.5;">
                Pohon keputusan satu tingkat (kedalaman = 1). Stump hanya membelah data berdasarkan satu fitur tunggal dengan satu aturan pemisah. Ini adalah "weak learner" bawaan dari AdaBoost.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card" style="text-align: left; height: 260px;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #FFCC5C; margin-bottom: 10px;">⚖️ Adaptive Weighting</div>
            <p style="font-size: 0.9rem; color: #555; line-height: 1.5;">
                Proses dinamis di mana bobot sampel data yang salah diklasifikasikan akan diperbesar, memaksa model berikutnya di iterasi selanjutnya untuk lebih fokus mempelajari sampel-sampel sulit tersebut.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    
    # 2. Cara Kerja Secara Detail (Step-by-Step)
    st.subheader("⚙️ Alur Kerja Algoritma AdaBoost")
    st.markdown("Berikut adalah langkah-langkah sistematis bagaimana AdaBoost melatih model dan memperbarui bobot data secara matematis:")
    
    # Accordion / Tabs untuk Alur Kerja agar interaktif
    tab_step1, tab_step2, tab_step3, tab_step4 = st.tabs([
        "1️⃣ Inisialisasi Bobot",
        "2️⃣ Pelatihan & Penghitungan Error",
        "3️⃣ Perhitungan Bobot Model (Amount of Say)",
        "4️⃣ Pembaruan Bobot Sampel & Normalisasi"
    ])
    
    with tab_step1:
        st.markdown(r"""
        Pada awal proses ($t = 1$), setiap sampel data ke-$i$ diberikan bobot awal ($w_i$) yang sama besar. Jika terdapat $N$ total sampel data, maka:
        """)
        st.latex(r"w_i = \frac{1}{N}, \quad \text{untuk setiap } i = 1, 2, \dots, N")
        st.markdown("""
        Artinya, pada langkah pertama, semua baris data dianggap memiliki tingkat kepentingan yang setara untuk dipelajari.
        """)
        
    with tab_step2:
        st.markdown(r"""
        Untuk setiap iterasi $t$ (dari total estimator $T$):
        
        1. Latih sebuah *weak learner* $h_t(x)$ (misalnya Decision Stump) menggunakan dataset dengan bobot sampel saat ini $w_i$.
        2. Hitung **tingkat kesalahan (error rate)** $\epsilon_t$ dari model tersebut pada data latih. Error rate dihitung sebagai jumlah bobot sampel yang salah diklasifikasikan:
        """)
        st.latex(r"\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i")
        st.markdown(r"""
        *Catatan: Jika model sangat buruk (error $\epsilon_t \ge 0.5$ pada klasifikasi biner), pelatihan biasanya dihentikan atau bobot diatur ulang karena performanya tidak lebih baik dari tebakan acak.*
        """)
        
    with tab_step3:
        st.markdown(r"""
        Setelah menghitung tingkat kesalahan $\epsilon_t$, kita menghitung seberapa besar pengaruh atau kontribusi model $h_t$ tersebut dalam pengambilan keputusan akhir. 
        Parameter ini disebut **Amount of Say** ($\alpha_t$) atau bobot model:
        """)
        st.latex(r"\alpha_t = \frac{1}{2} \ln \left( \frac{1 - \epsilon_t}{\epsilon_t} \right)")
        st.markdown(r"""
        **Sifat Matematis $\alpha_t$:**
        - Jika model memiliki error yang sangat kecil ($\epsilon_t \to 0$), maka $\alpha_t$ bernilai positif yang sangat besar (model dipercaya penuh).
        - Jika model memiliki error mendekati $0.5$ ($50\%$), maka $\alpha_t \to 0$ (model hampir tidak memiliki hak suara).
        - Jika model selalu salah ($\epsilon_t \to 1$), maka $\alpha_t$ bernilai negatif besar (kita bisa membalikkan keputusan model untuk mendapatkan jawaban benar).
        """)
        
    with tab_step4:
        st.markdown("""
        Untuk melatih model berikutnya di iterasi $t+1$, bobot sampel diperbarui agar model selanjutnya fokus pada sampel yang salah diklasifikasikan:
        
        - Untuk sampel yang **salah** diklasifikasikan ($h_t(x_i) \neq y_i$), tingkat kepentingannya **ditingkatkan**:
        """)
        st.latex(r"w_i^{(t+1)} = w_i^{(t)} \times e^{\alpha_t}")
        st.markdown("""
        - Untuk sampel yang **benar** diklasifikasikan ($h_t(x_i) = y_i$), tingkat kepentingannya **diturunkan**:
        """)
        st.latex(r"w_i^{(t+1)} = w_i^{(t)} \times e^{-\alpha_t}")
        
        st.markdown("""
        Terakhir, agar jumlah seluruh bobot sampel tetap sama dengan 1, kita melakukan **normalisasi** dengan membagi setiap bobot dengan total jumlah bobot:
        """)
        st.latex(r"w_i^{(t+1)} = \frac{w_i^{(t+1)}}{\sum_{j=1}^N w_j^{(t+1)}}")

    # Prediksi Konsensus Akhir
    st.write("")
    st.info("""
    **🗳️ Prediksi Akhir (Final Consensus):**
    Setelah seluruh $T$ model dasar selesai dilatih, prediksi akhir untuk sampel data baru ditentukan dengan menjumlahkan suara berbobot ($\alpha_t$) dari seluruh weak learners:
    """)
    st.latex(r"H(x) = \text{sign} \left( \sum_{t=1}^T \alpha_t h_t(x) \right)")
    st.markdown(r"Formula di atas adalah untuk klasifikasi biner $\{-1, 1\}$. Untuk klasifikasi multi-kelas (seperti Stress Level yang memiliki kelas 0, 1, dan 2), Scikit-Learn menggunakan variasi algoritma **SAMME** atau **SAMME.R** yang memperluas konsep ini ke ruang probabilitas multi-kelas.")

    st.write("")
    st.write("")
    
    # 3. Kelebihan dan Kekurangan AdaBoost
    st.subheader("⚖️ Kelebihan & Limitasi AdaBoost Classifier")
    col_pro, col_contra = st.columns(2)
    
    with col_pro:
        st.markdown("""
        <div style="background-color: rgba(80, 227, 194, 0.15); padding: 20px; border-radius: 10px; border-left: 5px solid #50E3C2; height: 100%;">
            <h4 style="color: #2E7D32; margin-top: 0px;">🌟 Kelebihan & Manfaat</h4>
            <ul style="font-size: 0.95rem; line-height: 1.6; margin-bottom: 0px;">
                <li><b>Peningkatan Akurasi Signifikan:</b> Mengubah gabungan tebakan lemah menjadi klasifikasi yang sangat presisi.</li>
                <li><b>Minim Overfitting:</b> Karena dilatih bertahap menggunakan model dasar sederhana (stump), AdaBoost cenderung lebih tahan terhadap overfitting dibandingkan Decision Tree tunggal yang dalam.</li>
                <li><b>Mudah Digunakan:</b> Hanya memerlukan sedikit parameter utama untuk diatur (terutama jumlah estimator dan learning rate).</li>
                <li><b>Feature Selection Alami:</b> Saat melatih Decision Stump, AdaBoost secara tidak langsung mengurutkan fitur mana yang paling informatif (dapat dilihat pada visualisasi <i>Feature Importance</i>).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_contra:
        st.markdown("""
        <div style="background-color: rgba(235, 87, 87, 0.15); padding: 20px; border-radius: 10px; border-left: 5px solid #eb5757; height: 100%;">
            <h4 style="color: #C62828; margin-top: 0px;">⚠️ Limitasi & Tantangan</h4>
            <ul style="font-size: 0.95rem; line-height: 1.6; margin-bottom: 0px;">
                <li><b>Sensitif terhadap Noise & Outliers:</b> Karena algoritma berulang kali memperbesar bobot data yang salah, data yang menyimpang jauh (outlier) atau salah input akan dipaksa untuk dipelajari, yang dapat merusak kualitas model secara keseluruhan.</li>
                <li><b>Sensitif terhadap Bias Data:</b> Jika dataset sangat tidak seimbang (class imbalance) tanpa penanganan pra-proses, performa kelas minoritas akan terabaikan.</li>
                <li><b>Pelatihan Sekuensial:</b> Proses pelatihan tidak bisa diparalelisasi (seperti Random Forest) karena setiap pohon $t$ harus menunggu hasil evaluasi dari pohon $t-1$ untuk menghitung pembaruan bobot.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    
    # 4. Hubungan dengan Parameter di Dashboard
    st.subheader("💡 Menghubungkan Teori dengan Praktik (GridSearchCV)")
    st.markdown(r"""
    Dalam menu evaluasi model di dashboard ini, Anda akan melihat parameter seperti **n_estimators** dan **learning_rate**. Berikut adalah makna akademis di balik parameter tersebut:
    
    *   **Jumlah Estimator (`n_estimators` / $T$):** 
        Menentukan berapa banyak Decision Stump yang akan dilatih secara sekuensial. 
        - Jika terlalu kecil, model akan *underfitting* karena tidak cukup waktu memperbaiki kesalahan.
        - Jika terlalu besar, model dapat mengalami *overfitting* dan waktu komputasi training akan meningkat secara linear.
    *   **Learning Rate ($\eta$ / Penyusutan):** 
        Berfungsi untuk memperkecil kontribusi setiap pohon baru sebesar faktor $\eta$ ($0 < \eta \le 1$). 
        - Menurunkan learning rate membantu memperlambat proses pembelajaran, yang membuat proses pencarian batas keputusan menjadi lebih halus dan mencegah overfitting. 
    """)


def render_ml_experiment_room():
    st.markdown('<div class="main-title">🧪 Ruang Eksperimen & Preprocessing ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Eksplorasi Efek Konfigurasi Preprocessing dan Hyperparameter Model AdaBoost</div>', unsafe_allow_html=True)
    
    import glob
    csv_files = glob.glob("data/*.csv")
    dataset_options = []
    for f in csv_files:
        basename = os.path.basename(f)
        if not (basename.endswith("_preprocessed.csv") or basename.startswith("template_") or basename == "hasil_prediksi_stress.csv"):
            dataset_options.append(basename)
            
    if not dataset_options:
        st.info("💡 Belum ada dataset yang terdaftar. Silakan unggah file CSV di menu 'Kelola & Latih Dataset'.")
        return
        
    # Memilih Dataset
    selected_csv = st.selectbox("Pilih Dataset untuk Eksperimen:", options=dataset_options)
    csv_filepath = os.path.join("data", selected_csv)
    df_temp = pd.read_csv(csv_filepath)
    
    # Deteksi Target Column
    cols = df_temp.columns.tolist()
    default_idx = len(cols) - 1
    for i, c in enumerate(cols):
        if c.lower() in ['target', 'outcome', 'class', 'label', 'source', 'stress level', 'classification']:
            default_idx = i
            break
    target_col = st.selectbox("Pilih Kolom Target untuk Eksperimen:", options=cols, index=default_idx)
    
    st.markdown("---")
    
    # 2 Kolom Kiri: Konfigurasi, Kanan: Teori Preprocessing & Parameter
    col_config, col_theory = st.columns([1, 1])
    
    with col_config:
        st.subheader("⚙️ Konfigurasi Eksperimen")
        with st.form("experiment_form"):
            # A. Preprocessing Settings
            st.markdown("**1. Pilihan Preprocessing Data:**")
            scaler_choice = st.selectbox(
                "Metode Normalisasi Fitur (Scaling):",
                options=["minmax", "standard", "none"],
                format_func=lambda x: "MinMaxScaler (Rentang 0-1)" if x == "minmax" else ("StandardScaler (z-score)" if x == "standard" else "Tanpa Normalisasi (No Scaling)")
            )
            
            clip_outliers = st.selectbox(
                "Pembersihan Outliers (IQR Clipping):",
                options=[True, False],
                format_func=lambda x: "AKTIF (Batasi nilai ekstrem Q1 - 1.5*IQR s/d Q3 + 1.5*IQR)" if x else "NON-AKTIF (Biarkan nilai asli)"
            )
            
            st.write("")
            
            # Feature Selection
            st.markdown("**2. Seleksi Fitur (Feature Selection):**")
            all_features_list = [c for c in cols if c != target_col]
            
            # Tentukan default fitur yang dibuang jika menggunakan sleep_health.csv
            default_drops = []
            if selected_csv == "sleep_health.csv":
                for f in ['Occupation', 'Diastolic', 'Daily Steps']:
                    if f in all_features_list:
                        default_drops.append(f)
            
            selected_features_to_drop = st.multiselect(
                "Pilih Fitur yang Ingin DIBUANG dari Pelatihan:",
                options=all_features_list,
                default=default_drops,
                help="Memilih fitur yang dibuang membantu mendeteksi pengaruh multikolinearitas (seperti Diastolic) dan korelasi rendah (seperti Occupation) terhadap kestabilan dan akurasi model."
            )
            
            st.write("")
            
            # B. AdaBoost Parameter Settings
            st.markdown("**3. Hyperparameter Model AdaBoost:**")
            base_depth = st.slider(
                "Kedalaman Model Dasar (max_depth Base Learner):",
                min_value=1, max_value=3, value=1,
                help="Kedalaman 1 disebut Decision Stump. Kedalaman lebih tinggi meningkatkan kompleksitas model."
            )
            
            n_estimators = st.slider(
                "Jumlah Estimator (n_estimators):",
                min_value=10, max_value=200, value=50, step=10,
                help="Berapa banyak pohon keputusan mini yang akan dilatih."
            )
            
            learning_rate = st.slider(
                "Laju Pembelajaran (learning_rate):",
                min_value=0.01, max_value=2.0, value=1.0, step=0.05,
                help="Mengontrol besarnya penyusutan kontribusi masing-masing pohon baru."
            )
            
            submitted = st.form_submit_button("Jalankan Eksperimen ML 🧪")
            
    with col_theory:
        st.subheader("📖 Penjelasan Alur Kerja & Parameter")
        
        # Penjelasan Preprocessing
        with st.expander("🛠️ Bagaimana missing value dan outliers ditangani?", expanded=True):
            st.markdown(r"""
            Sistem preprocessing melakukan penanganan otomatis secara rapi:
            *   **Missing Value (Nilai Kosong)**:
                *   Fitur kategorik diisi dengan **Modus** (kategori paling sering muncul).
                *   Fitur numerik diisi dengan **Median** (tahan terhadap pencilan ekstrem).
                *   Baris data dengan target bernilai `NaN` dihapus dari data latih/uji.
                *   Kolom yang 100% kosong dibuang secara otomatis.
            *   **Outlier (Pencilan)**:
                Jika diaktifkan, data numerik di luar batas $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$ akan dipotong (*clipped*) ke batas terdekat untuk menghindari ketidakstabilan algoritma AdaBoost.
            """)
            
        # Penjelasan Parameter
        with st.expander("⚙️ Apa makna parameter AdaBoost di samping?", expanded=True):
            st.markdown(r"""
            *   **Base Learner Depth (`max_depth`)**: 
                Kedalaman 1 (*Decision Stump*) membelah data berdasarkan 1 fitur. Pohon kedalaman 2 atau 3 mempelajari pola non-linear lebih kompleks, tapi mudah *overfitting*.
            *   **n_estimators**: 
                Jumlah pohon. Semakin banyak pohon, model semakin kuat, tapi komputasi lebih lama.
            *   **learning_rate**: 
                Penyusut kontribusi pohon baru. Nilai kecil mempermudah pencarian batas keputusan yang mulus, tapi membutuhkan `n_estimators` lebih tinggi.
            """)

    if submitted:
        with st.spinner("Sedang memproses preprocessing kustom dan melatih model AdaBoost..."):
            try:
                from group_trainer import run_custom_experiment
                # Jalankan eksperimen kustom
                custom_results = run_custom_experiment(
                    csv_filepath, target_col, 
                    scaler_type=scaler_choice, 
                    clip_outliers=clip_outliers, 
                    base_depth=base_depth, 
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate,
                    features_to_drop=selected_features_to_drop
                )
                
                st.success("🎉 Eksperimen berhasil diselesaikan!")
                
                # Membaca metrik model default yang sudah dilatih (AutoML) jika ada
                dataset_name = os.path.splitext(selected_csv)[0]
                if selected_csv == "sleep_health.csv":
                    model_exists = os.path.exists("models/adaboost_model.pkl")
                else:
                    model_exists = os.path.exists(f"models/group_{dataset_name}_model.pkl")
                
                # Menyiapkan kolom perbandingan
                st.subheader("📊 Hasil Perbandingan Performa Model")
                
                # A. Menampilkan Metrik Model Eksperimen Anda
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    st.markdown(f"""
                    <div style="background-color: rgba(74, 144, 226, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #4A90E2;">
                        <h4 style="color: #4A90E2; margin-top:0px;">🧪 Model Eksperimen Kustom</h4>
                        <p style="margin-bottom: 5px;">Akurasi Data Uji: <b>{custom_results['accuracy']*100:.2f}%</b></p>
                        <p style="margin-bottom: 5px;">ROC-AUC Score: <b>{custom_results['roc_auc']*100:.2f}%</b></p>
                        <p style="margin-bottom: 5px;">Mean Absolute Error (MAE): <b>{custom_results['mae']:.4f}</b></p>
                        <p style="margin-bottom: 5px;">Root Mean Squared Error (RMSE): <b>{custom_results['rmse']:.4f}</b></p>
                        <p style="margin-bottom: 5px;">MAPE (%)  : <b>{custom_results['mape']:.2f}%</b></p>
                        <p style="margin-bottom: 0px;">R² Score  : <b>{custom_results['r2']:.4f}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # B. Menampilkan Perbandingan dengan Model Default jika ada
                with col_m2:
                    if model_exists:
                        # Load model default metrics (we can run it with default parameters: scaler='minmax', clip=True, depth=1)
                        try:
                            # Mari hitung cepat model default untuk dibandingkan
                            default_results = run_custom_experiment(
                                csv_filepath, target_col,
                                scaler_type='minmax',
                                clip_outliers=True,
                                base_depth=1,
                                n_estimators=50,
                                learning_rate=1.0
                            )
                            st.markdown(f"""
                            <div style="background-color: rgba(80, 227, 194, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #50E3C2;">
                                <h4 style="color: #50E3C2; margin-top:0px;">🤖 Model Default (AutoML)</h4>
                                <p style="margin-bottom: 5px;">Akurasi Data Uji: <b>{default_results['accuracy']*100:.2f}%</b></p>
                                <p style="margin-bottom: 5px;">ROC-AUC Score: <b>{default_results['roc_auc']*100:.2f}%</b></p>
                                <p style="margin-bottom: 5px;">Mean Absolute Error (MAE): <b>{default_results['mae']:.4f}</b></p>
                                <p style="margin-bottom: 5px;">Root Mean Squared Error (RMSE): <b>{default_results['rmse']:.4f}</b></p>
                                <p style="margin-bottom: 5px;">MAPE (%)  : <b>{default_results['mape']:.2f}%</b></p>
                                <p style="margin-bottom: 0px;">R² Score  : <b>{default_results['r2']:.4f}</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # C. Visual perbandingan Akurasi & ROC-AUC
                            st.write("")
                            fig_exp, ax_exp = plt.subplots(figsize=(8, 3))
                            labels_chart = ['Akurasi (%)', 'ROC-AUC (%)']
                            custom_scores = [custom_results['accuracy']*100, custom_results['roc_auc']*100]
                            default_scores = [default_results['accuracy']*100, default_results['roc_auc']*100]
                            
                            x_c = np.arange(len(labels_chart))
                            b_w = 0.3
                            
                            ax_exp.bar(x_c - b_w/2, custom_scores, b_w, label='Eksperimen Kustom', color='#4A90E2')
                            ax_exp.bar(x_c + b_w/2, default_scores, b_w, label='AutoML Default', color='#50E3C2')
                            
                            ax_exp.set_ylabel('Skor (%)')
                            ax_exp.set_title('Eksperimen vs Default (Akurasi & ROC-AUC)', fontweight='bold')
                            ax_exp.set_xticks(x_c)
                            ax_exp.set_xticklabels(labels_chart)
                            ax_exp.set_ylim(0, 110)
                            ax_exp.legend()
                            sns.despine()
                            st.pyplot(fig_exp)
                            
                        except Exception as e_def:
                            st.info("💡 Tidak dapat membandingkan dengan model default karena kegagalan split: " + str(e_def))
                    else:
                        st.info("💡 Model default belum dilatih untuk dataset ini. Anda bisa melatih model default di menu 'Kelola & Latih Dataset'.")
                        
            except Exception as e:
                st.error(f"❌ Eksperimen gagal dijalankan: {e}")


if dashboard_type == "🌙 Dashboard Pribadi (Sleep Health)":
    # -----------------------------------------------------------------
    # MENU 1: DASHBOARD DATASET PRIBADI (EDA)
    # -----------------------------------------------------------------
    if menu == "📊 Dashboard Dataset (EDA)":
        st.markdown('<div class="main-title">Dashboard Analisis Kesehatan Tidur & Gaya Hidup</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Eksplorasi Data Analitis (EDA) dari Dataset Sleep Health and Lifestyle</div>', unsafe_allow_html=True)
        
        # 1. Metric Cards (Kolom Atas)
        col1, col2, col3, col4 = st.columns(4)
        
        total_data = len(df_raw)
        total_features = df_raw.shape[1] - 1  # Dikurangi Person ID
        missing_vals = df_raw.isnull().sum().sum()
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_data}</div>
                <div class="metric-label">Jumlah Data (Baris)</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_features}</div>
                <div class="metric-label">Jumlah Fitur</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{missing_vals}</div>
                <div class="metric-label">Missing Values</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">AdaBoost</div>
                <div class="metric-label">Algoritma Model</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.write("")
        
        # Tab layout untuk visualisasi agar rapi
        tab_dist, tab_hist, tab_corr = st.tabs([
            "🎯 Distribusi Target & Kategori", 
            "📊 Distribusi Fitur Numerik", 
            "🔥 Analisis Korelasi & Tabel Data"
        ])
        
        with tab_dist:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("Distribusi Tingkat Stress (Target)")
                stress_map = {0: 'Low', 1: 'Medium', 2: 'High'}
                df_temp = df_processed.copy()
                df_temp['Stress_Category'] = df_temp['Stress Level'].map(stress_map)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(
                    data=df_temp, x='Stress_Category', 
                    order=['Low', 'Medium', 'High'], 
                    palette=colors, ax=ax
                )
                ax.set_title("Jumlah Sampel Per Kategori Tingkat Stress", fontsize=11, fontweight='bold')
                ax.set_xlabel("Kategori Stress")
                ax.set_ylabel("Jumlah Responden")
                sns.despine()
                st.pyplot(fig)
                
            with col_chart2:
                st.subheader("Persentase Kategori Stress")
                fig, ax = plt.subplots(figsize=(5, 5))
                stress_counts = df_temp['Stress_Category'].value_counts()
                ax.pie(
                    stress_counts, labels=stress_counts.index, 
                    autopct='%1.1f%%', startangle=90, 
                    colors=colors,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
                )
                ax.axis('equal')  
                st.pyplot(fig)
     
        with tab_hist:
            col_hist1, col_hist2 = st.columns(2)
            
            with col_hist1:
                st.subheader("Distribusi Durasi Tidur")
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.histplot(
                    df_raw['Sleep Duration'], kde=True, 
                    color='#4A90E2', ax=ax, bins=15
                )
                ax.set_title("Histogram Durasi Tidur (Jam)", fontsize=11, fontweight='bold')
                ax.set_xlabel("Durasi Tidur (Jam)")
                ax.set_ylabel("Frekuensi")
                sns.despine()
                st.pyplot(fig)
                
            with col_hist2:
                st.subheader("Distribusi Langkah Harian (Daily Steps)")
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.histplot(
                    df_raw['Daily Steps'], kde=True, 
                    color='#50E3C2', ax=ax, bins=15
                )
                ax.set_title("Histogram Langkah Harian", fontsize=11, fontweight='bold')
                ax.set_xlabel("Langkah Harian")
                ax.set_ylabel("Frekuensi")
                sns.despine()
                st.pyplot(fig)
                
            st.subheader("Distribusi Usia Berdasarkan Kategori Stress")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.kdeplot(
                data=df_temp, x='Age', hue='Stress_Category', 
                palette=colors, fill=True, alpha=0.4, ax=ax
            )
            ax.set_xlabel("Usia (Tahun)")
            ax.set_ylabel("Density")
            sns.despine()
            st.pyplot(fig)
     
        with tab_corr:
            st.subheader("Heatmap Korelasi Fitur")
            st.write("Hubungan linear antar fitur setelah proses Label Encoding dan pemecahan Blood Pressure:")
            
            corr_cols = [c for c in df_processed.columns if not c.endswith('_Label')]
            corr_matrix = df_processed[corr_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_matrix, annot=True, fmt=".2f", 
                cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, annot_kws={"size": 8}
            )
            st.pyplot(fig)
            
            st.subheader("Lihat Sampel Data")
            st.dataframe(df_raw.head(10), use_container_width=True)

    # -----------------------------------------------------------------
    # MENU 2: SISTEM PREDIKSI REAL-TIME PRIBADI (SINGLE INPUT)
    # -----------------------------------------------------------------
    elif menu == "🔮 Prediksi Tingkat Stress":
        st.markdown('<div class="main-title">Sistem Prediksi Tingkat Stress Real-time</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Masukkan data kesehatan dan gaya hidup di bawah ini untuk memprediksi tingkat stress</div>', unsafe_allow_html=True)
        
        encoders = preprocessor['encoders']
        
        with st.form("prediction_form"):
            col_in1, col_in2, col_in3 = st.columns(3)
            
            with col_in1:
                gender = st.selectbox("Jenis Kelamin (Gender)", options=encoders['Gender'].classes_)
                age = st.slider("Usia (Age)", min_value=10, max_value=80, value=30)
                if 'Occupation' in encoders:
                    occupation = st.selectbox("Pekerjaan (Occupation)", options=encoders['Occupation'].classes_)
                else:
                    st.text_input("Pekerjaan (Occupation) [Tidak Digunakan Model]", value="Software Engineer", disabled=True, help="Fitur ini dibuang dari model utama untuk menghindari overfitting dan multikolinearitas.")
                    occupation = "Software Engineer"
                sleep_duration = st.slider("Durasi Tidur (Sleep Duration) dalam Jam", min_value=3.0, max_value=10.0, value=7.0, step=0.1)
                
            with col_in2:
                quality_sleep = st.slider("Kualitas Tidur (Quality of Sleep) [Skala 1-10]", min_value=1, max_value=10, value=7)
                phys_activity = st.slider("Tingkat Aktivitas Fisik (Menit/Hari)", min_value=10, max_value=150, value=60, step=5)
                bmi_cat = st.selectbox("Kategori BMI (BMI Category)", options=encoders['BMI Category'].classes_)
                heart_rate = st.slider("Denyut Jantung (Heart Rate) [bpm]", min_value=40, max_value=120, value=70)
                
            with col_in3:
                daily_steps = st.slider("Langkah Harian (Daily Steps)", min_value=1000, max_value=15000, value=6000, step=500)
                sleep_disorder = st.selectbox("Gangguan Tidur (Sleep Disorder)", options=encoders['Sleep Disorder'].classes_)
                bp_input = st.text_input("Tekanan Darah (Blood Pressure) [Systolic/Diastolic]", value="120/80", help="Format penulisan wajib: Systolic/Diastolic (Contoh: 120/80)")
            
            submitted = st.form_submit_button("Prediksi Tingkat Stress")
            
        if submitted:
            try:
                systolic, diastolic = map(int, bp_input.split('/'))
            except ValueError:
                st.error("❌ Format Tekanan Darah salah! Pastikan menggunakan format angka dipisah garis miring, contoh: `120/80`")
                st.stop()
                
            input_data = pd.DataFrame([{
                'Gender': gender,
                'Age': age,
                'Occupation': occupation,
                'Sleep Duration': sleep_duration,
                'Quality of Sleep': quality_sleep,
                'Physical Activity Level': phys_activity,
                'BMI Category': bmi_cat,
                'Heart Rate': heart_rate,
                'Daily Steps': daily_steps,
                'Sleep Disorder': sleep_disorder,
                'Systolic': systolic,
                'Diastolic': diastolic
            }])
            
            st.subheader("Data Masukan Pengguna")
            st.table(input_data)
            
            input_processed = input_data.copy()
            for col in preprocessor['categorical_cols']:
                le = encoders[col]
                input_processed[col] = le.transform(input_processed[col].astype(str))
                
            for col in preprocessor['numeric_cols']:
                lower_b, upper_b = preprocessor['clipping_bounds'][col]
                input_processed[col] = input_processed[col].clip(lower_b, upper_b)
                
            input_scaled = preprocessor['scaler'].transform(input_processed[preprocessor['feature_names']])
            
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            st.subheader("Hasil Prediksi")
            
            if prediction == 0:
                st.markdown("""
                <div class="result-box result-low">
                    <h3>LOW STRESS LEVEL (Tingkat Stress Rendah)</h3>
                    <p style="font-size: 1.1rem; margin-top: 10px;">
                        Tubuh dan gaya hidup Anda menunjukkan tanda-tanda rileks dan seimbang. Pertahankan kualitas tidur dan aktivitas fisik Anda!
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction == 1:
                st.markdown("""
                <div class="result-box result-medium">
                    <h3>MEDIUM STRESS LEVEL (Tingkat Stress Sedang)</h3>
                    <p style="font-size: 1.1rem; margin-top: 10px;">
                        Anda berada pada tingkat stress sedang. Cobalah beristirahat dengan cukup, atur waktu bekerja, dan luangkan waktu untuk relaksasi.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-box result-high">
                    <h3>HIGH STRESS LEVEL (Tingkat Stress Tinggi)</h3>
                    <p style="font-size: 1.1rem; margin-top: 10px;">
                        Peringatan! Tingkat stress Anda tergolong tinggi. Pertimbangkan untuk mengurangi beban kerja, memperbaiki durasi tidur, dan berkonsultasi dengan ahli kesehatan jika perlu.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.subheader("Probabilitas Klasifikasi Model:")
            col_prob1, col_prob2, col_prob3 = st.columns(3)
            
            with col_prob1:
                st.metric(label="Probabilitas Stress Rendah (Low)", value=f"{probabilities[0]*100:.2f}%")
                st.progress(float(probabilities[0]))
            with col_prob2:
                st.metric(label="Probabilitas Stress Sedang (Medium)", value=f"{probabilities[1]*100:.2f}%")
                st.progress(float(probabilities[1]))
            with col_prob3:
                st.metric(label="Probabilitas Stress Tinggi (High)", value=f"{probabilities[2]*100:.2f}%")
                st.progress(float(probabilities[2]))

    # -----------------------------------------------------------------
    # MENU 3: EVALUASI PRIBADI
    # -----------------------------------------------------------------
    elif menu == "📈 Evaluasi & Kinerja Model":
        st.markdown('<div class="main-title">Evaluasi Kinerja Model AdaBoost</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Analisis performa model klasifikasi berdasarkan GridSearchCV dan data uji</div>', unsafe_allow_html=True)
        
        st.subheader("🛠️ Hyperparameter & Metrik Terbaik (GridSearchCV)")
        col_eval1, col_eval2, col_eval3 = st.columns(3)
        
        with col_eval1:
            st.info(f"**Jumlah Estimator (n_estimators):**\n`{model.n_estimators}`")
        with col_eval2:
            st.info(f"**Learning Rate:**\n`{model.learning_rate}`")
        with col_eval3:
            algo = getattr(model, 'algorithm', 'SAMME (Default)')
            st.info(f"**Algoritma Adaboost:**\n`{algo}`")
            
        st.write("")
        
        df_eval = pd.read_csv('data/sleep_health_preprocessed.csv')
        X_eval = df_eval.drop(['Stress Level'] + [c for c in df_eval.columns if c.endswith('_Label')], axis=1, errors='ignore')
        y_eval = df_eval['Stress Level']
        
        from sklearn.model_selection import train_test_split
        _, X_test_eval, _, y_test_eval = train_test_split(
            X_eval, y_eval, test_size=0.2, random_state=42, stratify=y_eval
        )
        
        X_test_eval_scaled = preprocessor['scaler'].transform(X_test_eval[preprocessor['feature_names']])
        y_pred_eval = model.predict(X_test_eval_scaled)
        y_prob_eval = model.predict_proba(X_test_eval_scaled)
        
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
        cm = confusion_matrix(y_test_eval, y_pred_eval)
        acc = accuracy_score(y_test_eval, y_pred_eval)
        
        mae = mean_absolute_error(y_test_eval, y_pred_eval)
        rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_eval))
        r2 = r2_score(y_test_eval, y_pred_eval)
        
        y_test_mape = y_test_eval + 1
        y_pred_mape = y_pred_eval + 1
        mape = np.mean(np.abs((y_test_mape - y_pred_mape) / y_test_mape)) * 100
        
        roc_auc_val = roc_auc_score(y_test_eval, y_prob_eval, multi_class='ovr', average='macro')

        tab_metrics, tab_viz, tab_features = st.tabs([
            "📊 Metrik Evaluasi Lengkap",
            "🔮 Visualisasi Klasifikasi (Confusion Matrix & ROC)",
            "🔥 Analisis Fitur Penting"
        ])
        
        with tab_metrics:
            st.subheader("🎯 Metrik Utama Model (Data Uji)")
            col_main1, col_main2 = st.columns(2)
            with col_main1:
                st.metric(label="Akurasi Data Uji (Test Accuracy)", value=f"{acc*100:.2f}%")
            with col_main2:
                st.metric(label="ROC-AUC Score (One-vs-Rest Macro)", value=f"{roc_auc_val*100:.2f}%")
                
            st.write("")
            st.subheader("📐 Metrik Evaluasi Regresi Tambahan")
            st.write("*(Dihitung dengan menganalisis target tingkat stress ordinal 0: Low, 1: Medium, 2: High)*")
            
            col_reg1, col_reg2, col_reg3, col_reg4 = st.columns(4)
            with col_reg1:
                st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
            with col_reg2:
                st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.4f}")
            with col_reg3:
                st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{mape:.2f}%")
            with col_reg4:
                st.metric(label="R-squared (R² Score)", value=f"{r2:.4f}")
                
            st.write("")
            st.subheader("📝 Classification Report Lengkap")
            report_dict = classification_report(y_test_eval, y_pred_eval, target_names=['Low', 'Medium', 'High'], output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)

        with tab_viz:
            col_viz_left, col_viz_right = st.columns(2)
            
            with col_viz_left:
                st.subheader("🧩 Confusion Matrix Heatmap")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Low', 'Medium', 'High'],
                    yticklabels=['Low', 'Medium', 'High'],
                    ax=ax, cbar=False
                )
                ax.set_title("Confusion Matrix (Data Uji)", fontweight='bold')
                ax.set_xlabel("Prediksi Model")
                ax.set_ylabel("Kenyataan (Actual)")
                st.pyplot(fig)
                
            with col_viz_right:
                st.subheader("📈 Kurva ROC (Receiver Operating Characteristic)")
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_curve, auc
                
                y_test_bin = label_binarize(y_test_eval, classes=[0, 1, 2])
                
                fpr = dict()
                tpr = dict()
                roc_auc_dict = dict()
                class_colors = ['#11998e', '#f2994a', '#eb5757']
                classes_lbl = ['Low', 'Medium', 'High']
                
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                for i in range(3):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob_eval[:, i])
                    roc_auc_dict[i] = auc(fpr[i], tpr[i])
                    ax2.plot(fpr[i], tpr[i], color=class_colors[i], lw=2,
                             label=f'Kelas {classes_lbl[i]} (AUC = {roc_auc_dict[i]:.2f})')
                
                ax2.plot([0, 1], [0, 1], 'k--', lw=1.5)
                ax2.set_xlim([0.0, 1.0])
                ax2.set_ylim([0.0, 1.05])
                ax2.set_xlabel('False Positive Rate (FPR)')
                ax2.set_ylabel('True Positive Rate (TPR)')
                ax2.set_title('ROC Curve Per Kategori Stress', fontweight='bold')
                ax2.legend(loc="lower right")
                sns.despine()
                st.pyplot(fig2)

        with tab_features:
            st.subheader("🔥 Feature Importance (Tingkat Kepentingan Fitur)")
            st.write("Seberapa besar kontribusi masing-masing fitur dalam memprediksi tingkat stress:")
            
            importances = model.feature_importances_
            feature_names = preprocessor['feature_names']
            indices = np.argsort(importances)[::-1]
            
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.barplot(
                x=importances[indices], 
                y=[feature_names[i] for i in indices], 
                palette="viridis", ax=ax3
            )
            ax3.set_title("Tingkat Pentingnya Fitur (AdaBoost)", fontweight='bold')
            ax3.set_xlabel("Skor Kepentingan")
            sns.despine()
            st.pyplot(fig3)
            
            importance_df = pd.DataFrame({
                'Fitur': [feature_names[i] for i in indices],
                'Kontribusi (Importance Score)': importances[indices]
            })
            importance_df['Persentase Kontribusi'] = importance_df['Kontribusi (Importance Score)'].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(importance_df, use_container_width=True)

    # -----------------------------------------------------------------
    # MENU 4: PREDIKSI MASSAL PRIBADI (CSV)
    # -----------------------------------------------------------------
    elif menu == "📁 Prediksi Massal (CSV)":
        st.markdown('<div class="main-title">Sistem Prediksi Massal (Batch Prediction)</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Unggah file CSV baru berisi data kesehatan untuk diprediksi secara massal dan unduh hasilnya</div>', unsafe_allow_html=True)
        
        st.info("""
        ℹ️ **Petunjuk Format CSV:**
        File CSV yang diunggah harus memiliki kolom-kolom berikut:
        `Gender`, `Age`, `Occupation`, `Sleep Duration`, `Quality of Sleep`, `Physical Activity Level`, `BMI Category`, `Heart Rate`, `Daily Steps`, `Sleep Disorder`, `Blood Pressure`
        """)
        
        template_data = pd.DataFrame([{
            'Gender': 'Male',
            'Age': 32,
            'Occupation': 'Software Engineer',
            'Sleep Duration': 7.2,
            'Quality of Sleep': 8,
            'Physical Activity Level': 60,
            'BMI Category': 'Normal',
            'Heart Rate': 70,
            'Daily Steps': 8000,
            'Sleep Disorder': 'None',
            'Blood Pressure': '120/80'
        }])
        
        template_csv = template_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Template CSV 📥",
            data=template_csv,
            file_name="template_sleep_health.csv",
            mime="text/csv"
        )
        
        st.write("")
        uploaded_file = st.file_uploader("Pilih Berkas CSV Anda", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success("🎉 File berhasil diunggah!")
                
                required_cols = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 
                                 'Physical Activity Level', 'BMI Category', 'Heart Rate', 'Daily Steps', 
                                 'Sleep Disorder', 'Blood Pressure']
                
                missing_cols = [c for c in required_cols if c not in df_upload.columns]
                if len(missing_cols) > 0:
                    st.error(f"❌ Kolom berikut tidak ditemukan dalam file CSV: {missing_cols}")
                    st.stop()
                    
                st.subheader("Data yang Diunggah (Preview)")
                st.dataframe(df_upload.head(10), use_container_width=True)
                
                if st.button("Lakukan Prediksi Massal"):
                    with st.spinner("Sedang memproses prediksi massal..."):
                        df_prep = df_upload.copy()
                        if 'Person ID' in df_prep.columns:
                            df_prep.drop('Person ID', axis=1, inplace=True)
                            
                        bp_split = df_prep['Blood Pressure'].str.split('/', expand=True)
                        df_prep['Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
                        df_prep['Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')
                        df_prep['Systolic'] = df_prep['Systolic'].fillna(df_prep['Systolic'].median() if 'Systolic' in df_prep else 120)
                        df_prep['Diastolic'] = df_prep['Diastolic'].fillna(df_prep['Diastolic'].median() if 'Diastolic' in df_prep else 80)
                        df_prep.drop('Blood Pressure', axis=1, inplace=True)
                        
                        df_prep['Sleep Disorder'] = df_prep['Sleep Disorder'].fillna('None')
                        
                        encoders = preprocessor['encoders']
                        for col in preprocessor['categorical_cols']:
                            classes = list(encoders[col].classes_)
                            df_prep[col] = df_prep[col].astype(str).apply(lambda x: x if x in classes else classes[0])
                            df_prep[col] = encoders[col].transform(df_prep[col])
                            
                        for col in preprocessor['numeric_cols']:
                            lower_b, upper_b = preprocessor['clipping_bounds'][col]
                            df_prep[col] = df_prep[col].clip(lower_b, upper_b)
                            
                        features_ordered = preprocessor['feature_names']
                        df_scaled = preprocessor['scaler'].transform(df_prep[features_ordered])
                        
                        batch_preds = model.predict(df_scaled)
                        
                        stress_label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
                        df_upload['Predicted Stress Level Label'] = [stress_label_map[p] for p in batch_preds]
                        df_upload['Predicted Stress Level Class'] = batch_preds
                        
                        st.success("🎉 Prediksi Massal Selesai!")
                        
                        col_b1, col_b2 = st.columns([1, 2])
                        
                        with col_b1:
                            st.subheader("Distribusi Prediksi")
                            fig, ax = plt.subplots(figsize=(5, 4))
                            sns.countplot(
                                data=df_upload, x='Predicted Stress Level Label',
                                order=['Low', 'Medium', 'High'],
                                palette=colors, ax=ax
                            )
                            sns.despine()
                            st.pyplot(fig)
                            
                        with col_b2:
                            st.subheader("Tabel Hasil Prediksi")
                            st.dataframe(df_upload, use_container_width=True)
                            
                        result_csv = df_upload.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Hasil Prediksi 📥",
                            data=result_csv,
                            file_name="hasil_prediksi_stress.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat memproses file CSV: {e}")
                
    elif menu == "🧪 Ruang Eksperimen ML":
        render_ml_experiment_room()
                
    elif menu == "📚 Teori & Cara Kerja AdaBoost":
        render_adaboost_theory()

else:
    # =================================================================
    # DASHBOARD GABUNGAN KELOMPOK (MULTI-DATASET AUTOML)
    # =================================================================
    import glob
    from sklearn.model_selection import train_test_split
    
    # -----------------------------------------------------------------
    # MENU KELOMPOK 1: KELOLA & LATIH DATASET
    # -----------------------------------------------------------------
    if menu_kelompok == "📁 Kelola & Latih Dataset":
        st.markdown('<div class="main-title">Kelola & Latih Dataset Kelompok</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Unggah dataset baru dari teman kelompok Anda dan latih model AdaBoost secara otomatis</div>', unsafe_allow_html=True)
        
        # Area Upload
        st.subheader("📥 Unggah File Dataset Kelompok Baru")
        uploaded_dataset = st.file_uploader("Pilih file CSV dataset baru Anda", type=["csv"], help="Silakan unggah file CSV di sini untuk didaftarkan ke sistem dashboard.")
        
        if uploaded_dataset is not None:
            filename = uploaded_dataset.name
            save_path = os.path.join("data", filename)
            with open(save_path, "wb") as f:
                f.write(uploaded_dataset.getbuffer())
            st.success(f"🎉 Dataset `{filename}` berhasil diunggah dan disimpan ke folder `data/`!")
            st.rerun()
            
        st.write("")
        st.markdown("---")
        
        # Scaning files
        csv_files = glob.glob("data/*.csv")
        dataset_options = []
        for f in csv_files:
            basename = os.path.basename(f)
            if not (basename.endswith("_preprocessed.csv") or basename.startswith("template_") or basename == "hasil_prediksi_stress.csv"):
                dataset_options.append(basename)
                
        if not dataset_options:
            st.info("💡 Belum ada dataset kelompok yang terdaftar. Silakan unggah file CSV Anda menggunakan uploader di atas.")
        else:
            st.subheader("⚙️ Pilih dan Latih Dataset Terdaftar")
            selected_csv = st.selectbox("Pilih Dataset yang ingin dikelola/dilatih:", options=dataset_options)
            
            csv_filepath = os.path.join("data", selected_csv)
            df_temp = pd.read_csv(csv_filepath)
            
            st.markdown(f"**Ukuran Dataset:** `{df_temp.shape[0]}` baris, `{df_temp.shape[1]}` kolom.")
            st.dataframe(df_temp.head(5), use_container_width=True)
            
            # Memilih kolom target
            cols = df_temp.columns.tolist()
            default_idx = len(cols) - 1
            for i, c in enumerate(cols):
                if c.lower() in ['target', 'outcome', 'class', 'label', 'source', 'stress level', 'classification']:
                    default_idx = i
                    break
            target_col = st.selectbox("Pilih Kolom Target (Label yang akan diprediksi):", options=cols, index=default_idx)
            
            # Cek status model
            dataset_name = os.path.splitext(selected_csv)[0]
            model_exists = os.path.exists(f"models/group_{dataset_name}_model.pkl")
            
            if model_exists:
                st.success(f"✅ Model AdaBoost untuk dataset `{selected_csv}` sudah dilatih dan siap digunakan.")
            else:
                st.warning(f"⚠️ Model untuk dataset `{selected_csv}` belum dilatih. Klik tombol di bawah untuk melatih model.")
                
            if st.button("Latih Model AdaBoost untuk Dataset Ini 🚀"):
                with st.spinner("Sedang memproses pembersihan data, imputation, label encoding, scaling, IQR outlier clipping, dan pelatihan AdaBoost..."):
                    try:
                        from group_trainer import train_custom_dataset
                        eval_results = train_custom_dataset(csv_filepath, target_col)
                        st.success(f"🎉 Model AdaBoost untuk dataset `{selected_csv}` berhasil dilatih dengan sukses!")
                        st.metric("Akurasi Data Uji", f"{eval_results['accuracy']*100:.2f}%")
                        st.metric("ROC-AUC Score", f"{eval_results['roc_auc']*100:.2f}%")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Terjadi kesalahan saat melatih model: {e}")

    # -----------------------------------------------------------------
    # MENU KELOMPOK 2: ANALISIS DATA (EDA) KELOMPOK
    # -----------------------------------------------------------------
    elif menu_kelompok == "📊 Analisis Data (EDA)":
        st.markdown('<div class="main-title">Analisis Data (EDA) Kelompok</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Eksplorasi data analitis untuk dataset kelompok yang dipilih</div>', unsafe_allow_html=True)
        
        preprocessed_files = glob.glob("data/group_*_preprocessed.csv")
        options = [os.path.basename(f).replace("group_", "").replace("_preprocessed.csv", "") for f in preprocessed_files]
        
        if not options:
            st.info("💡 Belum ada dataset kelompok yang dilatih. Silakan latih dataset terlebih dahulu di menu 'Kelola & Latih Dataset'.")
        else:
            selected_dataset = st.selectbox("Pilih Dataset Kelompok:", options=options)
            
            prep_path = f"data/group_{selected_dataset}_preprocessed.csv"
            df_prep = pd.read_csv(prep_path)
            
            prep_meta_path = f"models/group_{selected_dataset}_prep.pkl"
            preprocessor_meta = joblib.load(prep_meta_path)
            target_col = preprocessor_meta['target_col']
            
            # Metric Cards
            col1, col2, col3 = st.columns(3)
            total_data = len(df_prep)
            total_features = len(preprocessor_meta['feature_names'])
            num_classes = len(preprocessor_meta['classes'])
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_data}</div>
                    <div class="metric-label">Jumlah Data (Baris)</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-value">{total_features}</div>
                <div class="metric-label">Jumlah Fitur</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{num_classes}</div>
                    <div class="metric-label">Jumlah Kelas Target ({target_col})</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            
            tab_dist, tab_corr, tab_raw = st.tabs(["🎯 Distribusi Target", "🔥 Korelasi Fitur", "📋 Sampel Data"])
            
            with tab_dist:
                st.subheader(f"Distribusi Kolom Target `{target_col}`")
                fig, ax = plt.subplots(figsize=(8, 4))
                
                target_series = df_prep[target_col]
                if preprocessor_meta['target_encoder'] is not None:
                    try:
                        labels = preprocessor_meta['target_encoder'].inverse_transform(target_series)
                        sns.countplot(x=labels, ax=ax, palette="viridis")
                    except Exception:
                        sns.countplot(x=target_series, ax=ax, palette="viridis")
                else:
                    sns.countplot(x=target_series, ax=ax, palette="viridis")
                    
                ax.set_title(f"Jumlah Sampel Per Kelas Target: {target_col}", fontsize=11, fontweight='bold')
                ax.set_xlabel("Kelas Target")
                ax.set_ylabel("Frekuensi")
                sns.despine()
                st.pyplot(fig)
                
            with tab_corr:
                st.subheader("Heatmap Korelasi Fitur Numerik")
                numeric_cols = preprocessor_meta['numeric_cols']
                if numeric_cols:
                    corr_matrix = df_prep[numeric_cols + [target_col]].corr()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
                    st.pyplot(fig)
                else:
                    st.info("Tidak ada fitur numerik di dataset ini untuk dihitung korelasinya.")
                    
            with tab_raw:
                st.subheader("Sampel Data Hasil Preprocessing")
                st.dataframe(df_prep.head(15), use_container_width=True)

    # -----------------------------------------------------------------
    # MENU KELOMPOK 3: PREDIKSI REAL-TIME KELOMPOK
    # -----------------------------------------------------------------
    elif menu_kelompok == "🔮 Prediksi Real-time":
        st.markdown('<div class="main-title">Prediksi Real-time Kelompok</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Prediksi kelas secara real-time menggunakan model AdaBoost kelompok yang dipilih</div>', unsafe_allow_html=True)
        
        preprocessed_files = glob.glob("data/group_*_preprocessed.csv")
        options = [os.path.basename(f).replace("group_", "").replace("_preprocessed.csv", "") for f in preprocessed_files]
        
        if not options:
            st.info("💡 Belum ada dataset kelompok yang dilatih. Silakan latih dataset terlebih dahulu di menu 'Kelola & Latih Dataset'.")
        else:
            selected_dataset = st.selectbox("Pilih Dataset Kelompok:", options=options)
            
            model_path = f"models/group_{selected_dataset}_model.pkl"
            prep_path = f"models/group_{selected_dataset}_prep.pkl"
            
            model_group = joblib.load(model_path)
            prep_meta = joblib.load(prep_path)
            
            target_col = prep_meta['target_col']
            feature_names = prep_meta['feature_names']
            categorical_cols = prep_meta['categorical_cols']
            numeric_cols = prep_meta['numeric_cols']
            encoders = prep_meta['encoders']
            
            st.write("")
            st.subheader(f"Form Input Data Baru untuk Prediksi `{selected_dataset}`")
            
            orig_data_path = f"data/{selected_dataset}.csv"
            df_orig = pd.read_csv(orig_data_path)
            
            input_dict = {}
            with st.form("group_prediction_form"):
                col_list = st.columns(2)
                
                for i, feat in enumerate(feature_names):
                    col_idx = i % 2
                    with col_list[col_idx]:
                        if feat in categorical_cols:
                            classes = list(encoders[feat].classes_)
                            input_dict[feat] = st.selectbox(f"{feat}", options=classes)
                        else:
                            # Mengambil range
                            min_v = float(df_orig[feat].min())
                            max_v = float(df_orig[feat].max())
                            mean_v = float(df_orig[feat].mean())
                            
                            if min_v == max_v:
                                input_dict[feat] = st.number_input(f"{feat}", value=min_v)
                            else:
                                if max_v - min_v > 1000:
                                    input_dict[feat] = st.slider(f"{feat}", min_value=min_v, max_value=max_v, value=mean_v, step=10.0)
                                else:
                                    step_size = 0.1 if df_orig[feat].dtype == 'float64' else 1.0
                                    input_dict[feat] = st.slider(f"{feat}", min_value=min_v, max_value=max_v, value=mean_v, step=step_size)
                                    
                submitted = st.form_submit_button(f"Prediksi {target_col}")
                
            if submitted:
                input_df = pd.DataFrame([input_dict])
                st.subheader("Data Masukan Pengguna")
                st.table(input_df)
                
                input_processed = input_df.copy()
                for col in categorical_cols:
                    le = encoders[col]
                    input_processed[col] = le.transform(input_processed[col].astype(str))
                    
                for col in numeric_cols:
                    lower_b, upper_b = prep_meta['clipping_bounds'][col]
                    input_processed[col] = input_processed[col].clip(lower_b, upper_b)
                    
                input_scaled = prep_meta['scaler'].transform(input_processed[feature_names])
                
                pred_class = model_group.predict(input_scaled)[0]
                probabilities = model_group.predict_proba(input_scaled)[0]
                
                st.subheader("Hasil Prediksi")
                
                if prep_meta['target_encoder'] is not None:
                    pred_label = prep_meta['target_encoder'].inverse_transform([pred_class])[0]
                else:
                    pred_label = str(pred_class)
                    
                st.markdown(f"""
                <div class="result-box result-low" style="background: linear-gradient(135deg, #11998e, #38ef7d);">
                    <h3>HASIL PREDIKSI: {pred_label}</h3>
                    <p style="font-size: 1.1rem; margin-top: 10px;">
                        Model AdaBoost berhasil memprediksi data masukan Anda ke dalam kelas target <b>{pred_label}</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.subheader("Probabilitas Klasifikasi Model:")
                classes_lbls = prep_meta['classes']
                if prep_meta['target_encoder'] is not None:
                    classes_lbls = list(prep_meta['target_encoder'].classes_)
                    
                col_probs = st.columns(len(classes_lbls))
                for idx, prob in enumerate(probabilities):
                    with col_probs[idx]:
                        lbl = classes_lbls[idx]
                        st.metric(label=f"Probabilitas {lbl}", value=f"{prob*100:.2f}%")
                        st.progress(float(prob))

    # -----------------------------------------------------------------
    # MENU KELOMPOK 4: EVALUASI MODEL KELOMPOK
    # -----------------------------------------------------------------
    elif menu_kelompok == "📈 Evaluasi Model":
        st.markdown('<div class="main-title">Evaluasi Kinerja Model Kelompok</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Analisis performa model AdaBoost untuk dataset kelompok yang dipilih</div>', unsafe_allow_html=True)
        
        preprocessed_files = glob.glob("data/group_*_preprocessed.csv")
        options = [os.path.basename(f).replace("group_", "").replace("_preprocessed.csv", "") for f in preprocessed_files]
        
        if not options:
            st.info("💡 Belum ada dataset kelompok yang dilatih. Silakan latih dataset terlebih dahulu di menu 'Kelola & Latih Dataset'.")
        else:
            selected_dataset = st.selectbox("Pilih Dataset Kelompok:", options=options)
            
            model_path = f"models/group_{selected_dataset}_model.pkl"
            prep_path = f"models/group_{selected_dataset}_prep.pkl"
            
            model_group = joblib.load(model_path)
            prep_meta = joblib.load(prep_path)
            
            df_eval = pd.read_csv(f"data/{selected_dataset}.csv")
            
            cols_to_drop = [col for col in df_eval.columns if 'id' in col.lower() or 'index' in col.lower() or 'person' in col.lower()]
            target_col = prep_meta['target_col']
            if target_col in cols_to_drop:
                cols_to_drop.remove(target_col)
            if cols_to_drop:
                df_eval.drop(columns=cols_to_drop, inplace=True)
                
            X_eval = df_eval.drop(columns=[target_col])
            y_eval = df_eval[target_col]
            
            # Hapus baris yang targetnya NaN agar evaluasi akurat
            valid_mask = y_eval.notna()
            X_eval = X_eval[valid_mask]
            y_eval = y_eval[valid_mask]
            
            if prep_meta['target_encoder'] is not None:
                le_t = prep_meta['target_encoder']
                classes_t = list(le_t.classes_)
                y_eval = y_eval.astype(str).apply(lambda x: x if x in classes_t else classes_t[0])
                y_eval = le_t.transform(y_eval)
                
            try:
                _, X_test_eval, _, y_test_eval = train_test_split(
                    X_eval, y_eval, test_size=0.2, random_state=42, stratify=y_eval
                )
            except Exception:
                _, X_test_eval, _, y_test_eval = train_test_split(
                    X_eval, y_eval, test_size=0.2, random_state=42
                )
                
            X_test_eval_processed = X_test_eval.copy()
            for col in prep_meta['categorical_cols']:
                le = prep_meta['encoders'][col]
                # Imputasi NaN pada data uji dengan mode/kelas pertama agar le.transform tidak error 'nan'
                mode_val = X_test_eval_processed[col].mode().iloc[0] if not X_test_eval_processed[col].mode().empty else le.classes_[0]
                X_test_eval_processed[col] = X_test_eval_processed[col].fillna(mode_val)
                # Safeguard terhadap label baru yang tidak dikenal
                classes = list(le.classes_)
                X_test_eval_processed[col] = X_test_eval_processed[col].astype(str).apply(
                    lambda x: x if x in classes else classes[0]
                )
                X_test_eval_processed[col] = le.transform(X_test_eval_processed[col])
                
            for col in prep_meta['numeric_cols']:
                # Imputasi NaN pada data uji dengan median dari kolom tersebut
                median_val = X_test_eval_processed[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X_test_eval_processed[col] = X_test_eval_processed[col].fillna(median_val)
                
                lower_b, upper_b = prep_meta['clipping_bounds'][col]
                X_test_eval_processed[col] = X_test_eval_processed[col].clip(lower_b, upper_b)
                
            X_test_eval_scaled = prep_meta['scaler'].transform(X_test_eval_processed[prep_meta['feature_names']])
            
            y_pred_eval = model_group.predict(X_test_eval_scaled)
            y_prob_eval = model_group.predict_proba(X_test_eval_scaled)
            
            from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
            acc = accuracy_score(y_test_eval, y_pred_eval)
            
            classes_lbls = prep_meta['classes']
            if prep_meta['target_encoder'] is not None:
                classes_lbls = list(prep_meta['target_encoder'].classes_)
                labels_to_show = list(range(len(classes_lbls)))
                target_names_to_show = [str(c) for c in classes_lbls]
            else:
                try:
                    orig_type = type(y_test_eval.iloc[0]) if hasattr(y_test_eval, 'iloc') else type(y_test_eval[0])
                    labels_to_show = [orig_type(c) for c in classes_lbls]
                    target_names_to_show = [str(c) for c in classes_lbls]
                except Exception:
                    labels_to_show = sorted(list(np.unique(np.concatenate([y_test_eval, y_pred_eval]))))
                    target_names_to_show = [str(c) for c in labels_to_show]

            n_classes_model = len(labels_to_show)
            if n_classes_model == 2:
                try:
                    roc_auc_val = roc_auc_score(y_test_eval, y_prob_eval[:, 1])
                except ValueError:
                    pos_lbl = labels_to_show[1]
                    roc_auc_val = roc_auc_score(y_test_eval, y_prob_eval[:, 1], pos_label=pos_lbl)
            else:
                try:
                    roc_auc_val = roc_auc_score(y_test_eval, y_prob_eval, multi_class='ovr', average='macro', labels=labels_to_show)
                except ValueError:
                    from sklearn.preprocessing import label_binarize
                    y_test_bin = label_binarize(y_test_eval, classes=labels_to_show)
                    roc_auc_val = roc_auc_score(y_test_bin, y_prob_eval, multi_class='ovr', average='macro')
                
            cm = confusion_matrix(y_test_eval, y_pred_eval, labels=labels_to_show)
            
            mae = mean_absolute_error(y_test_eval, y_pred_eval)
            rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_eval))
            r2 = r2_score(y_test_eval, y_pred_eval)
            
            y_test_mape = y_test_eval + 1
            y_pred_mape = y_pred_eval + 1
            mape = np.mean(np.abs((y_test_mape - y_pred_mape) / y_test_mape)) * 100
            
            tab_eval_metrics, tab_eval_viz = st.tabs(["📊 Metrik Lengkap", "🔮 Confusion Matrix & ROC"])
            
            with tab_eval_metrics:
                st.subheader("🎯 Metrik Utama Model")
                col_eval1, col_eval2 = st.columns(2)
                with col_eval1:
                    st.metric("Akurasi Data Uji", f"{acc*100:.2f}%")
                with col_eval2:
                    st.metric("ROC-AUC Score", f"{roc_auc_val*100:.2f}%")
                    
                st.write("")
                st.subheader("📐 Metrik Evaluasi Regresi Tambahan")
                col_reg1, col_reg2, col_reg3, col_reg4 = st.columns(4)
                with col_reg1:
                    st.metric("MAE", f"{mae:.4f}")
                with col_reg2:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col_reg3:
                    st.metric("MAPE", f"{mape:.2f}%")
                with col_reg4:
                    st.metric("R² Score", f"{r2:.4f}")
                    
                st.write("")
                st.subheader("📝 Classification Report")
                report_dict = classification_report(
                    y_test_eval, 
                    y_pred_eval, 
                    labels=labels_to_show, 
                    target_names=target_names_to_show, 
                    zero_division=0,
                    output_dict=True
                )
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4), use_container_width=True)
                
            with tab_eval_viz:
                col_viz_l, col_viz_r = st.columns(2)
                with col_viz_l:
                    st.subheader("🧩 Confusion Matrix Heatmap")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(
                        cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=target_names_to_show,
                        yticklabels=target_names_to_show,
                        ax=ax, cbar=False
                    )
                    ax.set_title("Confusion Matrix (Data Uji)", fontweight='bold')
                    ax.set_xlabel("Prediksi Model")
                    ax.set_ylabel("Kenyataan (Actual)")
                    st.pyplot(fig)
                    
                with col_viz_r:
                    st.subheader("📈 Kurva ROC")
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    
                    if n_classes_model == 2:
                        from sklearn.metrics import roc_curve, auc
                        pos_lbl = labels_to_show[1]
                        fpr_b, tpr_b, _ = roc_curve(y_test_eval, y_prob_eval[:, 1], pos_label=pos_lbl)
                        auc_b = auc(fpr_b, tpr_b)
                        ax2.plot(fpr_b, tpr_b, color='#11998e', lw=2, label=f'ROC curve (AUC = {auc_b:.2f})')
                    else:
                        from sklearn.preprocessing import label_binarize
                        from sklearn.metrics import roc_curve, auc
                        y_test_bin = label_binarize(y_test_eval, classes=labels_to_show)
                        class_colors = ['#11998e', '#f2994a', '#eb5757', '#9b59b6', '#34495e']
                        
                        for i in range(n_classes_model):
                            if np.sum(y_test_bin[:, i]) > 0:
                                fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob_eval[:, i])
                                auc_i = auc(fpr_i, tpr_i)
                                lbl_i = classes_lbls[i]
                                color_i = class_colors[i % len(class_colors)]
                                ax2.plot(fpr_i, tpr_i, color=color_i, lw=2, label=f'Kelas {lbl_i} (AUC = {auc_i:.2f})')
                            
                    ax2.plot([0, 1], [0, 1], 'k--', lw=1.5)
                    ax2.set_xlim([0.0, 1.0])
                    ax2.set_ylim([0.0, 1.05])
                    ax2.set_xlabel('False Positive Rate (FPR)')
                    ax2.set_ylabel('True Positive Rate (TPR)')
                    ax2.set_title('ROC Curve', fontweight='bold')
                    ax2.legend(loc="lower right")
                    sns.despine()
                    st.pyplot(fig2)

    # -----------------------------------------------------------------
    # MENU KELOMPOK 5: PERBANDINGAN & KESIMPULAN KELOMPOK
    # -----------------------------------------------------------------
    elif menu_kelompok == "🏆 Perbandingan & Kesimpulan":
        st.markdown('<div class="main-title">🏆 Perbandingan Kinerja & Kesimpulan Model</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Leaderboard performa model AdaBoost Classifier di seluruh dataset kelompok Anda</div>', unsafe_allow_html=True)
        
        # Logika Pengumpulan Metrik Evaluasi secara Dinamis
        metrics_list = []
        
        # A. Evaluasi Dataset Utama Pribadi (Sleep Health)
        main_model_path = 'models/adaboost_model.pkl'
        main_prep_path = 'models/preprocessor.pkl'
        main_data_path = 'data/sleep_health.csv'
        
        if os.path.exists(main_model_path) and os.path.exists(main_prep_path) and os.path.exists(main_data_path):
            try:
                model_main = joblib.load(main_model_path)
                prep_main = joblib.load(main_prep_path)
                df_main = pd.read_csv(main_data_path)
                
                if 'Person ID' in df_main.columns:
                    df_main.drop('Person ID', axis=1, inplace=True)
                if 'Blood Pressure' in df_main.columns:
                    bp_split = df_main['Blood Pressure'].str.split('/', expand=True)
                    df_main['Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
                    df_main['Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')
                    df_main['Systolic'] = df_main['Systolic'].fillna(df_main['Systolic'].median())
                    df_main['Diastolic'] = df_main['Diastolic'].fillna(df_main['Diastolic'].median())
                    df_main.drop('Blood Pressure', axis=1, inplace=True)
                if 'Sleep Disorder' in df_main.columns:
                    df_main['Sleep Disorder'] = df_main['Sleep Disorder'].fillna('None')
                    
                def transform_stress_level(level):
                    if level <= 3:
                        return 0
                    elif level <= 6:
                        return 1
                    else:
                        return 2
                df_main['Stress Level'] = df_main['Stress Level'].apply(transform_stress_level)
                
                X_m = df_main.drop('Stress Level', axis=1)
                y_m = df_main['Stress Level']
                
                _, X_test_m, _, y_test_m = train_test_split(X_m, y_m, test_size=0.2, random_state=42, stratify=y_m)
                
                X_test_m_proc = X_test_m.copy()
                for col in prep_main['categorical_cols']:
                    le = prep_main['encoders'][col]
                    X_test_m_proc[col] = le.transform(X_test_m_proc[col].astype(str))
                for col in prep_main['numeric_cols']:
                    lower_b, upper_b = prep_main['clipping_bounds'][col]
                    X_test_m_proc[col] = X_test_m_proc[col].clip(lower_b, upper_b)
                    
                X_test_m_scaled = prep_main['scaler'].transform(X_test_m_proc[prep_main['feature_names']])
                
                y_pred_m = model_main.predict(X_test_m_scaled)
                y_prob_m = model_main.predict_proba(X_test_m_scaled)
                
                from sklearn.metrics import accuracy_score, roc_auc_score
                acc_m = accuracy_score(y_test_m, y_pred_m)
                roc_auc_m = roc_auc_score(y_test_m, y_prob_m, multi_class='ovr', average='macro')
                
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                mae_m = mean_absolute_error(y_test_m, y_pred_m)
                rmse_m = np.sqrt(mean_squared_error(y_test_m, y_pred_m))
                r2_m = r2_score(y_test_m, y_pred_m)
                y_test_mape_m = y_test_m + 1
                y_pred_mape_m = y_pred_m + 1
                mape_m = np.mean(np.abs((y_test_mape_m - y_pred_mape_m) / y_test_mape_m)) * 100
                
                metrics_list.append({
                    'Nama Dataset': 'Sleep Health (Pribadi)',
                    'Target (Label)': 'Stress Level',
                    'Akurasi': acc_m,
                    'ROC-AUC': roc_auc_m,
                    'MAE': mae_m,
                    'RMSE': rmse_m,
                    'MAPE (%)': mape_m,
                    'R² Score': r2_m,
                    'Hyperparameter': f"n_est: {model_main.n_estimators}, lr: {model_main.learning_rate}"
                })
            except Exception as e:
                pass
                
        # B. Evaluasi Dataset Kelompok Terdaftar
        group_prep_files = glob.glob("models/group_*_prep.pkl")
        for prep_file in group_prep_files:
            try:
                basename_f = os.path.basename(prep_file)
                ds_name = basename_f.replace("group_", "").replace("_prep.pkl", "")
                model_file = f"models/group_{ds_name}_model.pkl"
                data_file = f"data/{ds_name}.csv"
                
                if os.path.exists(model_file) and os.path.exists(data_file):
                    model_g = joblib.load(model_file)
                    prep_g = joblib.load(prep_file)
                    df_g = pd.read_csv(data_file)
                    
                    cols_to_drop = [col for col in df_g.columns if 'id' in col.lower() or 'index' in col.lower() or 'person' in col.lower()]
                    target_col = prep_g['target_col']
                    if target_col in cols_to_drop:
                        cols_to_drop.remove(target_col)
                    if cols_to_drop:
                        df_g.drop(columns=cols_to_drop, inplace=True)
                        
                    X_g = df_g.drop(columns=[target_col])
                    y_g = df_g[target_col]
                    
                    # Hapus baris yang targetnya NaN agar evaluasi akurat
                    valid_mask_g = y_g.notna()
                    X_g = X_g[valid_mask_g]
                    y_g = y_g[valid_mask_g]
                    
                    if prep_g['target_encoder'] is not None:
                        le_t = prep_g['target_encoder']
                        classes_t = list(le_t.classes_)
                        y_g = y_g.astype(str).apply(lambda x: x if x in classes_t else classes_t[0])
                        y_g = le_t.transform(y_g)
                        
                    try:
                        _, X_test_g, _, y_test_g = train_test_split(X_g, y_g, test_size=0.2, random_state=42, stratify=y_g)
                    except Exception:
                        _, X_test_g, _, y_test_g = train_test_split(X_g, y_g, test_size=0.2, random_state=42)
                        
                    X_test_g_proc = X_test_g.copy()
                    for col in prep_g['categorical_cols']:
                        le = prep_g['encoders'][col]
                        # Imputasi NaN pada data uji dengan mode/kelas pertama agar le.transform tidak error 'nan'
                        mode_val = X_test_g_proc[col].mode().iloc[0] if not X_test_g_proc[col].mode().empty else le.classes_[0]
                        X_test_g_proc[col] = X_test_g_proc[col].fillna(mode_val)
                        # Safeguard terhadap label baru yang tidak dikenal
                        classes = list(le.classes_)
                        X_test_g_proc[col] = X_test_g_proc[col].astype(str).apply(
                            lambda x: x if x in classes else classes[0]
                        )
                        X_test_g_proc[col] = le.transform(X_test_g_proc[col])
                    for col in prep_g['numeric_cols']:
                        # Imputasi NaN pada data uji dengan median dari kolom tersebut
                        median_val = X_test_g_proc[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0
                        X_test_g_proc[col] = X_test_g_proc[col].fillna(median_val)
                        
                        lower_b, upper_b = prep_g['clipping_bounds'][col]
                        X_test_g_proc[col] = X_test_g_proc[col].clip(lower_b, upper_b)
                        
                    X_test_g_scaled = prep_g['scaler'].transform(X_test_g_proc[prep_g['feature_names']])
                    y_pred_g = model_g.predict(X_test_g_scaled)
                    y_prob_g = model_g.predict_proba(X_test_g_scaled)
                    
                    from sklearn.metrics import accuracy_score, roc_auc_score
                    acc_g = accuracy_score(y_test_g, y_pred_g)
                    n_classes = len(np.unique(y_test_g))
                    if n_classes == 2:
                        roc_auc_g = roc_auc_score(y_test_g, y_prob_g[:, 1])
                    else:
                        roc_auc_g = roc_auc_score(y_test_g, y_prob_g, multi_class='ovr', average='macro')
                        
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    mae_g = mean_absolute_error(y_test_g, y_pred_g)
                    rmse_g = np.sqrt(mean_squared_error(y_test_g, y_pred_g))
                    r2_g = r2_score(y_test_g, y_pred_g)
                    y_test_mape_g = y_test_g + 1
                    y_pred_mape_g = y_pred_g + 1
                    mape_g = np.mean(np.abs((y_test_mape_g - y_pred_mape_g) / y_test_mape_g)) * 100
                    
                    metrics_list.append({
                        'Nama Dataset': ds_name.replace("_", " ").title(),
                        'Target (Label)': target_col,
                        'Akurasi': acc_g,
                        'ROC-AUC': roc_auc_g,
                        'MAE': mae_g,
                        'RMSE': rmse_g,
                        'MAPE (%)': mape_g,
                        'R² Score': r2_g,
                        'Hyperparameter': f"n_est: {model_g.n_estimators}, lr: {model_g.learning_rate}"
                    })
            except Exception as e:
                pass
                
        # Tampilkan Hasil
        if not metrics_list:
            st.info("💡 Belum ada model kelompok yang selesai dilatih. Silakan latih model terlebih dahulu pada menu 'Kelola & Latih Dataset'.")
        else:
            df_comparison = pd.DataFrame(metrics_list)
            # Urutkan berdasarkan akurasi tertinggi
            df_comparison = df_comparison.sort_values(by='Akurasi', ascending=False).reset_index(drop=True)
            
            # Format tampilan DataFrame
            df_styled = df_comparison.copy()
            df_styled['Akurasi'] = df_styled['Akurasi'].apply(lambda x: f"{x*100:.2f}%")
            df_styled['ROC-AUC'] = df_styled['ROC-AUC'].apply(lambda x: f"{x*100:.2f}%")
            df_styled['MAE'] = df_styled['MAE'].apply(lambda x: f"{x:.4f}")
            df_styled['RMSE'] = df_styled['RMSE'].apply(lambda x: f"{x:.4f}")
            df_styled['MAPE (%)'] = df_styled['MAPE (%)'].apply(lambda x: f"{x:.2f}%")
            df_styled['R² Score'] = df_styled['R² Score'].apply(lambda x: f"{x:.4f}")
            
            st.subheader("🏆 Leaderboard Performa Model AdaBoost")
            st.write("Daftar performa model AdaBoost pada seluruh dataset yang diurutkan dari akurasi tertinggi:")
            st.dataframe(df_styled, use_container_width=True)
            
            st.write("")
            
            # Visualisasi Perbandingan
            st.subheader("📊 Visualisasi Perbandingan Akurasi vs ROC-AUC")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Buat bar chart perbandingan
            x_indices = np.arange(len(df_comparison))
            bar_width = 0.35
            
            ax.bar(x_indices - bar_width/2, df_comparison['Akurasi'] * 100, bar_width, label='Akurasi (%)', color='#77A6F7')
            ax.bar(x_indices + bar_width/2, df_comparison['ROC-AUC'] * 100, bar_width, label='ROC-AUC (%)', color='#50E3C2')
            
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Persentase (%)')
            ax.set_title('Perbandingan Akurasi dan ROC-AUC Model AdaBoost', fontweight='bold')
            ax.set_xticks(x_indices)
            ax.set_xticklabels(df_comparison['Nama Dataset'], rotation=15)
            ax.set_ylim(0, 110)
            ax.legend()
            sns.despine()
            st.pyplot(fig)
            
            # Kesimpulan Akhir Dinamis
            st.write("")
            st.subheader("📝 Kesimpulan Akademis Presentasi")
            
            best_dataset = df_comparison.loc[0, 'Nama Dataset']
            best_accuracy = df_comparison.loc[0, 'Akurasi']
            best_auc = df_comparison.loc[0, 'ROC-AUC']
            
            # Tampilkan kesimpulan dalam wadah info premium
            st.markdown(f"""
            <div class="result-box result-low" style="background: linear-gradient(135deg, #11998e, #38ef7d); text-align: left; padding: 20px;">
                <h4 style="margin: 0px; color: white;">🏆 Performa Terbaik: Dataset {best_dataset}</h4>
                <p style="margin-top: 10px; font-size: 1.05rem; line-height: 1.6; color: white;">
                    Berdasarkan eksperimen klasifikasi menggunakan algoritma <b>AdaBoost Classifier</b> pada seluruh dataset yang diunggah, model menghasilkan kinerja paling optimal pada <b>Dataset {best_dataset}</b>.
                    <br><br>
                    Model berhasil memperoleh tingkat <b>Akurasi sebesar {best_accuracy*100:.2f}%</b> dan <b>ROC-AUC Score sebesar {best_auc*100:.2f}%</b> pada data uji. Hal ini menunjukkan bahwa algoritma Boosting (khususnya AdaBoost dengan base estimator Decision Stump / Tree berkedalaman 1) sangat efektif memisahkan kategori target pada karakteristik dataset tersebut.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            # Penjelasan akademis terstruktur untuk membantu menjawab dosen penguji
            st.markdown(f"""
            **Analisis Perbandingan untuk Sidang/Presentasi:**
            *   **Mengapa {best_dataset} Memiliki Akurasi Tertinggi?**
                Umumnya, AdaBoost bekerja sangat baik pada dataset yang fiturnya memiliki kontribusi kuat dan batas keputusan yang relatif tegas (tidak terlalu banyak tumpang sulit antar kelas).
            *   **Bagaimana dengan Dataset Lain?**
                Jika ada dataset dengan akurasi lebih rendah, hal tersebut biasanya dipicu oleh tingginya ketidakseimbangan kelas (*class imbalance*), fitur yang kurang berkorelasi dengan target, atau dimensi data (*noise*) yang tinggi yang mempersulit Decision Stump dalam membuat keputusan split tunggal.
            """)
            
    elif menu_kelompok == "🧪 Ruang Eksperimen ML":
        render_ml_experiment_room()
            
    elif menu_kelompok == "📚 Teori & Cara Kerja AdaBoost":
        render_adaboost_theory()
