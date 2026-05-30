# Sleep Health Stress Prediction System 🌙

Sistem Machine Learning dan Dashboard Interaktif berbasis web untuk memprediksi tingkat stress seseorang berdasarkan data kesehatan dan gaya hidup. Proyek ini dibuat menggunakan Python, Streamlit, Scikit-Learn, dan AdaBoost Classifier.

Sistem ini sangat cocok digunakan sebagai bahan presentasi Tugas Besar Machine Learning karena memiliki visualisasi data yang interaktif (EDA), halaman evaluasi performa model yang lengkap dengan Confusion Matrix & Feature Importance, serta fitur prediksi massal (Batch Prediction) menggunakan file CSV.

---

## 📂 Struktur Folder Proyek

Proyek ini terstruktur secara rapi dan modular sebagai berikut:

```text
"Project ML"
├── data/
│   ├── sleep_health.csv               # Dataset asli dari GitHub (auto-download jika kosong)
│   └── sleep_health_preprocessed.csv  # Dataset hasil preprocessing untuk visualisasi di web
├── models/
│   ├── adaboost_model.pkl             # Model AdaBoost terbaik hasil GridSearchCV
│   └── preprocessor.pkl               # Objek preprocessing (Scaler, Encoders, IQR bounds)
├── requirements.txt                   # Daftar dependensi Python yang dibutuhkan
├── train_model.py                     # Skrip untuk preprocessing, training, tuning, dan evaluasi
├── app.py                             # Aplikasi utama Dashboard & Prediksi berbasis Streamlit
└── README.md                          # Panduan penggunaan proyek (file ini)
```

---

## 🛠️ Persyaratan Sistem & Instalasi

Ikuti langkah-langkah di bawah ini untuk mempersiapkan lingkungan dan memasang seluruh dependensi.

### 1. Prasyarat
Pastikan Anda telah memasang:
*   **Python** versi 3.8 ke atas (disarankan Python 3.9, 3.10, atau 3.11).
*   **VSCode** atau IDE Python lainnya.

### 2. Instalasi Dependensi
Buka terminal (Command Prompt/PowerShell) di VSCode pada folder proyek ini, lalu jalankan perintah berikut untuk menginstal semua pustaka yang diperlukan:

```bash
pip install -r requirements.txt
```

*File `requirements.txt` berisi:*
*   `pandas` (Manipulasi data & pembacaan CSV)
*   `numpy` (Komputasi numerik)
*   `scikit-learn` (Pembuatan model Machine Learning & Preprocessing)
*   `streamlit` (Pembuatan antarmuka web interaktif)
*   `matplotlib` & `seaborn` (Pembuatan grafik visualisasi data)
*   `joblib` (Penyimpanan dan pemuatan model `.pkl`)

---

## 🚀 Cara Menjalankan Proyek

Proyek ini dapat dijalankan melalui dua tahap: pelatihan model (opsional jika ingin melatih ulang) dan peluncuran web dashboard.

### Langkah 1: Pelatihan Model (`train_model.py`)
Skrip ini akan mengunduh dataset secara otomatis (jika berkas `sleep_health.csv` belum ada di folder `data/`), melakukan preprocessing data, mencari hyperparameter terbaik dengan `GridSearchCV` (5-fold CV), menampilkan evaluasi model, dan menyimpan berkas model ke folder `models/`.

Jalankan perintah berikut di terminal:
```bash
python train_model.py
```

*Output yang akan tampil di terminal berupa:*
*   Status pengunduhan dataset.
*   Log setiap langkah preprocessing (drop kolom, pemecahan Blood Pressure, imputasi missing values, target mapping, IQR outlier clipping, dan MinMax Scaling).
*   Hasil GridSearch parameter terbaik dan akurasi cross-validation terbaik.
*   Metrik evaluasi pada data uji (Akurasi, F1-Score, Precision, Recall, dan Confusion Matrix).
*   Konfirmasi penyimpanan file model (`adaboost_model.pkl`) & preprocessor (`preprocessor.pkl`).

### Langkah 2: Menjalankan Dashboard Streamlit (`app.py`)
Untuk membuka dashboard visualisasi data dan sistem prediksi tingkat stress interaktif pada peramban (browser) web, jalankan perintah berikut:

```bash
streamlit run app.py
```

Dashboard akan otomatis terbuka di browser Anda pada alamat default: `http://localhost:8501`.

---

## 📝 Penjelasan Detail File Proyek

### 1. `train_model.py` (Script Training)
Berisi seluruh alur pengolahan data dan modeling:
*   **Preprocessing**:
    *   **Drop**: Menghapus `Person ID` karena bersifat pengenal unik tak berkorelasi.
    *   **Blood Pressure Splitting**: Memisahkan string `Blood Pressure` (misal `120/80`) menjadi dua kolom numerik: `Systolic` dan `Diastolic`. Kolom asli dihapus.
    *   **Imputasi Missing Value**: Mengisi nilai kosong di kolom `Sleep Disorder` dengan kategori `'None'`.
    *   **Target Transformation**: Mengubah target `Stress Level` menjadi skala ordinal:
        *   `Low` (0) jika nilainya $\le 3$
        *   `Medium` (1) jika nilainya $4 - 6$
        *   `High` (2) jika nilainya $\ge 7$
    *   **Label Encoding**: Mengodekan kolom kategorik (`Gender`, `Occupation`, `BMI Category`, `Sleep Disorder`) ke bentuk angka biner/kategori numerik.
    *   **Outlier Handling (IQR & Clipping)**: Menghitung rentang interkuartil (IQR) pada data latihan dan melakukan *clipping* data ekstrem ke batas atas/bawah tanpa menghapus sampel data.
    *   **Normalisasi**: Menggunakan `MinMaxScaler` untuk mengubah seluruh fitur ke dalam rentang $[0, 1]$.
*   **Modeling**:
    *   Menggunakan `AdaBoostClassifier` dengan estimator dasar `DecisionTreeClassifier` kedalaman maksimal 1 (*Decision Stump*).
    *   `GridSearchCV` mencari kombinasi optimal untuk `n_estimators` `[50, 100, 200]` dan `learning_rate` `[0.01, 0.1, 1.0]`.
    *   Model dilatih menggunakan pemisahan data 80:20 berstratifikasi (`stratify=y`) untuk menjaga proporsi kelas stress.
*   **Penyimpanan**: Menyimpan model terbaik dan seluruh objek preprocessing (scaler, encoder, batas IQR) agar proses transformasi data uji/prediksi baru di Streamlit persis sama.

### 2. `app.py` (Streamlit Dashboard & Real-time Prediction)
Aplikasi antarmuka web modern dengan 4 menu navigasi di sidebar:
*   **📊 Dashboard Dataset (EDA)**: Menampilkan statistik dasar dataset, diagram batang dan pie chart distribusi target stress level, histogram interaktif durasi tidur & langkah harian, serta heatmap korelasi antar fitur.
*   **🔮 Prediksi Tingkat Stress**: Menyediakan formulir input interaktif lengkap dengan slider dan dropdown menu. Setelah tombol "Prediksi" ditekan, aplikasi akan memproses data masukan secara realtime dan mengklasifikasikan ke dalam 3 tingkat stress (Low, Medium, High) disertai visualisasi probabilitas keyakinan model.
*   **📈 Evaluasi & Kinerja Model**: Menampilkan informasi parameter terbaik hasil tuning model, visualisasi confusion matrix interaktif, diagram tingkat kepentingan fitur (*Feature Importance*), dan classification report detail.
*   **📁 Prediksi Massal (CSV)**: Memungkinkan pengguna mengunggah file CSV berisi data kesehatan banyak orang sekaligus, memproses prediksi massal di latar belakang, menampilkan tabel hasil prediksi, dan menyertakan tombol untuk mengunduh (download) hasil prediksi tersebut kembali sebagai file CSV baru.

---

## 🌟 Keunggulan Teknis Proyek Ini
1.  **Anti Data Leakage**: Pengitungan clipping pencilan (IQR) dan pencocokan normalisasi (Scaler) dipelajari (`fit`) hanya dari data latih (train set) dan diterapkan (`transform`) ke data uji maupun prediksi web, sesuai standar industri.
2.  **Lintas Versi Scikit-Learn**: Kode diinisialisasi secara modular untuk menangani perubahan nama parameter scikit-learn (misalnya penggantian parameter `base_estimator` menjadi `estimator` pada versi terbaru dan penghapusan parameter `algorithm` pada sklearn v1.6+).
3.  **UI Premium**: Menggunakan custom CSS dengan tipografi font Google 'Outfit', efek glassmorphism, bayangan melayang (*hover shadow*), dan visualisasi warna yang harmonis.
