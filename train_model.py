# =================================================================
# SECTION 1: IMPORT LIBRARIES
# =================================================================
import os
import urllib.request
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =================================================================
# SECTION 2: PREPROCESSING AND DATA LOADING
# =================================================================

# Membuat direktori data dan models jika belum ada
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

dataset_url = "https://raw.githubusercontent.com/Ann805/Sleep/main/Sleep_health_and_lifestyle_dataset.csv"
dataset_path = os.path.join('data', 'sleep_health.csv')

# Unduh dataset jika belum ada secara lokal
if not os.path.exists(dataset_path):
    print("[-] Dataset tidak ditemukan secara lokal. Mengunduh dataset dari GitHub...")
    try:
        urllib.request.urlretrieve(dataset_url, dataset_path)
        print("[+] Dataset berhasil diunduh dan disimpan di:", dataset_path)
    except Exception as e:
        print(f"[!] Gagal mengunduh dataset: {e}")
        print("[!] Silakan pastikan koneksi internet Anda aktif atau letakkan file sleep_health.csv di folder 'data'.")
        raise e
else:
    print("[+] Dataset ditemukan secara lokal.")

# Membaca dataset
df = pd.read_csv(dataset_path)
print(f"[+] Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")

# 1. Drop kolom yang tidak berguna (Person ID)
if 'Person ID' in df.columns:
    df.drop('Person ID', axis=1, inplace=True)
    print("[+] Kolom 'Person ID' telah dihapus.")

# 2. Memecah Blood Pressure menjadi Systolic dan Diastolic
if 'Blood Pressure' in df.columns:
    # Memisahkan string BP berdasarkan '/'
    bp_split = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
    df['Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')
    
    # Isi missing value pada BP jika ada (berdasarkan median)
    df['Systolic'] = df['Systolic'].fillna(df['Systolic'].median())
    df['Diastolic'] = df['Diastolic'].fillna(df['Diastolic'].median())
    
    # Hapus kolom Blood Pressure yang asli
    df.drop('Blood Pressure', axis=1, inplace=True)
    print("[+] Kolom 'Blood Pressure' berhasil dipecah menjadi 'Systolic' dan 'Diastolic'.")

# 3. Mengatasi Missing Value pada Sleep Disorder
# Mengisi nilai kosong (NaN) dengan kategori "None"
if 'Sleep Disorder' in df.columns:
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    print("[+] Missing values pada 'Sleep Disorder' diisi dengan 'None'.")

# 4. Transformasi Target (Stress Level)
# Low (<=3), Medium (4-6), High (>=7)
def transform_stress_level(level):
    if level <= 3:
        return 0  # Low
    elif level <= 6:
        return 1  # Medium
    else:
        return 2  # High

if 'Stress Level' in df.columns:
    df['Stress Level'] = df['Stress Level'].apply(transform_stress_level)
    print("[+] Target 'Stress Level' berhasil ditransformasi menjadi kelas ordinal (0: Low, 1: Medium, 2: High).")

# Menentukan kolom fitur dan target
# Berdasarkan analisis korelasi linear dan multikolinearitas (informasi redundan):
# 1. 'Occupation' dihapus karena korelasi target rendah (0.15) dan korelasi tinggi dengan 'BMI Category' (0.70).
# 2. 'Diastolic' dihapus karena korelasi 0.97 dengan 'Systolic' (multikolinearitas ekstrem).
# 3. 'Daily Steps' dihapus karena korelasi 0.77 dengan 'Physical Activity Level'.
features_to_drop = ['Occupation', 'Diastolic', 'Daily Steps']
X = df.drop(['Stress Level'] + [col for col in features_to_drop if col in df.columns], axis=1, errors='ignore')
y = df['Stress Level']

# Simpan daftar kolom asli setelah BP dipecah dan fitur redundan dibuang
feature_names = X.columns.tolist()

# 5. Label Encoding untuk Fitur Kategorik
categorical_cols = ['Gender', 'BMI Category', 'Sleep Disorder']
encoders = {}

print("[+] Memulai Label Encoding untuk fitur kategorik...")
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        # Menggunakan fit pada seluruh X untuk memastikan semua kategori terdaftar
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"    - Fitur '{col}': {list(le.classes_)} -> {le.transform(le.classes_)}")

# 6. Train Test Split (80:20, Stratify Target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[+] Pembagian data selesai: Train set = {X_train.shape[0]} data, Test set = {X_test.shape[0]} data.")

# 7. Outlier Handling dengan IQR & Clipping (hanya pada data numerik)
numeric_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                'Heart Rate', 'Systolic']
clipping_bounds = {}

print("[+] Melakukan Outlier Handling menggunakan IQR & Clipping...")
# Salin data untuk menghindari settingwithcopy warning
X_train = X_train.copy()
X_test = X_test.copy()

for col in numeric_cols:
    if col in X_train.columns:
        # Hitung Q1, Q3, dan IQR pada Train Set untuk menghindari data leakage
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        clipping_bounds[col] = (lower_bound, upper_bound)
        
        # Lakukan clipping pada Train & Test Set
        X_train[col] = X_train[col].clip(lower_bound, upper_bound)
        X_test[col] = X_test[col].clip(lower_bound, upper_bound)

# 8. Normalisasi menggunakan MinMaxScaler
print("[+] Melakukan normalisasi dengan MinMaxScaler...")
scaler = MinMaxScaler()
# Fit scaler pada Train set dan transform Train & Test set
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Konversi kembali ke DataFrame untuk kemudahan visualisasi/feature importance
X_train_final = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_final = pd.DataFrame(X_test_scaled, columns=feature_names)

# Menyimpan seluruh komponen preprocessing untuk digunakan saat deployment/Streamlit
preprocessor_data = {
    'encoders': encoders,
    'clipping_bounds': clipping_bounds,
    'scaler': scaler,
    'feature_names': feature_names,
    'categorical_cols': categorical_cols,
    'numeric_cols': numeric_cols
}
joblib.dump(preprocessor_data, 'models/preprocessor.pkl')
print("[+] Objek preprocessing disimpan ke 'models/preprocessor.pkl'.")


# =================================================================
# SECTION 3: TRAINING AND HYPERPARAMETER TUNING
# =================================================================
print("\n[+] Memulai Hyperparameter Tuning menggunakan GridSearchCV...")

# Inisialisasi base estimator (Decision Stump)
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

# Mengatasi ketidakcocokan parameter scikit-learn versi lama dan baru
# (Di sklearn v1.2+, 'base_estimator' diganti menjadi 'estimator')
# (Di sklearn v1.6+, parameter 'algorithm' dihapus karena hanya mendukung SAMME secara default)
try:
    adaboost_base = AdaBoostClassifier(estimator=base_estimator, algorithm='SAMME', random_state=42)
except TypeError:
    try:
        # Untuk sklearn v1.6+
        adaboost_base = AdaBoostClassifier(estimator=base_estimator, random_state=42)
    except TypeError:
        try:
            # Untuk sklearn < v1.2
            adaboost_base = AdaBoostClassifier(base_estimator=base_estimator, algorithm='SAMME', random_state=42)
        except TypeError:
            adaboost_base = AdaBoostClassifier(base_estimator=base_estimator, random_state=42)

# Tentukan parameter grid secara dinamis berdasarkan versi scikit-learn
max_depth_key = 'estimator__max_depth' if hasattr(adaboost_base, 'estimator') and adaboost_base.estimator is not None else 'base_estimator__max_depth'

# Parameter Grid
param_grid = {
    max_depth_key: [1, 2, 3],
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
}

# GridSearchCV dengan 5-Fold Cross Validation
grid_search = GridSearchCV(
    estimator=adaboost_base,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Melatih Model
grid_search.fit(X_train_final, y_train)

# Mendapatkan model terbaik
best_model = grid_search.best_estimator_

print("\n=======================================================")
print("HASIL HYPERPARAMETER TUNING")
print("=======================================================")
print(f"Hyperparameter Terbaik : {grid_search.best_params_}")
print(f"Accuracy Terbaik (CV)  : {grid_search.best_score_:.4f}")
print("=======================================================\n")


# =================================================================
# SECTION 4: EVALUATION
# =================================================================
print("[+] Mengevaluasi model pada data Test Set...")
y_pred = best_model.predict(X_test_final)

# Metrik Evaluasi Klasifikasi
test_accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'])
conf_matrix = confusion_matrix(y_test, y_pred)

# Metrik Evaluasi Regresi Tambahan (Ordinal Target Analysis)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Menghitung MAPE secara aman (menggeser kelas 0,1,2 ke 1,2,3 untuk menghindari pembagian dengan nol)
y_test_mape = y_test + 1
y_pred_mape = y_pred + 1
mape = np.mean(np.abs((y_test_mape - y_pred_mape) / y_test_mape)) * 100

# Menghitung ROC-AUC Score (One-vs-Rest Macro)
y_prob = best_model.predict_proba(X_test_final)
roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

print("=======================================================")
print("METRIK EVALUASI MODEL (DATA UJI)")
print("=======================================================")
print(f"Akurasi Data Uji: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"ROC-AUC Score (One-vs-Rest Macro): {roc_auc_ovr:.4f} ({roc_auc_ovr * 100:.2f}%)")
print("\nClassification Report:")
print(class_report)
print("Confusion Matrix:")
print(conf_matrix)
print("-------------------------------------------------------")
print("METRIK EVALUASI REGRESI TAMBAHAN (ANALISIS TARGET ORDINAL)")
print("-------------------------------------------------------")
print(f"Mean Absolute Error (MAE)            : {mae:.4f}")
print(f"Root Mean Squared Error (RMSE)       : {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R-squared (R2) Score                 : {r2:.4f}")
print("=======================================================\n")


# =================================================================
# SECTION 5: DEPLOYMENT (SAVE MODEL)
# =================================================================
# Menyimpan model terbaik
model_save_path = 'models/adaboost_model.pkl'
joblib.dump(best_model, model_save_path)
print(f"[+] Model terbaik berhasil disimpan ke '{model_save_path}'.")

# Menyimpan dataset olahan yang sudah di-preprocess untuk kepentingan visualisasi di app.py
df_preprocessed = pd.concat([X.copy(), y.copy()], axis=1)
# Tambahkan kembali kolom string/label asli untuk visualisasi visual yang lebih mudah dibaca
for col in categorical_cols:
    df_preprocessed[col + '_Label'] = encoders[col].inverse_transform(df_preprocessed[col])
df_preprocessed.to_csv('data/sleep_health_preprocessed.csv', index=False)
print("[+] Dataset hasil preprocessing disimpan ke 'data/sleep_health_preprocessed.csv' untuk visualisasi Streamlit.")

print("\n[+] PROSES TRAINING SELESAI DENGAN SUKSES!")
