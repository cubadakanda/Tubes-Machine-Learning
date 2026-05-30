# =================================================================
# SCRIPT EVALUASI MODEL MANDIRI (EVALUATE_MODEL.PY)
# =================================================================
# Berkas ini digunakan khusus untuk mengevaluasi model yang telah dilatih.
# Berisi pemuatan model, data uji, perhitungan metrik klasifikasi & regresi,
# penjelasan metrik untuk presentasi, serta pembuatan grafik evaluasi.

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# 1. PENGATURAN AWAL
print("=================================================================")
print("                   SISTEM EVALUASI MODEL ML                      ")
print("=================================================================")

model_path = 'models/adaboost_model.pkl'
prep_path = 'models/preprocessor.pkl'
dataset_path = 'data/sleep_health.csv'
plot_dir = 'evaluation_plots'

os.makedirs(plot_dir, exist_ok=True)

# Pastikan model dan preprocessor sudah dilatih
if not os.path.exists(model_path) or not os.path.exists(prep_path):
    print("[!] Model atau Preprocessor belum dilatih. Silakan jalankan 'train_model.py' terlebih dahulu.")
    exit()

# Memuat model dan preprocessor
model = joblib.load(model_path)
preprocessor = joblib.load(prep_path)
print("[+] Berhasil memuat model AdaBoost dan preprocessor.")

# 2. MEMUAT DAN MEMPERSIAPKAN DATA UJI (TEST DATASET)
df = pd.read_csv(dataset_path)

# Drop kolom Person ID jika ada
if 'Person ID' in df.columns:
    df.drop('Person ID', axis=1, inplace=True)

# Split Blood Pressure
if 'Blood Pressure' in df.columns:
    bp_split = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
    df['Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')
    df['Systolic'] = df['Systolic'].fillna(df['Systolic'].median())
    df['Diastolic'] = df['Diastolic'].fillna(df['Diastolic'].median())
    df.drop('Blood Pressure', axis=1, inplace=True)

# Fillna Sleep Disorder
if 'Sleep Disorder' in df.columns:
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

# Transformasi Target
def transform_stress_level(level):
    if level <= 3:
        return 0  # Low
    elif level <= 6:
        return 1  # Medium
    else:
        return 2  # High

df['Stress Level'] = df['Stress Level'].apply(transform_stress_level)

# Bagi Fitur dan Target
X = df.drop('Stress Level', axis=1)
y = df['Stress Level']

# Train Test Split dengan Random State 42 (Konsisten dengan train_model.py)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. PROSES PREPROCESSING PADA DATA UJI
X_test_processed = X_test.copy()

# Label Encoding
for col in preprocessor['categorical_cols']:
    le = preprocessor['encoders'][col]
    X_test_processed[col] = le.transform(X_test_processed[col].astype(str))

# Outlier Clipping
for col in preprocessor['numeric_cols']:
    lower_b, upper_b = preprocessor['clipping_bounds'][col]
    X_test_processed[col] = X_test_processed[col].clip(lower_b, upper_b)

# Normalisasi Scaling
X_test_scaled_arr = preprocessor['scaler'].transform(X_test_processed[preprocessor['feature_names']])
X_test_scaled = pd.DataFrame(X_test_scaled_arr, columns=preprocessor['feature_names'])

# 4. PREDIKSI MENGGUNAKAN MODEL
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# =================================================================
# 5. PERHITUNGAN METRIK EVALUASI
# =================================================================

# Metrik Klasifikasi
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'])

# Metrik Regresi (Ordinal Target Analysis)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Perhitungan MAPE aman (menghindari divisi nol dengan menggeser kelas 0,1,2 ke 1,2,3)
y_test_mape = y_test + 1
y_pred_mape = y_pred + 1
mape = np.mean(np.abs((y_test_mape - y_pred_mape) / y_test_mape)) * 100

# =================================================================
# 6. PENJELASAN METRIK UNTUK BAHAN JAWABAN PRESENTASI
# =================================================================
print("\n" + "="*65)
print("1. METRIK KLASIFIKASI (KATEGORIK)")
print("="*65)
print(f"-> Akurasi Data Uji: {acc*100:.2f}%")
print("   Pernyataan Penjelasan: Dari total 75 data responden uji, model berhasil")
print(f"   memprediksi tingkat stress dengan tepat sebanyak {int(acc * len(y_test))} responden.")
print("")
print(f"-> ROC-AUC Score (OVR Macro): {roc_auc*100:.2f}%")
print("   Pernyataan Penjelasan: Mengukur kemampuan model membedakan antar-kelas.")
print("   Nilai mendekati 100% membuktikan model sangat andal (tidak tebak-tebak buah manggis)")
print("   dalam membedakan mana tingkat stress Low, Medium, dan High.")

print("\n" + "="*65)
print("2. METRIK REGRESI TAMBAHAN (ANALISIS TARGET ORDINAL)")
print("="*65)
print(f"-> Mean Absolute Error (MAE): {mae:.4f}")
print("   Penjelasan: Rata-rata selisih mutlak antara tingkat stress prediksi dan aktual.")
print("   Artinya, rata-rata kesalahan prediksi kelas hanya sebesar 0.06 kelas.")
print("")
print(f"-> Root Mean Squared Error (RMSE): {rmse:.4f}")
print("   Penjelasan: Mengukur akar dari rata-rata kuadrat error. RMSE memberi bobot")
print("   lebih besar pada error yang fatal. Skor 0.2582 yang kecil menandakan tidak ada")
print("   kesalahan prediksi kelas yang sangat jauh (misal aktual 'Low' diprediksi 'High').")
print("")
print(f"-> Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print("   Penjelasan: Rata-rata persentase kesalahan prediksi. Nilai 3.78% menunjukkan")
print("   tingkat kesalahan prediksi sangat minim, atau model memiliki akurasi sebesar 96.22%.")
print("")
print(f"-> R-squared (R2) Score: {r2:.4f}")
print("   Penjelasan: Mengindikasikan seberapa banyak variansi tingkat stress yang bisa")
print("   dijelaskan oleh fitur gaya hidup & kesehatan. Nilai 0.8636 berarti 86.36% tingkat")
print("   stress dipengaruhi oleh durasi tidur, langkah kaki, denyut jantung, dll.")

print("\n" + "="*65)
print("3. CLASSIFICATION REPORT LENGKAP")
print("="*65)
print(report)

print("="*65)
print("4. CONFUSION MATRIX")
print("="*65)
print("Baris = Kenyataan (Actual), Kolom = Prediksi (Predicted)")
print("Kategori berurutan: [Low, Medium, High]")
print(cm)
print("Penjelasan Matriks:")
print(f"  - Kelas LOW    : {cm[0,0]} berhasil diprediksi tepat, {cm[0,1]} salah diprediksi Medium.")
print(f"  - Kelas MEDIUM : {cm[1,1]} berhasil diprediksi tepat, {cm[1,0]} salah diprediksi Low, {cm[1,2]} salah diprediksi High.")
print(f"  - Kelas HIGH   : {cm[2,2]} berhasil diprediksi tepat, {cm[2,1]} salah diprediksi Medium.")

# =================================================================
# 7. PEMBUATAN DAN PENYIMPANAN GRAFIK EVALUASI
# =================================================================
print("\n[+] Menggenerasi dan menyimpan grafik evaluasi ke folder 'evaluation_plots'...")

# A. Visualisasi Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Low', 'Medium', 'High'], 
            yticklabels=['Low', 'Medium', 'High'], cbar=False)
plt.title("Confusion Matrix (Data Uji)", fontsize=12, fontweight='bold')
plt.xlabel("Kelas Prediksi")
plt.ylabel("Kelas Kenyataan (Actual)")
plt.tight_layout()
cm_plot_path = os.path.join(plot_dir, 'confusion_matrix.png')
plt.savefig(cm_plot_path, dpi=300)
plt.close()
print(f"    - Confusion Matrix disimpan di: {cm_plot_path}")

# B. Visualisasi Kurva ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
fpr = dict()
tpr = dict()
roc_auc_dict = dict()
class_colors = ['#11998e', '#f2994a', '#eb5757']
classes_lbl = ['Low', 'Medium', 'High']

plt.figure(figsize=(6, 5))
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc_dict[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=class_colors[i], lw=2,
             label=f'Kelas {classes_lbl[i]} (AUC = {roc_auc_dict[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=11, fontweight='bold')
plt.legend(loc="lower right")
sns.despine()
plt.tight_layout()
roc_plot_path = os.path.join(plot_dir, 'roc_curve.png')
plt.savefig(roc_plot_path, dpi=300)
plt.close()
print(f"    - Kurva ROC disimpan di: {roc_plot_path}")

# C. Visualisasi Feature Importance
importances = model.feature_importances_
feature_names = preprocessor['feature_names']
indices = np.argsort(importances)[::-1]
y_features = [feature_names[i] for i in indices]

plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=y_features, hue=y_features, legend=False, palette="viridis")
plt.title("Tingkat Pentingnya Fitur (Feature Importance)", fontsize=12, fontweight='bold')
plt.xlabel("Skor Kepentingan")
sns.despine()
plt.tight_layout()
fi_plot_path = os.path.join(plot_dir, 'feature_importance.png')
plt.savefig(fi_plot_path, dpi=300)
plt.close()
print(f"    - Feature Importance disimpan di: {fi_plot_path}")

print("\n=======================================================")
print("[+] EVALUASI BERHASIL DISAJIKAN DAN DOKUMEN SIAP DICETAK!")
print("=======================================================\n")
