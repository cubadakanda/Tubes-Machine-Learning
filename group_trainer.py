# =================================================================
# UTILITY TRAINER UNTUK DATASET KELOMPOK (GROUP_TRAINER.PY)
# =================================================================
# Berkas ini berisi logika latih otomatis (AutoML) untuk dataset teman kelompok.
# Menangani preprocessing dinamis, pengkodean kategori, normalisasi,
# serta pelatihan model AdaBoost Classifier dan kalkulasi metrik evaluasi lengkap.

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
)

def train_custom_dataset(filepath, target_col):
    """
    Melatih model AdaBoost secara dinamis untuk dataset kustom kelompok.
    Mengembalikan dictionary hasil evaluasi dan menyimpan model + preprocessor.
    """
    # 1. Load Dataset
    df = pd.read_csv(filepath)
    df_original = df.copy()
    
    # 2. Pembersihan Kolom ID/Index otomatis
    cols_to_drop = [col for col in df.columns if 'id' in col.lower() or 'index' in col.lower() or 'person' in col.lower()]
    # Pastikan tidak menghapus target_col
    if target_col in cols_to_drop:
        cols_to_drop.remove(target_col)
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        
    # 3. Pisahkan Fitur (X) dan Target (y)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' tidak ditemukan di dataset.")
        
    X = df.drop(columns=[target_col])
    
    # Hapus kolom yang seluruh nilainya kosong (all NaN)
    X.dropna(axis=1, how='all', inplace=True)
    
    y = df[target_col]
    
    # Encode target jika kategorik
    target_encoder = None
    if y.dtype == 'object' or len(y.unique()) < 15:
        # Periksa jumlah kelas unik
        unique_vals = y.unique()
        # Jika bertipe string/object, lakukan label encoding
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            
    # 4. Imputasi & Deteksi Kolom Kategorik/Numerik
    categorical_cols = []
    numeric_cols = []
    
    encoders = {}
    clipping_bounds = {}
    
    for col in X.columns:
        # Jika bertipe object/string
        if X[col].dtype == 'object':
            categorical_cols.append(col)
            # Imputasi kategori kosong dengan mode
            mode_val = X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown'
            X[col] = X[col].fillna(mode_val)
            
            # Label Encoder
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            numeric_cols.append(col)
            # Imputasi numerik dengan median
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X[col] = X[col].fillna(median_val)
            
    # 5. Train-Test Split (80:20)
    # Gunakan stratifikasi jika klasifikasi memiliki sampel yang cukup
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except Exception:
        # Jika jumlah kelas target terlalu sedikit untuk stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
    # 6. IQR Outlier Clipping pada Train & Test
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in numeric_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        clipping_bounds[col] = (lower_bound, upper_bound)
        
        X_train[col] = X_train[col].clip(lower_bound, upper_bound)
        X_test[col] = X_test[col].clip(lower_bound, upper_bound)
        
    # 7. Normalisasi Scaling (MinMaxScaler)
    scaler = MinMaxScaler()
    feature_names = X.columns.tolist()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_final = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_final = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # 8. Pelatihan Model AdaBoost
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
    
    # Mengatasi versi scikit-learn
    try:
        adaboost = AdaBoostClassifier(estimator=base_estimator, algorithm='SAMME', random_state=42)
    except TypeError:
        try:
            adaboost = AdaBoostClassifier(estimator=base_estimator, random_state=42)
        except TypeError:
            try:
                adaboost = AdaBoostClassifier(base_estimator=base_estimator, algorithm='SAMME', random_state=42)
            except TypeError:
                adaboost = AdaBoostClassifier(base_estimator=base_estimator, random_state=42)
                
    # Hyperparameter tuning cepat (GridSearchCV kecil)
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 1.0]
    }
    
    grid = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_final, y_train)
    
    best_model = grid.best_estimator_
    
    # 9. Evaluasi
    y_pred = best_model.predict(X_test_final)
    y_prob = best_model.predict_proba(X_test_final)
    
    acc = accuracy_score(y_test, y_pred)
    
    # ROC-AUC (Mendukung Binary & Multi-class)
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        # binary class
        roc_auc_val = roc_auc_score(y_test, y_prob[:, 1])
    else:
        # multi class
        roc_auc_val = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        
    cm = confusion_matrix(y_test, y_pred)
    
    # Regresi Metrik Tambahan
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # MAPE Aman
    y_test_mape = y_test + 1
    y_pred_mape = y_pred + 1
    mape = np.mean(np.abs((y_test_mape - y_pred_mape) / y_test_mape)) * 100
    
    # 10. Menyimpan Model dan Metadata Preprocessing
    dataset_name = os.path.splitext(os.path.basename(filepath))[0]
    
    # Model Save
    model_save_path = f'models/group_{dataset_name}_model.pkl'
    joblib.dump(best_model, model_save_path)
    
    # Prep Save
    preprocessor_data = {
        'encoders': encoders,
        'clipping_bounds': clipping_bounds,
        'scaler': scaler,
        'feature_names': feature_names,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'target_col': target_col,
        'target_encoder': target_encoder,
        'classes': [str(c) for c in np.unique(y_test)]
    }
    prep_save_path = f'models/group_{dataset_name}_prep.pkl'
    joblib.dump(preprocessor_data, prep_save_path)
    
    # Simpan dataset olahan untuk visualisasi EDA cepat di streamlit
    df_preprocessed = pd.concat([X.copy(), pd.Series(y, name=target_col)], axis=1)
    df_preprocessed.to_csv(f'data/group_{dataset_name}_preprocessed.csv', index=False)
    
    # Hasil evaluasi
    eval_results = {
        'accuracy': acc,
        'roc_auc': roc_auc_val,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'confusion_matrix': cm.tolist(),
        'best_params': grid.best_params_
    }
    
    return eval_results


def run_custom_experiment(filepath, target_col, scaler_type='minmax', clip_outliers=True, base_depth=1, n_estimators=50, learning_rate=1.0, features_to_drop=None):
    """
    Melatih model AdaBoost secara dinamis dengan parameter preprocessing dan model kustom.
    Mengembalikan dict hasil evaluasi.
    """
    # 1. Load Dataset
    df = pd.read_csv(filepath)
    
    # 2. Pembersihan Kolom ID/Index otomatis
    cols_to_drop = [col for col in df.columns if 'id' in col.lower() or 'index' in col.lower() or 'person' in col.lower()]
    if target_col in cols_to_drop:
        cols_to_drop.remove(target_col)
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        
    # 3. Pisahkan Fitur (X) dan Target (y)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' tidak ditemukan di dataset.")
        
    X = df.drop(columns=[target_col])
    
    # Hapus kolom yang seluruh nilainya kosong (all NaN)
    X.dropna(axis=1, how='all', inplace=True)
    
    # Hapus fitur kustom yang ingin dibuang (Feature Selection)
    if features_to_drop:
        X.drop(columns=[col for col in features_to_drop if col in X.columns], inplace=True, errors='ignore')
        
    y = df[target_col]
    
    # Encode target jika kategorik
    target_encoder = None
    if y.dtype == 'object' or len(y.unique()) < 15:
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            
    # 4. Imputasi & Deteksi Kolom Kategorik/Numerik
    categorical_cols = []
    numeric_cols = []
    
    encoders = {}
    clipping_bounds = {}
    
    for col in X.columns:
        if X[col].dtype == 'object':
            categorical_cols.append(col)
            # Imputasi kategori dengan mode
            mode_val = X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown'
            X[col] = X[col].fillna(mode_val)
            # Label Encoder
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            numeric_cols.append(col)
            # Imputasi numerik dengan median
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X[col] = X[col].fillna(median_val)
            
    # 5. Train-Test Split (80:20)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
    # 6. Outlier Handling (Clipping) jika diaktifkan
    X_train = X_train.copy()
    X_test = X_test.copy()
    if clip_outliers:
        for col in numeric_cols:
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            clipping_bounds[col] = (lower_bound, upper_bound)
            
            X_train[col] = X_train[col].clip(lower_bound, upper_bound)
            X_test[col] = X_test[col].clip(lower_bound, upper_bound)
            
    # 7. Normalisasi Scaling
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    elif scaler_type == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        # No scaling
        scaler = None
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
        
    # 8. Pelatihan Model AdaBoost dengan parameter kustom
    base_estimator = DecisionTreeClassifier(max_depth=base_depth, random_state=42)
    
    try:
        adaboost = AdaBoostClassifier(
            estimator=base_estimator, 
            n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            algorithm='SAMME', 
            random_state=42
        )
    except TypeError:
        try:
            adaboost = AdaBoostClassifier(
                estimator=base_estimator, 
                n_estimators=n_estimators, 
                learning_rate=learning_rate, 
                random_state=42
            )
        except TypeError:
            try:
                adaboost = AdaBoostClassifier(
                    base_estimator=base_estimator, 
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate, 
                    algorithm='SAMME', 
                    random_state=42
                )
            except TypeError:
                adaboost = AdaBoostClassifier(
                    base_estimator=base_estimator, 
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate, 
                    random_state=42
                )
                
    adaboost.fit(X_train_scaled, y_train)
    
    # 9. Evaluasi
    y_pred = adaboost.predict(X_test_scaled)
    y_prob = adaboost.predict_proba(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        roc_auc_val = roc_auc_score(y_test, y_prob[:, 1])
    else:
        roc_auc_val = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        
    cm = confusion_matrix(y_test, y_pred)
    
    # Regresi Metrik Tambahan
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # MAPE Aman
    y_test_mape = y_test + 1
    y_pred_mape = y_pred + 1
    mape = np.mean(np.abs((y_test_mape - y_pred_mape) / y_test_mape)) * 100
    
    eval_results = {
        'accuracy': acc,
        'roc_auc': roc_auc_val,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'confusion_matrix': cm.tolist()
    }
    
    return eval_results
