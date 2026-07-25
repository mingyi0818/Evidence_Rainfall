"""
Data loader and preprocessing pipeline for Australian Weather dataset.
Handles missing values, feature engineering, encoding, scaling, and stratified splitting.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    RAW_CSV, PROCESSED_DIR, OUTPUT, PREPROCESS,
    RANDOM_SEEDS, TARGET_COL, DATE_COL, LOCATION_COL,
    NUMERIC_COLS, CATEGORICAL_COLS, NUM_CLASSES, CLASS_NAMES, POSITIVE_CLASS
)


def load_raw_data(csv_path=RAW_CSV):
    """Load raw CSV and return DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"[DataLoader] Loaded raw data: {df.shape}")
    return df


def extract_date_features(df, date_col=DATE_COL):
    """Extract temporal features from Date column."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['DayOfYear'] = df[date_col].dt.dayofyear
    # Season: 0=Summer(Dec-Feb), 1=Autumn(Mar-May), 2=Winter(Jun-Aug), 3=Spring(Sep-Nov)
    month = df['Month']
    df['Season'] = ((month % 12 + 3) // 3) % 4
    # Cyclical encoding
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    # Drop original date column
    df = df.drop(columns=[date_col])
    print("[DataLoader] Date features extracted: Year, Month, DayOfYear, Season, Month_sin/cos, DayOfYear_sin/cos")
    return df


def drop_high_missing_columns(df, threshold=PREPROCESS['missing_drop_threshold'],
                               drop_cols=PREPROCESS['drop_cols']):
    """Drop columns with excessive missing values."""
    missing_rates = df.isnull().mean()
    high_missing = missing_rates[missing_rates > threshold].index.tolist()
    # Ensure user-specified columns are dropped even if slightly below threshold
    to_drop = list(set(high_missing + drop_cols))
    to_drop = [c for c in to_drop if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"[DataLoader] Dropped columns ({len(to_drop)}): {to_drop}")
    return df


def handle_missing_values(df, numeric_strategy='median', categorical_strategy='mode'):
    """Impute missing values for numeric and categorical columns."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Exclude target from imputation
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)
    if TARGET_COL in categorical_cols:
        categorical_cols.remove(TARGET_COL)

    if numeric_cols:
        imputer_num = SimpleImputer(strategy=numeric_strategy)
        df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
    if categorical_cols:
        imputer_cat = SimpleImputer(strategy='constant', fill_value='Missing')
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
    print(f"[DataLoader] Missing values imputed: numeric={numeric_strategy}, categorical=constant('Missing')")
    return df


def preprocess_target(df, target_col=TARGET_COL, positive_class=POSITIVE_CLASS):
    """Encode binary target to 0/1 integers."""
    df = df.copy()
    df = df.dropna(subset=[target_col])
    df[target_col] = (df[target_col] == positive_class).astype(np.int64)
    counts = df[target_col].value_counts()
    print(f"[DataLoader] Target encoded. Distribution: {counts.to_dict()}")
    return df


def group_rare_categories(df, cat_cols, min_freq=PREPROCESS['min_category_frequency']):
    """Group rare categories into 'Other'."""
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        rare = counts[counts < min_freq].index
        if len(rare) > 0:
            df[col] = df[col].replace(rare, 'Other')
    return df


def build_preprocessor(df, numeric_cols, categorical_cols, scaler_type='standard',
                       encoder_type='onehot'):
    """Fit scaler and encoder on training data."""
    # Numeric scaling
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler: {scaler_type}")
    X_num = df[numeric_cols].values
    scaler.fit(X_num)

    # Categorical encoding
    if encoder_type == 'onehot':
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = df[categorical_cols].values
        encoder.fit(X_cat)
    else:
        raise ValueError(f"Unsupported encoder: {encoder_type}")

    return scaler, encoder


def apply_preprocessor(df, numeric_cols, categorical_cols, scaler, encoder):
    """Transform DataFrame using fitted scaler and encoder."""
    X_num = scaler.transform(df[numeric_cols].values)
    X_cat = encoder.transform(df[categorical_cols].values)
    X = np.hstack([X_num, X_cat])
    return X


def get_feature_names(numeric_cols, encoder):
    """Return list of feature names after preprocessing."""
    cat_names = encoder.get_feature_names_out().tolist()
    return numeric_cols + cat_names


class WeatherDataset(Dataset):
    """PyTorch Dataset for weather data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test,
                 batch_size=256, num_workers=0):
    """Create DataLoaders for train/val/test."""
    train_ds = WeatherDataset(X_train, y_train)
    val_ds = WeatherDataset(X_val, y_val)
    test_ds = WeatherDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def preprocess_and_split(seed=42, save=True, split_mode='temporal'):
    """
    Full preprocessing pipeline (leakage-free):
      1. Load raw data
      2. Drop target-missing rows
      3. Drop high-missing columns (statistics on full data — leakage is negligible for column-drop)
      4. Extract date features
      5. Encode target
      6. Split FIRST (temporal S1 / random S4) — before any imputation/scaling/encoding fit
      7. Fit imputer/scaler/encoder on TRAIN ONLY, then transform val/test
      8. Group rare categories on TRAIN ONLY
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder, feature_names
    split_mode: 'temporal' (S1, 2007-2014/2015/2016-2017) or 'random' (S4, stratified)
    """
    np.random.seed(seed)
    # Load
    df = load_raw_data()

    # Drop target-missing rows first
    df = df.dropna(subset=[TARGET_COL])
    print(f"[DataLoader] After dropping target-missing rows: {df.shape}")

    # Drop high-missing columns (column-drop decision uses full-data missing rates;
    # this is a feature-selection step, not a statistics-fit, so leakage is acceptable)
    df = drop_high_missing_columns(df)

    # Extract date features (deterministic transform, no fit)
    df = extract_date_features(df)

    # Encode target
    df = preprocess_target(df)

    # Identify remaining categorical and numeric columns
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns and c != TARGET_COL]
    num_cols = [c for c in df.columns if c not in cat_cols + [TARGET_COL]
                and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    # Separate features and target
    y = df[TARGET_COL].values
    df_features = df.drop(columns=[TARGET_COL])

    # Ensure column order
    num_cols = [c for c in num_cols if c in df_features.columns]
    cat_cols = [c for c in cat_cols if c in df_features.columns]

    # ===== SPLIT FIRST (leakage-free) =====
    if split_mode == 'temporal':
        # S1: temporal split — 2007-2014 train / 2015 val / 2016-2017 test
        # Business-valid: no future leakage, measures time-extrapolation skill
        if 'Year' in df_features.columns:
            year = df_features['Year'].values
        else:
            # reconstruct Year from original Date (already extracted)
            year = df['Year'].values if 'Year' in df.columns else pd.to_datetime(df_features.get('Date', pd.NaT), errors='coerce').dt.year.values

        train_mask = year <= 2014
        val_mask = (year == 2015)
        test_mask = year >= 2016
        # If Year not available (edge case), fall back to random
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print("[DataLoader] Warning: temporal split failed (no Year), falling back to random")
            split_mode = 'random'

    if split_mode == 'random':
        # S4: random stratified split (original behavior, kept as reference)
        df_temp, df_test, y_temp, y_test = train_test_split(
            df_features, y, test_size=PREPROCESS['test_size'],
            stratify=y if PREPROCESS['stratify'] else None,
            random_state=seed, shuffle=PREPROCESS['shuffle']
        )
        val_ratio = PREPROCESS['val_size'] / (1 - PREPROCESS['test_size'])
        df_train, df_val, y_train, y_val = train_test_split(
            df_temp, y_temp, test_size=val_ratio,
            stratify=y_temp if PREPROCESS['stratify'] else None,
            random_state=seed, shuffle=PREPROCESS['shuffle']
        )
    else:
        # temporal split (already computed masks)
        df_train = df_features[train_mask].copy()
        df_val = df_features[val_mask].copy()
        df_test = df_features[test_mask].copy()
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]

    print(f"[DataLoader] Split mode: {split_mode}")
    print(f"[DataLoader] Splits -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # ===== IMPUTE / ENCODE / SCALE ON TRAIN ONLY =====
    # Median imputation fit on train, transform val/test
    imputer_num = SimpleImputer(strategy='median')
    df_train[num_cols] = imputer_num.fit_transform(df_train[num_cols])
    df_val[num_cols] = imputer_num.transform(df_val[num_cols])
    df_test[num_cols] = imputer_num.transform(df_test[num_cols])

    imputer_cat = SimpleImputer(strategy='constant', fill_value='Missing')
    df_train[cat_cols] = imputer_cat.fit_transform(df_train[cat_cols])
    df_val[cat_cols] = imputer_cat.transform(df_val[cat_cols])
    df_test[cat_cols] = imputer_cat.transform(df_test[cat_cols])

    # Group rare categories using TRAIN statistics only
    for col in cat_cols:
        counts = df_train[col].value_counts()
        rare = counts[counts < PREPROCESS['min_category_frequency']].index
        if len(rare) > 0:
            df_train[col] = df_train[col].replace(rare, 'Other')
            df_val[col] = df_val[col].replace(rare, 'Other')
            df_test[col] = df_test[col].replace(rare, 'Other')

    # Fit preprocessor on training data only
    scaler, encoder = build_preprocessor(
        df_train, num_cols, cat_cols,
        scaler_type=PREPROCESS['numeric_scaler'],
        encoder_type=PREPROCESS['categorical_encoder']
    )

    # Transform
    X_train = apply_preprocessor(df_train, num_cols, cat_cols, scaler, encoder)
    X_val = apply_preprocessor(df_val, num_cols, cat_cols, scaler, encoder)
    X_test = apply_preprocessor(df_test, num_cols, cat_cols, scaler, encoder)

    feature_names = get_feature_names(num_cols, encoder)
    print(f"[DataLoader] Feature dimension after preprocessing: {len(feature_names)}")

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        pickle.dump(scaler, open(OUTPUT['scaler'], 'wb'))
        pickle.dump(encoder, open(OUTPUT['encoder'], 'wb'))
        np.savez(OUTPUT['split_indices'],
                 X_train=X_train, X_val=X_val, X_test=X_test,
                 y_train=y_train, y_val=y_val, y_test=y_test,
                 feature_names=feature_names,
                 num_cols=num_cols, cat_cols=cat_cols)
        print(f"[DataLoader] Preprocessed data saved to {PROCESSED_DIR}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder, feature_names


def load_preprocessed_data():
    """Load previously saved preprocessed data."""
    data = np.load(OUTPUT['split_indices'], allow_pickle=True)
    scaler = pickle.load(open(OUTPUT['scaler'], 'rb'))
    encoder = pickle.load(open(OUTPUT['encoder'], 'rb'))
    return (data['X_train'], data['X_val'], data['X_test'],
            data['y_train'], data['y_val'], data['y_test'],
            scaler, encoder, data['feature_names'].tolist())


if __name__ == "__main__":
    preprocess_and_split(seed=42, save=True)
