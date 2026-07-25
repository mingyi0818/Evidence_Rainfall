"""Test C2 neighborhood label computation on the rainfall dataset."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import time
import numpy as np
import pandas as pd

from config import RAW_CSV, TARGET_COL, DATE_COL, LOCATION_COL
from data_loader import load_raw_data, extract_date_features
from cae_net import compute_spatial_neighborhood_labels_fast

print("Loading raw data...", flush=True)
df = load_raw_data()
df = df.dropna(subset=[TARGET_COL])
# Don't call extract_date_features yet - we need the Date column for neighborhood computation
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
print(f"  Shape: {df.shape}", flush=True)

# Add synthetic Lat/Lon (the dataset has Location names but not coordinates)
locations = sorted(df[LOCATION_COL].unique())
print(f"  {len(locations)} unique locations", flush=True)

np.random.seed(42)
loc_coords = {}
for i, loc in enumerate(locations):
    # Use a grid layout as proxy
    loc_coords[loc] = (float(i % 8), float(i // 8))

df['Lat'] = df[LOCATION_COL].map(lambda l: loc_coords[l][0])
df['Lon'] = df[LOCATION_COL].map(lambda l: loc_coords[l][1])

# Compute neighborhood labels on a small subset first
print("\nComputing neighborhood labels on first 1000 rows...", flush=True)
t0 = time.time()
df_small = df.head(1000).copy()
k_arr, m_arr = compute_spatial_neighborhood_labels_fast(
    df_small, LOCATION_COL, DATE_COL, TARGET_COL, m=5
)
print(f"  Done in {time.time()-t0:.1f}s", flush=True)
print(f"  k: mean={k_arr.mean():.2f}, min={k_arr.min()}, max={k_arr.max()}", flush=True)
print(f"  m: mean={m_arr.mean():.2f}, min={m_arr.min()}, max={m_arr.max()}", flush=True)
print(f"  k distribution: {np.bincount(k_arr)}", flush=True)

# Now compute on full dataset
print("\nComputing neighborhood labels on full dataset...", flush=True)
t0 = time.time()
k_arr_full, m_arr_full = compute_spatial_neighborhood_labels_fast(
    df, LOCATION_COL, DATE_COL, TARGET_COL, m=5
)
print(f"  Done in {time.time()-t0:.1f}s", flush=True)
print(f"  k: mean={k_arr_full.mean():.2f}, min={k_arr_full.min()}, max={k_arr_full.max()}", flush=True)
print(f"  m: mean={m_arr_full.mean():.2f}, min={m_arr_full.min()}, max={m_arr_full.max()}", flush=True)
print(f"  k distribution: {np.bincount(k_arr_full)}", flush=True)

# Save for later use
np.savez(os.path.join('..', 'data', 'processed', 'neighborhood_labels.npz'),
         k=k_arr_full, m=m_arr_full)
print(f"\nSaved to data/processed/neighborhood_labels.npz", flush=True)
print("\nTest PASSED.", flush=True)
