import os
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import ee
import re
import math
import sys
sys.path.append('../')
from main import fetch_corestack_data

# Create exports directory
os.makedirs('./exports', exist_ok=True)

# Step 1: Fetch CoreStack data for cropping intensity
# LOCATION parameter is hardcoded as per the prompt's context.
LOCATION = "Navalgund Dharwad Karnataka"

result = fetch_corestack_data(query=f"{LOCATION} cropping intensity")
data = json.loads(result)

vector_layers = data['spatial_data']['vector_layers']

print("Available vector layers:")
for layer in vector_layers:
    print(f" - {layer['layer_name']}")

# Step 2: Load Cropping Intensity layer and calculate mean CI
ci_gdf = None

for layer in vector_layers:
    lname = layer['layer_name'].lower()
    if 'cropping' in lname and 'intensity' in lname and ci_gdf is None:
        print(f"Found Cropping Intensity layer: {layer['layer_name']}")
        ci_gdf = pd.concat(
            [gpd.read_file(u['url']) for u in layer['urls']],
            ignore_index=True
        ).to_crs('EPSG:4326')

        print(f"CI shape: {ci_gdf.shape}, columns: {ci_gdf.columns.tolist()}")

# Calculate mean CI for years 2017–2023
ci_cols = [c for c in ci_gdf.columns if c.startswith('cropping_intensity_')]

ci_cols_filtered = [
    c for c in ci_cols
    if 2017 <= int(c.split('_')[-1]) <= 2023
]

ci_gdf['mean_ci'] = (
    ci_gdf[ci_cols_filtered]
    .apply(pd.to_numeric, errors='coerce')
    .mean(axis=1)
)
breakpoint()

print(f"CI year columns used: {ci_cols_filtered}")
print(f"Mean CI range: {ci_gdf['mean_ci'].min():.1f} to {ci_gdf['mean_ci'].max():.1f}")
ci_gdf.to_file("test_ci.geojson", driver='GeoJSON')