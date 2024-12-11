import os
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import numpy as np

geotiff_paths = ["ASTGTMV003_N36W117_dem.tif", "ASTGTMV003_N36W118_dem.tif", "ASTGTMV003_N36W119_dem.tif"]

# Open the GeoTIFF files as datasets
datasets = [rasterio.open(path) for path in geotiff_paths]

# Merge the datasets
merged_data, merged_transform = merge(datasets)


# Save the merged raster to a new GeoTIFF (optional)
output_path = "death_valley_whitney.tif"
with rasterio.open(
    output_path,
    "w",
    driver="GTiff",
    height=merged_data.shape[1],
    width=merged_data.shape[2],
    count=1,
    dtype=merged_data.dtype,
    crs=datasets[0].crs,
    transform=merged_transform,
) as dest:
    dest.write(merged_data[0], 1)  # Write the first band of the merged data

