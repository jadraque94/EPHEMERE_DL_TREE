import rasterio
from rasterio.enums import Resampling
import numpy as np

# Open the 4-band image

path = './bi_classe/images_3bds/*'


for i in range(1,20):

    with rasterio.open(f'./bi_classe/image/image{i}.tif') as dataset:
        # Read the Red, Green, and NIR bands (assuming they are in order R, G, B, NIR)
        img_rgbnir = dataset.read([1, 2, 4])  # Exclude the 3rd band (Blue)

    # Save the new 3-band image
    with rasterio.open(
        f'./bi_classe/images_3bds/image{i}.tif', 
        'w', 
        driver='GTiff',
        height=img_rgbnir.shape[1],
        width=img_rgbnir.shape[2],
        count=3,  # Now 3 bands
        dtype=img_rgbnir.dtype,
        crs=dataset.crs,
        transform=dataset.transform,
    ) as dst:
        dst.write(img_rgbnir)
