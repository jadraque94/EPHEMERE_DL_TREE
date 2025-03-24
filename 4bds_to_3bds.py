import rasterio
from rasterio.enums import Resampling
import numpy as np
import cv2
import os
# Open the 4-band image

#path = './bi_classe/images_3bds/*'


for i in range(194,387):

    with rasterio.open(f"C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue3_2023_traitement/Pleiade_Vue3_image/image{i}.tif") as dataset:
        # Read the Red, Green, and NIR bands (assuming they are in order R, G, B, NIR)
        img_rgbnir = dataset.read([4, 3, 2])  # Exclude the 3rd band (Blue)

    # Save the new 3-band image
    with rasterio.open(
        f'C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue3_2023_traitement/Pleiade_Vue3_image/image{i}.tif', 
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


tif_path = "C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue3_2023_traitement/Pleiade_Vue3_image"
png_path = "C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue3_2023_traitement/Pleiade_Vue3_image_png"

def tif_to_png_opencv(input_tif, output_png, n=194):

    for i in range(n,400):
    # Open the TIF image using rasterio
        fix_tif = f"image{i}.tif" 
        fix_png = f"image{i}.png"

        input_path = os.path.join(input_tif,fix_tif)
        with rasterio.open(input_path) as src:
            band = src.read(1)  # Read the first (and only) band
        
            # Normalize the band to 0-255 (8-bit grayscale)
            band_normalized = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Save the normalized image as a PNG using OpenCV
        output_path = os.path.join(output_png,fix_png)
        cv2.imwrite(output_path, band_normalized)
        print(f"Saved PNG: {output_path}")

# Example usage



tif_to_png_opencv(tif_path, png_path)
