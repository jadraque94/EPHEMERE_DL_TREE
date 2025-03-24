import numpy as np
import geopandas as gpd
import cv2
import os
import rasterio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class Preprocessing:
     
     """Handles image loading, conversion, and saving."""

def __init__(self, path_grid = 'grid', path_tif = ' tif', path_label="label", n = 1):
        
        self.path_label = path_label
        self.path_grid = path_grid
        self.path_tif = path_tif
        self.n = n 



def preprocess_grid(self):
    classe_geo = []
    classe_img = []

    with rasterio.open(self.path_tif) as src:
        for i, row in grid.iterrows():
            # Get the geometry of the grid cell
            geometry = [row.geometry]
            geometry_shp = row.geometry

            # Clip the raster with the grid cell
            out_image, out_transform = rasterio.mask(src, geometry, crop=True)
            out_meta = src.meta.copy()

            # Update metadata for the sub-image
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            
            points_in_rectangle = train_point[train_point.geometry.within(geometry_shp)]
            if points_in_rectangle.shape[0] != 0 :
                
                print(points_in_rectangle.shape,self.n)
                classe_geo = classe_geo.append([points_in_rectangle])

                out_image, out_transform = rasterio.mask(src, geometry, crop=True)
                out_meta = src.meta.copy()

                # Update metadata for the sub-image
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                n = n + 1
                classe_img = classe_img.append([out_image])