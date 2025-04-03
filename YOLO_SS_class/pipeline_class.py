import os
from pathlib import Path
import rasterio.transform
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os
from shapely.geometry import box
import numpy as np
import glob
from rasterio.mask import mask
import cv2
import rasterio
import geopandas as gpd


class YoloProcessor:
    
    
    def __init__(self, path_tif, path_grid, path_tree, split_ratio = 0.75, output='./image_test/'):
        self.path_tif = path_tif
        self.path_tree = path_tree
        self.path_grid = path_grid
        self.split_ratio = split_ratio
        self.output = Path(output)
    



    def extract_images(self)-> [np.ndarray, dict] : # we will store the small images cutting by the grid in image_array and store the metadata and corner associated at each image and the
        grid = gpd.read_file(self.path_grid).head(100)

        image_array = []
        dict_transform = {}
        with rasterio.open(self.path_tif) as src:
            for idx, row in grid.iterrows():
                geometry = [row.geometry]
                out_image, out_transform = mask(src, geometry, crop=True)

                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                new_bounds = rasterio.transform.array_bounds(out_image.shape[1],out_image.shape[2],out_transform)
                #print(new_bounds)
            
                image_array.append(out_image)
                dict_transform[idx, f'bounds{idx}'] = out_meta, new_bounds # we store the metadata and the 4 corners of every image in the dictionnary
                #print( idx, type(out_image))
        return [image_array, dict_transform]
                

    def separate_grid_unlabel_label(self, array, dicti) -> [np.ndarray, np.ndarray] : #we will separate image regarding as there are any wrosnw of tree which has been delineated and store in unlabel or label dataset
        shape_tree = gpd.read_file(self.path_tree) #shapefile which contains the 524 polygons which represents every tree
        print("Nombre d'arbres délinées :", shape_tree.shape)

        valid_gdf = gpd.GeoDataFrame(columns=["id", 'geometry', 'number_label'], geometry='geometry', crs=shape_tree.crs)

        dicti_unlab = {} # save metadata for unlabel image and bounding box
        dicti_lab = {} #same
        unlabel_image = []
        label_image = []

        for i in tqdm(range((len(array)))):
            valid_polygons = []
            
            xmin_r, ymin_r, xmax_r, ymax_r = dicti[i,f'bounds{i}'][1] # les limites de l'image

            for _, row in shape_tree.iterrows():
                geometry = row.geometry
                xmin_p, ymin_p = np.array(geometry.exterior.coords).min(axis=0)
                xmax_p, ymax_p = np.array(geometry.exterior.coords).max(axis=0)

                #print(xmin_r, xmin_p, xmax_r)

                if xmin_r <= xmin_p <= xmax_r and ymin_r <= ymin_p <= ymax_r and xmin_r <= xmax_p <= xmax_r and ymin_r <= ymax_p <= ymax_r : # condition : every corner of the crowns should be inside of the image
                    valid_polygons.append(row)


            if valid_polygons:
                
                new_data = gpd.GeoDataFrame(valid_polygons)
                new_data['number_label'] =  len(label_image)
                print(new_data) # we will store every group of polygon for the image associated with the same label
                valid_gdf = pd.concat([valid_gdf,new_data], ignore_index=True) 
                dicti_lab[len(label_image),f'bounds{len(label_image)}'] = dicti[i, f'bounds{i}']
                label_image.append(array[i])


                print(len(valid_polygons), i)


            else:
                dicti_unlab[len(unlabel_image), f'bounds{len(unlabel_image)}'] = dicti[i, f'bounds{i}']
                unlabel_image.append(array[i])

        return label_image , valid_gdf, unlabel_image, dicti_lab, dicti_unlab
    


    def train_test_split(self, gdf, image_label, dicti):
       

        split_idx = int(self.split_ratio * len(image_label))
        print(split_idx)
        train_image, test_image = image_label[:split_idx], image_label[split_idx:]
        items = list(dicti.items())
        gdf_train , gdf_test = gdf[gdf['number_label'] < split_idx] , gdf[gdf['number_label'] >= split_idx]
        dict_train, dict_test = dict(items[:split_idx]) , dict(items[split_idx:])

        return train_image, test_image , dict_train, dict_test , gdf_train , gdf_test


    @staticmethod
    def polygon_to_yolo(polygon, image_width, image_height, im_transform):
        minx, miny, maxx, maxy = polygon.bounds
        px_minx, px_miny = ~im_transform * (minx, miny)
        px_maxx, px_maxy = ~im_transform * (maxx, maxy)

        bbox_width = (px_maxx - px_minx) / image_width
        bbox_height = (px_miny - px_maxy) / image_height
        center_x = (px_minx + px_maxx) / 2 / image_width
        center_y = (px_miny + px_maxy) / 2 / image_height

        return center_x, center_y, bbox_width, bbox_height
    



    def convert_yolo(self, gdf_train, dicti_train):

        ## pour train
        length = len(gdf_train.groupby('number_label').size()) # we will convert every polygon into yolo format, we have gather for each image their polygons associated by the same label number
        list_yolo = []
        for i in tqdm(range(length)):
            data = gdf_train[gdf_train['number_label'] == i]
            yolo_annotations = []
            for _, row in data.iterrows():
                polygon = row['geometry']
                class_id = int(row['id']) - 1 #on initalise à 0 car id est 1 et pour utiliser yolo l'identifiant de la premiere classe doit est 0
                image_height = dicti_train[i,f'bounds{i}'][0]['height']
                image_width = dicti_train[i,f'bounds{i}'][0]['width']
                image_transform = dicti_train[i,f'bounds{i}'][0]['transform']

                center_x, center_y, bbox_width, bbox_height = self.polygon_to_yolo(
                    polygon, image_width, image_height, image_transform
                )
                yolo_annotations.append(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}")

            list_yolo.append(yolo_annotations)
        return list_yolo

 

    def orthoimage_to_png_opencv(self, image):

        list_png = []
        for t in range(len(image)):

            band = image[t]
            # Normalize the band to 0-255 (8-bit grayscale)
            band_normalized = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            list_png.append(band_normalized)
            print(len(list_png))
            # Save the normalized image as a PNG using OpenCV
        return list_png


def pansharpen_to_png(self, image , ouput_png_3bds):


    for t in tqdm(range(len(list_image))):

        with rasterio.open(list_image[t]) as dataset:
        # Read the Red, Green, and NIR bands (assuming they are in order R, G, B, NIR)

            print(dataset.count)
            img_rgbnir = dataset.read([4, 3, 2])  # Exclude the 3rd band (Blue)
            #print(np.unique(img_rgbnir))
            band_normalized = cv2.normalize(img_rgbnir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            #print(band_normalized.shape, np.unique(band_normalized))

            # Save the new 3-band image
            with rasterio.open(os.path.join(ouput_png_3bds, f'image{t}.png'),
                'w', 
                driver='GTiff',
                height=band_normalized.shape[1],
                width=band_normalized.shape[2],
                count=3,  # Now 3 bands
                dtype=band_normalized.dtype,
                crs=dataset.crs,
                transform=dataset.transform,
            ) as dst:
                dst.write(band_normalized)

def run_pipeline(self):
    print("Extraction des images...")
    image_array , dicti = self.extract_images()

    image_lab , gdf, image_unlab, dicti_lab, dicti_unlab = self.separate_grid_unlabel_label(image_array, dicti)
    train_image, test_image , dict_train, dict_test , gdf_train , gdf_test = self.train_test_split( gdf, image_lab, dicti_lab )
    yolo_train = self.convert_yolo(gdf_train,dict_train)
    yolo_test = self.convert_yolo(gdf_test,dict_test)
    new_png = self.orthoimage_to_png_opencv(train_image)


    return train_image, test_image , dict_train, yolo_test, yolo_train, new_png

# Utilisation de la classe
if __name__ == "__main__":

    #
    yolo_preprocessor = YoloProcessor(
        path_tif="C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiades_Vue1_2023/C1_orthoimage_forward.tif",
        path_tree="C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/TreeSample_ImagePleiade14feb2023_Pansharpen.shp",
        path_grid="C:/Users/rahim/Deeplearning_oct_2024/CHADI_DeepLearning_Tree/yolo_semi_janv_2025/grid_320_semi.shp",      
        split_ratio = 0.75,
    )



    train_image, test_image , dict_train, yolo_test, yolo_train, train_png = yolo_preprocessor.run_pipeline()