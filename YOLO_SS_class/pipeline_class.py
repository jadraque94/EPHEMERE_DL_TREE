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
    
    
    def __init__(self, path_tif, path_grid, path_tree, band1 , band2, band3 , split_ratio = 0.75,  output='./image_test/'):
        self.path_tif = path_tif
        self.path_tree = path_tree
        self.path_grid = path_grid
        self.band1 = band1 
        self.band2 = band2
        self.band3 = band3
        self.split_ratio = split_ratio


        self.output = Path(output)
    



    def extract_images(self)-> [ np.ndarray, dict] : # we will store the small images cutting by the grid in image_array and store the metadata and corner associated at each image and the
        grid = gpd.read_file(self.path_grid).head(100) #100 premières images pour vérifier

        image_array = []
        dict_transform = {}
        with rasterio.open(self.path_tif) as src:
            for idx, row in grid.iterrows():
                geometry = [row.geometry]
                out_image, out_transform = mask(src, geometry, crop=True)
                out_image = np.transpose(out_image, (1, 2, 0))  # Transpose to (H, W, C)

                ##condition which allow us to be get just 3 bands or 1 band in the case of panchromatic, because yolo work
                if out_image.shape[2] > 3 : #multispecral case
                    row_keep = [self.band1,self.band2,self.band3]
                    keep_image = out_image[:,:,row_keep]
                    band_normalized = cv2.normalize(keep_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    print(band_normalized.shape)
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": band_normalized.shape[0],
                        "width": band_normalized.shape[1],
                        "transform": out_transform
                    })

                    new_bounds = rasterio.transform.array_bounds(band_normalized.shape[0],band_normalized.shape[1],out_transform)
                
                    image_array.append(band_normalized)
                    dict_transform[idx, f'bounds{idx}'] = out_meta, new_bounds # we store the metadata and the 4 corners of every image in the dictionnary

            
                
                
                elif  out_image.shape[2] <=3 :
                    band_normalized = cv2.normalize(out_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    print(band_normalized.shape)
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": band_normalized.shape[0],
                        "width": band_normalized.shape[1],
                        "transform": out_transform
                    })

                    new_bounds = rasterio.transform.array_bounds(band_normalized.shape[0],band_normalized.shape[1],out_transform)
                    #print(new_bounds)
                
                    image_array.append(band_normalized)
                    dict_transform[idx, f'bounds{idx}'] = out_meta, new_bounds # we store the metadata and the 4 corners of every image in the dictionnary


                    #print( idx, type(out_image))
                
         
        return [image_array, dict_transform]

                

    def separate_grid_unlabel_label(self, array, dicti) -> [np.ndarray, pd.DataFrame, np.ndarray, dict, dict ] : #we will separate image regarding as there are any wrosnw of tree which has been delineated and store in unlabel or label dataset
        shape_tree = gpd.read_file(self.path_tree) #shapefile which contains the 524 polygons which represents every tree

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

        return label_image, valid_gdf, unlabel_image, dicti_lab, dicti_unlab
    


    def train_test_split(self, gdf, image_label, dicti) -> [np.ndarray, np.ndarray, dict, dict, pd.DataFrame, pd.DataFrame]:

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
    



    def convert_yolo(self, gdf, dicti) -> list: #convert format shapefile into yolo format

        ## pour train
        length = len(gdf.groupby('number_label').size()) # we will convert every polygon into yolo format, we have gather for each image their polygons associated by the same label number
        list_yolo = []
        for i in tqdm(range(length)):
            index = np.unique(gdf.number_label)[i]
            data = gdf[gdf['number_label'] == index ]
            yolo_annotations = []
            for _, row in data.iterrows():
                polygon = row['geometry']
                class_id = int(row['id']) - 1 #on initalise à 0 car id est 1 et pour utiliser yolo l'identifiant de la premiere classe doit est 0
                image_height = dicti[index,f'bounds{index}'][0]['height']
                image_width = dicti[index,f'bounds{index}'][0]['width']
                image_transform = dicti[index,f'bounds{index}'][0]['transform']

                center_x, center_y, bbox_width, bbox_height = self.polygon_to_yolo(
                    polygon, image_width, image_height, image_transform
                )
                yolo_annotations.append(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}")

            list_yolo.append(yolo_annotations)
        return list_yolo

    
    @staticmethod
    def create_folder(label, output):

        os.makedirs(output, exist_ok =True)
        for t in tqdm(range(len(label))):

            output_txt_path = os.path.join(output,f'image{t}.txt')
            # print(output_txt_path)
            with open(output_txt_path, 'w') as f:
                for annotation in label[t] :

                    values = annotation.split()
                    if len(values) == 5:  # Ensure it has exactly 5 elements
                        class_id, center_x, center_y, bbox_width, bbox_height = map(float, values)
                        class_id = int(class_id)
                        yolo_line = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"
                        print(yolo_line)
                        f.write(f"{yolo_line}\n")



    def run_pipeline(self):
        print("Extraction des images...")
        image_array , dicti = self.extract_images()

        image_lab , gdf, image_unlab, dicti_lab, dicti_unlab = self.separate_grid_unlabel_label(image_array, dicti)
        train_image, test_image , dict_train, dict_test , gdf_train , gdf_test = self.train_test_split( gdf, image_lab, dicti_lab )
        yolo_train = self.convert_yolo(gdf_train,dict_train)
        yolo_test = self.convert_yolo(gdf_test,dict_test)


        return yolo_train, yolo_test, train_image, test_image, gdf_test, dict_test, gdf_train

# Utilisation de la classe
if __name__ == "__main__":

    #
    yolo_preprocessor = YoloProcessor(
        path_tif="image.tif",
        path_tree="tree.shp",
        path_grid="grid.shp",      
        band1 = 3,
        band2 = 2,
        band3 = 1,
        split_ratio = 0.75,

    )


    yolo_preprocessor.run_pipeline()
