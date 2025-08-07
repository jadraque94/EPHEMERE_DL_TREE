import os
from tqdm import tqdm
import rasterio.transform
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box
import pathlib
import numpy as np
from rasterio.mask import mask
import cv2
import rasterio
import geopandas as gpd


class YoloProcessor:
    
    
    def __init__(self, path_tree, path_grid, band1 , band2, band3 , split_ratio):
        self.path_tree = path_tree
        self.path_grid = path_grid
        self.band1 = band1 
        self.band2 = band2
        self.band3 = band3
        self.split_ratio = split_ratio


    def extract_images(self, path_tif)-> [ np.ndarray, dict] : # we will store the small images cutting by the grid in image_array and store the metadata and corner associated at each image and the

        grid = gpd.read_file(self.path_grid)

        image_array = []
        dict_transform = {}
        with rasterio.open(path_tif) as src:
            for idx, row in tqdm(grid.iterrows()):
                geometry = [row.geometry]
                out_image, out_transform = mask(src, geometry, crop=True)
                out_image = np.transpose(out_image, (1, 2, 0))  # Transpose to (H, W, C)

                ##condition which allow us to be get just 3 bands or 1 band in the case of panchromatic, because yolo work
                if out_image.shape[2] > 3 : #multispecral case
                    row_keep = [self.band1,self.band2,self.band3]
                    keep_image = out_image[:,:,row_keep]
                    band_normalized = cv2.normalize(keep_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    #print(band_normalized.shape)
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
                    t = np.transpose(band_normalized , (2,0,1)) 


            
                
                
                elif  out_image.shape[2] == 3 : # image has 3 bands, yolo can just be feed with image with 3 bands or less
                    band_normalized = cv2.normalize(out_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

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


                elif  out_image.shape[2] == 1 : # image has 1 band, yolo can just be feed with image with 3 bands or less

                    band_normalized = cv2.normalize(out_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": band_normalized.shape[0],
                        "width": band_normalized.shape[1],
                        "count" : 1,
                        "transform": out_transform
                    })

                    new_bounds = rasterio.transform.array_bounds(band_normalized.shape[0],band_normalized.shape[1],out_transform)
                
                    image_array.append(band_normalized)
                    dict_transform[idx, f'bounds{idx}'] = out_meta, new_bounds # we store the metadata and the 4 corners of every image in the dictionnary
                
         
        return image_array, dict_transform

                

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


                if xmin_r <= xmin_p <= xmax_r and ymin_r <= ymin_p <= ymax_r and xmin_r <= xmax_p <= xmax_r and ymin_r <= ymax_p <= ymax_r : # condition : every corner of the crowns should be inside of the image
                    valid_polygons.append(row)


            if valid_polygons:
                
                new_data = gpd.GeoDataFrame(valid_polygons)
                new_data['number_label'] =  len(label_image)
                #print(new_data) # we will store every group of polygon for the image associated with the same label
                valid_gdf = pd.concat([valid_gdf,new_data], ignore_index=True) 
                dicti_lab[len(label_image),f'bounds{len(label_image)}'] = dicti[i, f'bounds{i}']
                label_image.append(array[i])


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
                if row['id'] == np.nan :
                    continue

                class_id = int(row['id']) - 1 #on initalise à 0 car l'identifiant est 1 et pour utiliser yolo l'identifiant de la premiere classe doit est 0
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
    def create_folder_txt(label, direction, output) -> list: # for creating folder for all of the label file 


        file_roots = os.path.join(output, direction)
        pathlib.Path(file_roots).mkdir(parents =True, exist_ok = True)
        for t in tqdm(range(len(label))):

            output_txt_path = os.path.join(file_roots, f'image{t}.txt')

            with open(output_txt_path, 'w') as f:
                for annotation in label[t] :

                    values = annotation.split()
                    if len(values) == 5:  # Ensure it has exactly 5 elements
                        class_id, center_x, center_y, bbox_width, bbox_height = map(float, values)
                        class_id = int(class_id)
                        yolo_line = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"
                        f.write(f"{yolo_line}\n")
    
    @staticmethod
    def create_folder_image(image, direction, dicti_image, number_band, output_image) -> list : # for creating folder for all of the label file 


        file_roots = os.path.join(output_image, direction)
        pathlib.Path(file_roots).mkdir(parents =True, exist_ok = True)
        for t in tqdm(range(len(image))):

            output_image_path = os.path.join(file_roots, f'image{t}.png')
            
            index = list(dicti_image.items())[t][1][0]


            with rasterio.open(output_image_path,
                'w', 
                driver = index['driver'],
                height = index['height'],
                width = index['width'],
                count = number_band,  
                dtype = 'uint8',
                crs = index['crs'],
                transform = index['transform'],
            ) as dst :
                dst.write(image[t],1)


    @staticmethod
    def concat_output(im1,im2,im3, path, direction, output_tif_image):


        file_roots = os.path.join(output_tif_image, direction)
        pathlib.Path(file_roots).mkdir(parents =True, exist_ok = True)
        records = []

        for i in range(len(im1)): #im1,im2 and im3  have the same length

            file = os.path.join(path, direction, f'image{i}.png') #retrieve the correct small image for the metadata
            with rasterio.open(file) as src:
                out_meta = src.meta.copy()
                out_meta.update({
                    "count" : 3})
                
            out = os.path.join(output_tif_image, direction, f'image{i}.png')
            image = np.stack((im1[i], im2[i], im3[i]))

            records.append(np.dstack((im1[i], im2[i], im3[i])))

            with rasterio.open(out, 'w' , **out_meta) as dst:
                dst.write(image)
        
        return records          
               
    

    def run_pipeline(self, path_tif, test, train,  output):
        print("Extraction des images...")
        image_array , dicti = self.extract_images(path_tif)

        print("Change ")

        image_lab , gdf, image_unlab, dicti_lab, dicti_unlab = self.separate_grid_unlabel_label(image_array, dicti)
        train_image, test_image , dict_train, dict_test , gdf_train , gdf_test = self.train_test_split( gdf, image_lab, dicti_lab )
        yolo_train = self.convert_yolo(gdf_train,dict_train)
        yolo_test = self.convert_yolo(gdf_test,dict_test)


        self.create_folder_txt(yolo_test, test, output )
        self.create_folder_txt(yolo_train, train, output )

        self.create_folder_image(train_image, direction = train, dicti_image = dict_train, number_band = 1, output_image = output)
        self.create_folder_image(test_image, direction = test, dicti_image = dict_test, number_band = 1, output_image = output)
        return  train_image, dict_train, test_image, dict_test, image_unlab, dicti_unlab

# Utilisation de la classe
if __name__ == "__main__":

    path_tif_C1 = "C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiades_Vue1_2023/C1_orthoimage_forward.tif"
    path_tif_C2 = "C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiades_Vue2_2023/C2_orthoimage_nadir.tif"
    path_tif_C3 = "C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiades_Vue3_2023/C3_orthoimage_backward.tif"
    out_path_C1 = './last_whole_image_C1'
    out_path_C2 = './last_whole_image_C2'
    out_path_C3 = './last_whole_image_C3'
    out_path_ortho = './last_whole_ortho'
    train = 'train'
    test = 'test'

    yolo_preprocessor = YoloProcessor(
        path_grid = "C:/Users/rahim/Deeplearning_oct_2024/CHADI_DeepLearning_Tree/yolo_semi_janv_2025/grid_320_semi.shp",
        path_tree = "C:/Users/rahim/Deeplearning_oct_2024/CHADI_DeepLearning_Tree/yolo_semi_janv_2025/TreeSample_last.shp",      
        band1 = 3,
        band2 = 2,
        band3 = 1,
        split_ratio = 0.7,

    )


    train_image_c1, dicti_train_c1 ,test_image_c1, dicti_test_c1, image_unlab_c1, dicti_unlab_c1 = yolo_preprocessor.run_pipeline(path_tif_C1, test, train, out_path_C1)
    train_image_c2, dicti_train_c2, test_image_c2, dicti_test_c2, image_unlab_c2, dicti_unlab_c2 = yolo_preprocessor.run_pipeline(path_tif_C2, test, train, out_path_C2)
    train_image_c3, dicti_train_c3, test_image_c3, dicti_test_c3, image_unlab_c3, dicti_unlab_c3 = yolo_preprocessor.run_pipeline(path_tif_C3, test, train, out_path_C3)


    image_unlabel_c1 = yolo_preprocessor.create_folder_image(image_unlab_c1, 'unlabel', dicti_unlab_c1, number_band=1, output_image = out_path_C1)
    image_unlabel_c2 = yolo_preprocessor.create_folder_image(image_unlab_c2, 'unlabel', dicti_unlab_c2, number_band=1, output_image = out_path_C2)
    image_unlabel_c3 = yolo_preprocessor.create_folder_image(image_unlab_c3, 'unlabel', dicti_unlab_c3, number_band=1, output_image = out_path_C3)


    #### parti TRAIN
    yolo_preprocessor.concat_output(train_image_c1, train_image_c2, train_image_c3, out_path_C1,
                                     train ,out_path_ortho)

    #### parti TEST
    yolo_preprocessor.concat_output(test_image_c1, test_image_c2, test_image_c3, out_path_C1, 
                                    test, out_path_ortho)
    
    #### parti UNLABEL
    yolo_preprocessor.concat_output(image_unlab_c1, image_unlab_c2, image_unlab_c3, out_path_C1, 
                                    'unlabel', out_path_ortho)


