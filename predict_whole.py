import os
from affine import Affine
import rasterio.transform
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import pathlib 
import glob
from shapely.geometry import box
import numpy as np
from ultralytics import YOLO
from rasterio.mask import mask
import cv2
import rasterio
import geopandas as gpd


class Extract():

    def __init__(self, output_tif_image, path_grid, model, band1, band2, band3):
        self.output_tif_image = output_tif_image
        self.path_grid = path_grid
        self.model = model
        self.band1 = band1
        self.band2 = band2
        self.band3 = band3

    def extract_images(self, path, out_path) -> tuple[list, dict] : # we will store the small images cutting by the grid in image_array and store the metadata and corner associated at each image and the

        grid = gpd.read_file(self.path_grid)
        image_array = []
        dict_transform = {}

        pathlib.Path(out_path).mkdir(parents =True, exist_ok = True)

        with rasterio.open(path) as src:
            for idx, row in tqdm(grid.iterrows()):
                # if idx == 300:
                #     return image_array, dict_transform
                

                out = os.path.join(out_path, f'image{idx}.tif')

                geometry = [row.geometry]
                out_image, out_transform = mask(src, geometry, crop=True)

                ### we will store this image in a folder for visualize with the predictions made after

                out_image = np.transpose(out_image, (1, 2, 0))  # Transpose to (H, W, C) for using opencv and doing other manipulations
                if out_image.shape[2] > 3 : # multispectral case
                    row_keep = [self.band1,self.band2,self.band3]
                    keep_image = out_image[:,:,row_keep]
                    band_normalized = cv2.normalize(keep_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore
                    #print(band_normalized.shape)
                    out_meta = src.meta.copy()

                    out_meta.update({
                        "driver": "GTiff",
                        "height": band_normalized.shape[0],
                        "width": band_normalized.shape[1],
                        "count" : band_normalized.shape[2],
                        "transform": out_transform
                    })

                    new_bounds = rasterio.transform.array_bounds(band_normalized.shape[0],band_normalized.shape[1],out_transform)
                    image_array.append(band_normalized)
                    t = np.transpose(band_normalized , (2,0,1)) 
                    

                    with rasterio.open(out, 'w' , **out_meta) as dst:
                        dst.write(t)
                    dict_transform[idx, f'bounds{idx}'] = out_meta, new_bounds # we store the metadata and the 4 corners of every image in the dictionnary

                elif  out_image.shape[2] == 1 : # image has 3 or less bands, yolo can just be feed with image with 3 bands or less

                    band_normalized = cv2.normalize(out_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore

                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": band_normalized.shape[0],
                        "width": band_normalized.shape[1],
                        "count" : 1,
                        "transform": out_transform,
                    })

                    new_bounds = rasterio.transform.array_bounds(band_normalized.shape[0],band_normalized.shape[1],out_transform)
                    image_array.append(band_normalized)


                    with rasterio.open(out, 'w' , **out_meta) as dst:
                        dst.write(band_normalized,1)
                    dict_transform[idx, f'bounds{idx}'] = out_meta, new_bounds # we store the metadata and the 4 corners of every image in the dictionnary


        return image_array, dict_transform
    
    @staticmethod 
    def update_affine(dicti):
        for i in range(len(dicti)):
            trans = dicti[i,f'bounds{i}'][0]['transform']
            height = int(dicti[i,f'bounds{i}'][0]['height'])
            width = int(dicti[i,f'bounds{i}'][0]['width'])

            flip_up = trans * Affine(1,0,0,0,-1,height)
            rot_flip = flip_up * Affine.rotation(270)
            transla = rot_flip *Affine.translation(-width,0)
            dicti[i,f'bounds{i}'][0].update({'flip_up' : flip_up, 'final_transf' : rot_flip, 'transla' : transla})
        return dicti

    def concat_output(self, path, im1,im2,im3):

        pathlib.Path(self.output_tif_image).mkdir(parents =True, exist_ok = True)
        records = []

        for i in tqdm(range(len(im1))): #im1,im2 and im3  have the same length

            file = os.path.join(path, f'image{i}.tif') #retrieve the correct small image for the metadata
            with rasterio.open(file) as src:
                band = src.read()
                out_meta = src.meta.copy()
                out_meta.update({
                    "count" : 3})
                
            out = os.path.join(self.output_tif_image, f'image{i}.tif')
            try :
                image = np.stack((im1[i], im2[i], im3[i]))
                records.append(np.dstack((im1[i], im2[i], im3[i])))

            except :
                min1 = min(im1[i].shape[0], im2[i].shape[0], im3[i].shape[0])
                print(min1)
                min2 = min(im1[i].shape[1], im2[i].shape[1], im3[i].shape[1])
                print(min2)

                new_im1 = im1[i][:min1,:min2] # we will reshape every image to concatenatethem
                new_im2 = im2[i][:min1,:min2]
                new_im3 = im3[i][:min1,:min2]

                print(new_im1.shape, new_im2.shape, new_im3.shape)


                image = np.stack((new_im1, new_im2, new_im3))
                records.append(np.dstack((new_im1, new_im2, new_im3)))
            with rasterio.open(out, 'w' , **out_meta) as dst:
                dst.write(image)
        
        return records


    @staticmethod
    def extract_number(filename: str):
    # Extrait le premier nombre trouvé dans le nom de fichier (utile pour trier)
        match = re.search(r"(\d+)(?=\.)", filename)
        return int(match.group(1)) if match else float('inf')



    def predict(self, output_gdf):# dicti, affine_transf, ) -> [pd.DataFrame, list]:

        grid = gpd.read_file(self.path_grid)
        crs_projet = grid.crs

        records = []
        # for i in tqdm(range(len(list_image))):
        #     image_pred = list_image[i]
        out = os.path.join(self.output_tif_image, '*.tif')
        print(out)


        path_ = glob.glob(out)
        image_label = sorted(path_, key = self.extract_number)    
        for i in range(len(image_label)):
            print(image_label[i])
            with rasterio.open(image_label[i]) as src:
                bo = src.read()
                transf = src.transform
                bot = np.transpose(bo, (2,1,0))

            results = self.model.predict(source = bot , task='segment', conf = 0.2, iou = 0.4 )

            for result in results:
                boxes = result.boxes  # List of bounding boxes in the format [x1, y1, x2, y2]
                #fig, ax = plt.subplots(figsize=(15, 15))

                for b in boxes:
                    # Extract the bounding box coordinates
                    x1, y1, x2, y2 = map(int, b.xyxy[0])  # Convert to integers for OpenCV
                    geom1 = box(x1, y1, x2, y2)

                    #cv2.rectangle(image_pred, (x1, y1), (x2, y2), (0, 0, 0), 2)

                    #transf = dicti[i,f'bounds{i}'][0][affine_transf]
                    transformer = rasterio.transform.AffineTransformer(transf)
                    xx1 , yy1 = transformer.xy( x1, y1)
                    xx2 , yy2 = transformer.xy( x2, y2)
                    
                    geom_crs = box(xx1, yy1, xx2, yy2)
                    records.append({
                        'label' : i ,
                        'geometry_norm' : geom1,
                        'geometry' : geom_crs,
                        })
                    


                    
                # ax.imshow(image_pred, cmap='RdBu')
                #plt.show()
        gdf = gpd.GeoDataFrame(records)  # or another CRS if you know it
        gdf.set_crs(crs_projet, inplace= True)
        gdf.to_file(output_gdf)

        return gdf



if __name__ == "__main__":

    # Panchro / Tri-stereo : forward, nadir, backward
    # Multi/Hyperspectral : band1, band2; band3
    # RGB : Red, Green, Blue

    path_tif_C1 = "C1_orthoimage_forward.tif"
    path_tif_C2 = "C2_orthoimage_nadir.tif"
    path_tif_C3 = "C3_orthoimage_backward.tif"
    out_path_C1 = './whole_image_C1/'
    out_path_C2 = './whole_image_C2/'
    out_path_C3 = './whole_image_C3/'

    whole_image = Extract(
    output_tif_image = 'whole_image_tif/',
    path_grid = "grid.shp",
    model = YOLO("./weight/yolo.pt"),
    band1 = 3,
    band2 = 2,
    band3 = 1,
    )


    # image_C1, dicti = whole_image.extract_images(path_tif_C1, out_path_C1)
    # image_C2, dicti = whole_image.extract_images(path_tif_C2, out_path_C2)
    # image_C3, dicti = whole_image.extract_images(path_tif_C3, out_path_C3)

    # print('le nouveau records')
    # records = whole_image.concat_output(out_path_C1,image_C1 ,image_C2 ,image_C3)


    # print('Change to update')
    # new_dictionnary = whole_image.update_affine(dicti)
    # g2 = whole_image.predict(new_dictionnary, 'transform', './df_total_1300TREE.shp')

    g2 = whole_image.predict('./map_predict.shp')


