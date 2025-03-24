import os
from pathlib import Path
from tqdm import tqdm
import re
import cv2
from shapely.geometry import box
import numpy as np
import glob
from rasterio.mask import mask
import shutil
import rasterio
import random
import geopandas as gpd


### extract l'image via la grille


name = 'ortho_new_vue1'
path_satellite = 'Pleiades_Vue2_2023/Pansharpen_Vue2_20230214_0535061_Ortho.tif'

path_tif = f"C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/{path_satellite}"
path_tree = "C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/TreeSample_ImagePleiade14feb2023_Pansharpen.shp"
path_shp = "C:/Users/rahim/Deeplearning_oct_2024/CHADI_DeepLearning_Tree/yolo_semi_janv_2025/grid_320_semi.shp"
grid = gpd.read_file(path_shp)



def extract_images_(path_grid, path_tif, output = "C:/Users/rahim/Deeplearning_oct_2024/DeepLearning_EPHEMERE_Tree/Programme_DL/YOLO_SEMI_supervised/image_set/"):

    os.makedirs(output, exist_ok =True)
    p_g = gpd.read_file(path_grid)

    
    with rasterio.open(path_tif) as src:
        for idx, row in p_g.iterrows():
            # Get the geometry of the grid cell
            geometry = [row.geometry]
            out_image, out_transform = mask(src, geometry, crop=True)

            out_meta = src.meta.copy()

            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # Save the clipped image
            output_path = os.path.join(output, f"images{idx}.tif")
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

            print(f"Saved: {output_path}")


    return 

def separate_grid_unlabel_label(path_shp, path_image = "./image_set/*", path_unlabel_ima = "./image_unlabel/" , path_label_box = "./box_label/", path_label_ima = "./image_label/"):


    os.makedirs(path_unlabel_ima, exist_ok = True)
    os.makedirs(path_label_box, exist_ok = True)
    os.makedirs(path_label_ima, exist_ok = True)

    shape_tree = gpd.read_file(path_shp)
    print("le nombre d'arbres délinées est : " ,shape_tree.shape)

    image_paths = sorted( glob.glob(path_image), key=os.path.getctime ) # sorted according the time of creation of each tif image that we split thanks to the grid
    number_label = 0 # number of images and polygons that we will store in the LABELED folder
    number_unlabel = 0

    for image in tqdm(image_paths): # we will select every path of image


        valid_polygons = []
        print(image)
        with rasterio.open(image) as src:

            #print("les coordonnnes de notre image est :", *src.bounds)
            xmin_r, ymin_r, xmax_r, ymax_r = src.bounds

            # Store valid polygons

            for i, row in shape_tree.iterrows():
            # Get the geometry of all of the polygons

                geometry = row.geometry
                # print('coordonnees ext sont :', np.array(geometry.exterior.coords).min(axis=0))

                xmin_p , ymin_p = np.array(geometry.exterior.coords).min(axis=0)
                xmax_p , ymax_p = np.array(geometry.exterior.coords).max(axis=0)
                
                if xmin_r <= xmin_p and xmax_p <= xmax_r and ymin_r <= ymin_p and ymax_p <= ymax_r:
                    valid_polygons.append(row)
            
            print('vrai ou faux' ,len(valid_polygons))
            if len(valid_polygons) != 0:
                valid_gdf = gpd.GeoDataFrame(valid_polygons, crs=shape_tree.crs) # we convert the list into dataframe
                #print(valid_gdf)
                path_box = os.path.join(path_label_box,f'image{number_label}.shp')
                valid_gdf.to_file(path_box)   # Save to a new shapefile

                chemin_lab = os.path.join(path_label_ima,f'image{number_label}.tif')
                shutil.copy2(image,chemin_lab) # we can copy the same image
                number_label += 1



            else : 

                chemin_unlab = os.path.join(path_unlabel_ima,f'image{number_unlabel}.tif')
                shutil.copy2(image ,chemin_unlab) # we can copy the same image
                number_unlabel +=1
                print('number_label', number_label,'/n', 'number_unlabel' , number_unlabel)

def train_test_split(image_label='./image_label/', box_label = './box_label/' , train_box ='./train_box', test_box = './test_box', train_image = './train_image', test_image = './test_image'):

    os.makedirs(train_box, exist_ok = True)
    os.makedirs(test_box, exist_ok = True)
    os.makedirs(train_image, exist_ok= True)
    os.makedirs(test_image , exist_ok= True)

    box_files = sorted(glob.glob(box_label), key=os.path.getctime)
    image_files = sorted(glob.glob(image_label), key=os.path.getctime)

    split_idx_box = int( 0.75 * len(box_files)) #we divided the folder of box_label and image_label into 2 subfolders (train and test) because the length of the box and image folder are same
    train_files_box = box_files[:split_idx_box]
    test_files_box = box_files[split_idx_box:]

    split_idx_im = int(0.75 * len(image_files))
    train_files_images = image_files[:split_idx_im]
    test_files_images = image_files[split_idx_im:]


    for file in train_files_box: # we will copy every box label to a new folder
        #print( 'le fichier est :', Path(file).name)
        shutil.copy2(file, os.path.join(train_box,Path(file).name) )#os.path.join(box_label , file) , os.path.join(train_box , file)) # we can copy the same image

    for file in tqdm(test_files_box) :
        shutil.copy2(file, os.path.join(test_box , Path(file).name ))

    for file in train_files_images:
        shutil.copy2(file, os.path.join(train_image , Path(file).name))
    
    for file in test_files_images:
        shutil.copy2(file, os.path.join(test_image, Path(file).name ))
    return print('FIN DU SPLIT')

def polygon_to_yolo(polygon, image_width, image_height, im_transform):
    minx, miny, maxx, maxy = polygon.bounds  # Get bounding box of the polygon
    # Transform coordinates from geo-space to pixel-space
    px_minx, px_miny = ~im_transform * (minx, miny)  # Convert to pixel coordinates
    px_maxx, px_maxy = ~im_transform * (maxx, maxy)


    # Calculate YOLO format (normalized)
    bbox_width = (px_maxx - px_minx) / image_width
    bbox_height = (px_miny - px_maxy) / image_height
    center_x = (px_minx + px_maxx) / 2 / image_width
    center_y = (px_miny + px_maxy) / 2 / image_height
    
    return center_x, center_y, bbox_width, bbox_height
def extract_number(filename):
    match = re.search(r"(\d+)(?=\.\w+$)", filename)  # Finds digits before file extension
    return int(match.group(1)) if match else float('inf')  # Use 'inf' if no number found

def convert_yolo( box = './train_box/*.shp', box_yolo_path = './train_yolo/', image = './train_image/*'):

    os.makedirs(box_yolo_path, exist_ok=True)
    list_box = sorted(glob.glob(box), key=os.path.getctime)
    list_image = sorted(glob.glob(image), key=extract_number)
    assert len(list_box) == len(list_image)

    for t in range(len(list_image)) :
        print(list_image[t])
        with rasterio.open(list_image[t]) as src:

            image_width = src.width
            image_height = src.height
            image_crs = src.crs
            image_transform = src.transform

        gdf = gpd.read_file(list_box[t])
        #print(list_box[t])

        yolo_annotations = []
        
        for idx, row in gdf.iterrows():
            polygon = row['geometry']
            class_id = int(row['id']) - 1 
            
            # Convert polygon to YOLO format (bounding box)
            center_x, center_y, bbox_width, bbox_height = polygon_to_yolo(polygon, image_width, image_height, image_transform)
            
            # Create YOLO annotation line
            yolo_line = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"
            yolo_annotations.append(yolo_line)


        output_txt_path = os.path.join(box_yolo_path,f'image{t}.txt')
            #print(output_txt_path)


        with open(output_txt_path, 'w') as f:
            for annotation in yolo_annotations:
                f.write(f"{annotation}\n")
        
    return print('FIN DE LA CONVERSION YOLO')
def orthoimage_to_png_opencv(input_image_tif, output_png):

    os.makedirs(output_png)
    list_image = sorted(glob.glob(input_image_tif), key=extract_number)

    for t in range(len(list_image)):

        with rasterio.open(list_image[t]) as src:
            band = src.read(1)  # Read the first (and only) band
        
            # Normalize the band to 0-255 (8-bit grayscale)
            band_normalized = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Save the normalized image as a PNG using OpenCV
        output_path = os.path.join(output_png,f'image{t}.png')
        cv2.imwrite(output_path, band_normalized)
        print(f"Saved PNG: {output_path}")
def pansharpen_to_png(image_pan_tif, ouput_png_3bds):

    os.makedirs(ouput_png_3bds)
    list_image = sorted(glob.glob(image_pan_tif), key=extract_number)

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

def concatenatation_ortho_vue123(output_path , path_v1, path_v2, path_v3):

    os.makedirs(output_path, exist_ok = True)
    back = sorted(glob.glob(path_v1), key=extract_number)
    nadir = sorted(glob.glob(path_v2), key=extract_number)
    forw = sorted(glob.glob(path_v3), key=extract_number)
    print(len(back))

    for i in range(len(forw)):

        b = cv2.imread(back[i],cv2.IMREAD_GRAYSCALE)
        n = cv2.imread(nadir[i],cv2.IMREAD_GRAYSCALE)
        f = cv2.imread(forw[i], cv2.IMREAD_GRAYSCALE)
        print(f.shape)

        image_merge = cv2.merge([b, n, f])
        final_path = os.path.join(output_path,f'./image{i}.png')
        cv2.imwrite(final_path, image_merge)
        #cv2.imshow("RGB_Image", image_merge) 
        #cv2.waitKey(0) 

    return print('MERGE IS FINISHED')


# extract_images_(path_shp, path_tif, output = f"C:/Users/rahim/Deeplearning_oct_2024/DeepLearning_EPHEMERE_Tree/Programme_DL/YOLO_SS_class/image_set_{name}/")
# separate_grid_unlabel_label(path_tree, path_image = f"./image_set_{name}/*", 
# path_unlabel_ima = f"./image_unlabel_{name}/" ,path_label_box = f"./box_label_{name}/", path_label_ima = f"./image_label_{name}/")
# train_test_split(image_label = f'./image_label_{name}/*', box_label = f'./box_label_{name}/*', train_box =f'./train_box_{name}', test_box = f'./test_box_{name}', train_image = f'./train_image_{name}', test_image = f'./test_image_{name}' )



#ON CONVERTIT LES COORDONNEES DES BOX EN FORMAT YOLO
# convert_yolo(box = f'./test_box_{name}/*.shp', box_yolo_path = f'./test_yolo_{name}/' , image = f'./test_image_{name}/*.tif')
# convert_yolo(box = f'./train_box_{name}/*.shp', box_yolo_path = f'./train_yolo_{name}/' , image = f'./train_image_{name}/*.tif')

# # ON CONVERTIT LES IMAGE EN FORMAT TIF EN FORMAT PNG

# orthoimage_to_png_opencv( f'./image_unlabel_{name}/*', f'./png_unlabel_{name}/')
# orthoimage_to_png_opencv( f'./train_image_{name}/*', f'./png_train_image_{name}/')
#orthoimage_to_png_opencv( f'./test_image_{name}/*', f'./png_test_image_{name}/')


# pansharpen_to_png(image_pan_tif = f'./image_unlabel_{name}/*', ouput_png_3bds = f'./png_unlabel_{name}/')
# pansharpen_to_png(image_pan_tif = f'./train_image_{name}/*', ouput_png_3bds = f'./png_train_{name}/')
# pansharpen_to_png(image_pan_tif = f'./test_image_{name}/*', ouput_png_3bds = f'./png_test_{name}/')


###ON CONCATENE 
# concatenatation_ortho_vue123(output_path= './ortho_full/train/', path_v1 ='./ortho_v1/png_train_image_ortho_new_vue1/*' , 
#                              path_v2 ='./ortho_v2/png_train_image_ortho_new_vue2/*', path_v3 = './ortho_v3/png_train_image_ortho_new_vue3/*')

# concatenatation_ortho_vue123(output_path= './ortho_full/unlabel/', path_v1 ='./ortho_v1/png_unlabel_ortho_new_vue1/*' , 
#                            path_v2 ='./ortho_v2/png_unlabel_ortho_new_vue2/*', path_v3 = './ortho_v3/png_unlabel_ortho_new_vue3/*')

concatenatation_ortho_vue123(output_path= './ortho_full/test/', path_v1 ='./ortho_v1/png_test_image_ortho_new_vue1/*' , 
                           path_v2 ='./ortho_v2/png_test_image_ortho_new_vue2/*', path_v3 = './ortho_v3/png_test_image_ortho_new_vue3/*')

                    