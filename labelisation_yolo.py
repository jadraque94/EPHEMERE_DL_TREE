import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio

path_shp = 'C:/Users/rahim/Deeplearning_oct_2024/CHADI_DeepLearning_Tree/Traning_coconut_palm.shp'
path_img = 'C:/Users/rahim/Deeplearning_oct_2024/CHADI_DeepLearning_Tree/Pleiades_14feb2023_pansharpen_Berambadi_decoup_SansForet.tif'

# Load the shapefile
shape = gpd.read_file(path_shp)
image = rio.open(path_img)
img_width = image.width
img_height = image.height

def convert_to_yolo_format(minx, miny, maxx, maxy, img_width, img_height, class_id=1):
    # Calculate center coordinates, width, and height of the bounding box
    x_center = (minx + maxx) / 2.0 / img_width
    y_center = (miny + maxy) / 2.0 / img_height
    width = (maxx - minx) / img_width
    height = (maxy - miny) / img_height

    # YOLO format string
    return f"{class_id} {x_center} {y_center} {width} {height}\n"


def convert_txt(shape, image, path = './first_yolo.txt'):
    img_width = image.width
    img_height = image.height
    with open(path, "a") as f:
    

        for idx, row in shape.iterrows():
            minx, miny, maxx, maxy = row.geometry.bounds
            print(f"Bounding Box for feature {idx}: {minx}, {miny}, {maxx}, {maxy}")
            yolo_annotation = convert_to_yolo_format(minx, miny, maxx, maxy, img_width, img_height)
            f.write(yolo_annotation)
    
convert_txt(shape, image)
