import rasterio
import cv2
import geopandas as gpd
import matplotlib.pyplot as plt


# Step 4: Function to convert bounding box to YOLO format
def polygon_to_yolo(polygon, image_width, image_height):
    minx, miny, maxx, maxy = polygon.bounds  # Get bounding box of the polygon
    # Transform coordinates from geo-space to pixel-space
    px_minx, px_miny = ~image_transform * (minx, miny)  # Convert to pixel coordinates
    px_maxx, px_maxy = ~image_transform * (maxx, maxy)


    # Calculate YOLO format (normalized)
    bbox_width = (px_maxx - px_minx) / image_width
    bbox_height = (px_miny - px_maxy) / image_height
    center_x = (px_minx + px_maxx) / 2 / image_width
    center_y = (px_miny + px_maxy) / 2 / image_height
    
    return center_x, center_y, bbox_width, bbox_height

# Step 5: Convert each polygon to YOLO format
yolo_annotations = []

# print("YOLO annotations saved!")
n = 194

for i in range(1,n):
    file_tif = f'C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue1_2023_traitement/Pleiade_Vue1_image/image{i}.tif'
    file_shp = f"C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue1_2023_traitement/Pleiade_Vue1_label/image{i}.shp"
    file_png = f'C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue1_2023_traitement/Pleiade_Vue1_image_png/image{i}.png'



    with rasterio.open(file_tif) as src:
        image_width = src.width
        image_height = src.height
        image_crs = src.crs
        image_bounds = src.bounds
        image_transform = src.transform
        
    gdf = gpd.read_file(file_shp)
    print(i,gdf.columns)
    yolo_annotations = []
    for idx, row in gdf.iterrows():
        polygon = row['geometry']
        class_id = int(row['id']) - 1 
        
        # Convert polygon to YOLO format (bounding box)
        center_x, center_y, bbox_width, bbox_height = polygon_to_yolo(polygon, image_width, image_height)
        
        # Create YOLO annotation line
        yolo_line = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"
        yolo_annotations.append(yolo_line)

    output_txt_path = f'C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue1_2023_traitement/Pleiade_Vue1_texte/image{i}.txt'
    with open(output_txt_path, 'w') as f:
        for annotation in yolo_annotations:
            f.write(f"{annotation}\n")