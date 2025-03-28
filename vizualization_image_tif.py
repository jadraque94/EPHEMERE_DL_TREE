import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import glob

# Step 1: Open the GeoTIFF image using rasterio
# path = "./images/train/train_image_yolo_1.tif"
path_shp = "./bi_classe/shapefile//tree4.shp"
gdf = gpd.read_file(path_shp)
print(gdf.columns)


# with rasterio.open(path) as src:
#     image_width = src.width
#     image_height = src.height
#     image_crs = src.crs
#     image_bounds = src.bounds
#     image_transform = src.transform  # For transforming coordinates

# # Step 3: Reproject the shapefile to match the CRS of the image (if necessary)
# if gdf.crs != image_crs:
#     gdf = gdf.to_crs(image_crs)

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

# for idx, row in gdf.iterrows():
#     polygon = row['geometry']
#     print(row['Id'])
#     # Convert polygon to YOLO format (bounding box)
#     center_x, center_y, bbox_width, bbox_height = polygon_to_yolo(polygon, image_width, image_height)
    
#     # Create YOLO annotation line
#     yolo_line = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"
#     yolo_annotations.append(yolo_line)

# Step 6: Save YOLO annotations to a text file
# output_txt_path = './Programme_DL/label/train/anno_train_1.txt'
# with open(output_txt_path, 'w') as f:
#     for annotation in yolo_annotations:
#         f.write(f"{annotation}\n")

# print("YOLO annotations saved!")
    
for i in range(1,11):
    file_tif = f'./bi_classe/image/image{i}.tif'
    file_shp = f"./bi_classe/shapefile/tree{i}.shp"

    with rasterio.open(file_tif) as src:
        image_width = src.width
        image_height = src.height
        image_crs = src.crs
        image_bounds = src.bounds
        image_transform = src.transform
    gdf = gpd.read_file(file_shp)
    yolo_annotations = []
    for idx, row in gdf.iterrows():
        polygon = row['geometry']
        class_id = int(row['Id']) -1 
        print(class_id)
        
        # Convert polygon to YOLO format (bounding box)
        center_x, center_y, bbox_width, bbox_height = polygon_to_yolo(polygon, image_width, image_height)
        
        # Create YOLO annotation line
        yolo_line = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"
        yolo_annotations.append(yolo_line)

    output_txt_path = f'./labels/train/image{i}.txt'
    with open(output_txt_path, 'w') as f:
        for annotation in yolo_annotations:
            f.write(f"{annotation}\n")