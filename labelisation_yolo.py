import geopandas as gpd
import matplotlib.pyplot as plt


path = 'C:/Users/rahim/Deeplearning_oct_2024/CHADI_DeepLearning_Tree/Traning_coconut_palm.shp'


# Load the shapefile
shape = gpd.read_file(path)

# Plot the shapefile geometries
shape.plot()
print(shape)


# we will extract the boundind box and after convert to yolo format
for idx, row in shape.iterrows():
    minx, miny, maxx, maxy = row.geometry.bounds
    print(f"Bounding Box for feature {idx}: {minx}, {miny}, {maxx}, {maxy}")

def convert_to_yolo_format(minx, miny, maxx, maxy, img_width, img_height, class_id=0):
    # Calculate center coordinates, width, and height of the bounding box
    x_center = (minx + maxx) / 2.0 / img_width
    y_center = (miny + maxy) / 2.0 / img_height
    width = (maxx - minx) / img_width
    height = (maxy - miny) / img_height

    # YOLO format string
    return f"{class_id} {x_center} {y_center} {width} {height}"

# Example usage

img_width, img_height = 1024, 1024  # Example image dimensions

minx, miny, maxx, maxy = (10, 20, 200, 300)  # Example bounding box

yolo_annotation = convert_to_yolo_format(minx, miny, maxx, maxy, img_width, img_height)

print(yolo_annotation)  # Output: 0 0.102 0.156 0.186 0.273
