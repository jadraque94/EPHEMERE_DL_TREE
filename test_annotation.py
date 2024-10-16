from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import cv2

path  ="C:/Users/rahim/Deeplearning_oct_2024/CHADI_DeepLearning_Tree/Pleiades_14feb2023_pansharpen_imagette.tif"
image = rio.open(path)
w = image.width
h = image.height

with rio.open(path) as src:
    # Read the first band or all bands as needed
    red_band = src.read(1)  # Read band 1
    green_band = src.read(2)  # Read band 2
    blue_band = src.read(3)  # Read band 3
    rgb_image = np.dstack((red_band, green_band, blue_band))

# Step 2: Convert the numpy array (band1) into a Pillow image
test = int((255 * (rgb_image / np.max(rgb_image))))
pil_image = Image.fromarray(red_band)
plt.imshow(test)


def read_yolo_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Create an empty list to store the bounding box data
    data = []
    
    for line in lines:
        # Split each line into components and convert to float
        components = list(map(float, line.strip().split()))
        
        # Append the components to the data list
        data.append(components)
        
    # Convert the list to a NumPy array
    return np.array(data)

annotations = read_yolo_file('./first_yolo1.txt')

transformed_annotations = np.copy(annotations)
transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 

transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]


for ann in transformed_annotations:
    #print(ann)
    obj_cls, x0, y0, x1, y1 = ann
    print(x0, y0, x1, y1)

    cv2.rectangle(test, (x0,y0), (x1,y1), (0, 255, 0))
    # pil_image.text((x0, y0 - 10), int(obj_cls))

plt.imshow(rgb_image)