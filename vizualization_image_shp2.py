import rasterio # type: ignore
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Step 1: Open the GeoTIFF image using rasterio
path = "./classe_1/image/image4.tif"
with rasterio.open(path) as src:
    # Read the first band of the image (assuming it's grayscale)
    red_band = src.read(1)  # Read band 1
    green_band = src.read(2)  # Read band 2
    blue_band = src.read(3)  # Read band 3
    rgb_image = np.dstack((red_band, green_band, blue_band))
    image_width = src.width
    image_height = src.height
    image_transform = src.transform

# Step 2: Read YOLO annotations from a file
yolo_txt_path = './labels/train/image4.txt'
annotations = []
with open(yolo_txt_path, 'r') as file:
    for line in file:
        # Parse the YOLO annotation line
        class_id, center_x, center_y, bbox_width, bbox_height = map(float, line.split())
        annotations.append((class_id, center_x, center_y, bbox_width, bbox_height))

# Step 3: Convert YOLO annotations back to pixel coordinates
def yolo_to_pixel(center_x, center_y, bbox_width, bbox_height, img_width, img_height):
    x_min = int((center_x - bbox_width / 2) * img_width)
    y_min = int((center_y - bbox_height / 2) * img_height)
    x_max = int((center_x + bbox_width / 2) * img_width)
    y_max = int((center_y + bbox_height / 2) * img_height)
    return x_min, y_min, x_max, y_max

# Step 4: Plot the image and draw rectangles for YOLO bounding boxes
fig, ax = plt.subplots(figsize=(15, 15))
test = rgb_image / np.max(rgb_image)

brightness_factor = 3  # You can adjust this value for more/less brightness
bright_image = test * brightness_factor

# Step 3: Clip the pixel values to avoid overflow (ensure values are within [0, 255] for 8-bit images)
bright_image = np.clip(bright_image, 0, 255)

ax.imshow(bright_image, cmap='gray')

# Iterate over annotations and draw rectangles
for annotation in annotations:
    class_id, center_x, center_y, bbox_width, bbox_height = annotation
    
    # Convert YOLO format to pixel coordinates
    x_min, y_min, x_max, y_max = yolo_to_pixel(center_x, center_y, bbox_width, bbox_height, image_width, image_height)
    
    # Create a rectangle patch and add it to the plot
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

# Show the final image with bounding boxes
plt.title("YOLO Annotations Overlaid on GeoTIFF Image")
plt.show()
