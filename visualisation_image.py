import cv2 # type: ignore
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# Step 1: Open the png 
path = "./run_v11m_test_1007_thresh_high/new_dataset/images/train/image148.png"
image = cv2.imread(path)
image_height, image_width = image.shape[:2]

# Step 2: Read YOLO annotations from a file
yolo_txt_path = './run_v11m_test_1007_thresh_high/new_dataset/labels/train/image148.txt'
annotations = []

with open(yolo_txt_path, 'r') as file :
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
test = image / np.max(image)

brightness_factor = 1  # You can adjust this value for more/less brightness
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
plt.title("YOLO Annotations Overlaid on png Image")
plt.show()