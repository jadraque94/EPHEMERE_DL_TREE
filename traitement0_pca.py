import numpy as np
import cv2
import rasterio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load the 4-band TIFF image
for i in range(1,143):
    input_file = f'./sub_yolo_image_640/image{i}.tif'



    with rasterio.open(input_file) as src:
        image = src.read()  # shape will be (bands, height, width)
        profile = src.profile  # save profile for output if needed

# im = np.round(image/image.max(),5)
# print(len(np.unique(im)))

# image_f = np.transpose(im)
# print(image_f.shape)
# plt.imshow(image_f,cmap='grey') 
# plt.show()

    bands, height, width = image.shape
    image_2d = image.reshape(bands, height * width).T  # shape is (num_pixels, num_bands)

    # Step 3: Apply PCA to reduce the 4 bands to the desired number of components (e.g., 3)
    n_components = 3  # Adjust based on the number of components you want to keep
    pca = PCA(n_components=n_components)
    image_pca = pca.fit_transform(image_2d)  # shape is now (num_pixels, n_components)

    # Step 4: Reshape the PCA result to display it as an image
    # Reshape back to (height, width, n_components) format
    image_pca_reshaped = image_pca.T.reshape(n_components, height, width)

    # Optional: Normalize for visualization

    var_cumu = np.cumsum(pca.explained_variance_ratio_)*100
    #print('la variance cumulé est :' , var_cumu)
    print(pca.explained_variance_ratio_)

    # Normalizing to range 0-255 for display purposes
    image_pca_normalized = (image_pca_reshaped - image_pca_reshaped.min()) / \
                        (image_pca_reshaped.max() - image_pca_reshaped.min()) * 255
    image_pca_normalized = image_pca_normalized.astype(np.uint8)

    #print(image_pca_normalized.shape)

    im = np.transpose(image_pca_normalized, (1, 2, 0))
    # Step 5: Display the PCA components as an image
    # Here we assume the first 3 principal components can represent RGB
    # plt.imshow(np.transpose(image_pca_normalized, (1, 2, 0)), cmap='gray')  # Transpose to (height, width, channels)
    # plt.axis('off')
    # plt.title(f"PCA Image with {n_components} Components")
    # plt.show()

    im = image_pca_normalized.T
    print(im.shape)


    imaget = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)

    # plt.figure()
    # plt.imshow(imaget, cmap='gray')  # Transpose to (height, width, channels)
    # plt.show()

    im = cv2.flip(imaget,1)


    output_file = f'./image_pca_640/image{i}.png'
    plt.imsave(output_file, im)
