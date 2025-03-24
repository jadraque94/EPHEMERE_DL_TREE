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
import geopandas as gpd





class YOLOPreprocessor:


    def __init__(self, path_tif, path_tree, path_grid, name, output="./image_set/"):
        self.path_tif = path_tif
        self.path_tree = path_tree
        self.path_grid = path_grid
        self.output = Path(output)
        self.name = name

        self.path_unlabel_ima = Path(f"./image_unlabel_{name}/")
        self.path_label_box = Path(f"./box_label_{name}/")
        self.path_label_ima = Path(f"./image_label_{name}/")

        self.train_box = Path(f"./train_box_{name}/")
        self.test_box = Path(f"./test_box_{name}/")
        self.train_image = Path(f"./train_image_{name}/")
        self.test_image = Path(f"./test_image_{name}/")
        self.box_yolo_path = Path(f"./train_yolo_{name}/")
        self.output_concat_ortho = Path(f"./ortho_full/")

        # Création des dossiers si non existants
        for path in [self.output, self.path_unlabel_ima, self.path_label_box, self.path_label_ima,
                     self.train_box, self.test_box, self.train_image, self.test_image, 
                     self.box_yolo_path, self.output_concat_ortho]:
            path.mkdir(parents=True, exist_ok=True)


    def extract_images(self):
        grid = gpd.read_file(self.path_grid)

        with rasterio.open(self.path_tif) as src:
            for idx, row in grid.iterrows():
                geometry = [row.geometry]
                out_image, out_transform = mask(src, geometry, crop=True)

                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                output_path = self.output / f"images{idx}.tif"
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)

                print(f"Saved: {output_path}")

    def separate_grid_unlabel_label(self):
        shape_tree = gpd.read_file(self.path_tree)
        print("Nombre d'arbres délinées :", shape_tree.shape)

        image_paths = sorted(glob.glob(f"{self.output}/*"), key = os.path.extract_number)
        number_label, number_unlabel = 0, 0

        for image in tqdm(image_paths):
            valid_polygons = []
            with rasterio.open(image) as src:
                xmin_r, ymin_r, xmax_r, ymax_r = src.bounds

                for _, row in shape_tree.iterrows():
                    geometry = row.geometry
                    xmin_p, ymin_p = np.array(geometry.exterior.coords).min(axis=0)
                    xmax_p, ymax_p = np.array(geometry.exterior.coords).max(axis=0)

                    if xmin_r <= xmin_p <= xmax_r and ymin_r <= ymin_p <= ymax_r:
                        valid_polygons.append(row)

            if valid_polygons:
                valid_gdf = gpd.GeoDataFrame(valid_polygons, crs=shape_tree.crs)
                valid_gdf.to_file(self.path_label_box / f'image{number_label}.shp')

                shutil.copy2(image, self.path_label_ima / f'image{number_label}.tif')
                number_label += 1
            else:
                shutil.copy2(image, self.path_unlabel_ima / f'image{number_unlabel}.tif')
                number_unlabel += 1

    def train_test_split(self, split_ratio=0.75):
        box_files = sorted(glob.glob(f"{self.path_label_box}/*"), key = self.extract_number)
        image_files = sorted(glob.glob(f"{self.path_label_ima}/*"), key = self.extract_number)

        split_idx = int(split_ratio * len(box_files))
        train_files_box, test_files_box = box_files[:split_idx], box_files[split_idx:]
        train_files_images, test_files_images = image_files[:split_idx], image_files[split_idx:]

        for file in train_files_box:
            shutil.copy2(file, self.train_box / Path(file).name)
        for file in test_files_box:
            shutil.copy2(file, self.test_box / Path(file).name)
        for file in train_files_images:
            shutil.copy2(file, self.train_image / Path(file).name)
        for file in test_files_images:
            shutil.copy2(file, self.test_image / Path(file).name)

        print('FIN DU SPLIT')

    @staticmethod
    def polygon_to_yolo(polygon, image_width, image_height, im_transform):
        minx, miny, maxx, maxy = polygon.bounds
        px_minx, px_miny = ~im_transform * (minx, miny)
        px_maxx, px_maxy = ~im_transform * (maxx, maxy)

        bbox_width = (px_maxx - px_minx) / image_width
        bbox_height = (px_miny - px_maxy) / image_height
        center_x = (px_minx + px_maxx) / 2 / image_width
        center_y = (px_miny + px_maxy) / 2 / image_height

        return center_x, center_y, bbox_width, bbox_height

    @staticmethod
    def extract_number(filename):
        match = re.search(r"(\d+)(?=\.\w+$)", filename)  # Finds digits before file extension
        return int(match.group(1)) if match else float('inf')  # Use 'inf' if no number found


    def convert_yolo(self):
        list_box = sorted(glob.glob(f"{self.train_box}/*.shp"), key= self.extract_number)
        list_image = sorted(glob.glob(f"{self.train_image}/*"), key=self.extract_number)

        assert len(list_box) == len(list_image)

        for t in range(len(list_image)):
            with rasterio.open(list_image[t]) as src:
                image_width, image_height = src.width, src.height
                image_transform = src.transform

            gdf = gpd.read_file(list_box[t])
            yolo_annotations = []

            for _, row in gdf.iterrows():
                polygon = row['geometry']
                class_id = int(row['id']) - 1
                center_x, center_y, bbox_width, bbox_height = self.polygon_to_yolo(
                    polygon, image_width, image_height, image_transform
                )
                yolo_annotations.append(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}")

            output_txt_path = self.box_yolo_path / f'image{t}.txt'
            with open(output_txt_path, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(f"{annotation}\n")

        return print('FIN DE LA CONVERSION YOLO')

    def orthoimage_to_png_opencv(self, input_image_tif, output_png):

        os.makedirs(output_png)
        list_image = sorted(glob.glob(input_image_tif), key=self.extract_number)

        for t in range(len(list_image)):

            with rasterio.open(list_image[t]) as src:
                band = src.read(1)  # Read the first (and only) band
            
                # Normalize the band to 0-255 (8-bit grayscale)
                band_normalized = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Save the normalized image as a PNG using OpenCV
            output_path = os.path.join(output_png,f'image{t}.png')
            cv2.imwrite(output_path, band_normalized)
            print(f"Saved PNG: {output_path}")

    


    def run_pipeline(self):
        print("Extraction des images...")
        self.extract_images()

        print("Séparation des images en label/unlabel...")
        self.separate_grid_unlabel_label()

        print("Séparation train/test...")
        self.train_test_split()

        print("Conversion des annotations en format YOLO...")
        self.convert_yolo()


        print("Pipeline terminé !")


# Utilisation de la classe
if __name__ == "__main__":


    #
    yolo_preprocessor = YOLOPreprocessor(
        path_tif="Pleiades_Vue1_2023/C1_orthoimage_forward.tif",
        path_tree="TreeSample_ImagePleiade14feb2023_Pansharpen.shp",
        path_grid="yolo_semi_janv_2025/grid_320_semi.shp",
        name = 'ortho_vue1'
         
    )
    yolo_preprocessor.run_pipeline()


    def orthoimage_to_png_opencv(input_image_tif, output_png):

        os.makedirs(output_png)
        list_image = sorted(glob.glob(input_image_tif), key= os.path.extract_number)

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
                print(np.unique(img_rgbnir))
                band_normalized = cv2.normalize(img_rgbnir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                print(band_normalized.shape, np.unique(band_normalized))

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
