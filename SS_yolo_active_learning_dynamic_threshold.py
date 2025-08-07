import os
import re
from typing import List
import numpy as np
import csv
import shutil
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import keyboard
import time


def extract_indexed_number(filename: str) -> int:
    # Extrait le premier nombre trouvé dans le nom de fichier (utile pour trier)
    match = re.search(r"(\d+)(?=\.)", filename)
    return int(match.group(1)) if match else float('inf')


class DataManager:
    # Gère la préparation et la configuration du dataset
    def __init__(self, source_dir: Path, target_dir: Path):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.images_train = self.target_dir / 'images' / 'train'
        self.images_val = self.target_dir / 'images' / 'test'
        self.labels_train = self.target_dir / 'labels' / 'train'
        self.labels_val = self.target_dir / 'labels' / 'test'
        self.folder_image = self.target_dir / 'new_datas' / 'images'
        self.folder_label = self.target_dir / 'new_datas' / 'labels'

        self.unlabeled = self.target_dir / 'unlabel'
        self.data_config = self.target_dir / 'data.yaml'

    def prepare(self) -> None:
        # Copie récursive des fichiers du dossier source vers le dossier cible et creation du dossier folder où on place les speudos labels durant la prediction
        self.target_dir.mkdir(parents = True, exist_ok=True)
        self.folder_image.mkdir(parents= True, exist_ok = True)
        self.folder_label.mkdir(parents= True, exist_ok = True)

        for item in tqdm(self.source_dir.rglob('*'), desc='Copying data'):
            dest = self.target_dir / item.relative_to(self.source_dir)
            if item.is_dir():
                dest.mkdir(parents = True, exist_ok=True)
            else:
                shutil.copy2(item, dest)
        
        self._write_config()

    def transfer_data_folder(self):
        #permet de déplacer les pseudos_labels et leur pseudos-images associés dans le jeu de données d'entrainement à la fin de chaque itération
        files_to_transfer = os.listdir(self.folder_label)
        if len(files_to_transfer) != 0:
            for item_image in tqdm(self.folder_image.glob('*.png')):
                destination_im = os.path.join(self.images_train , item_image.name)
                shutil.copy2(item_image, destination_im)
                os.remove(item_image)

            for item_label in tqdm(self.folder_label.glob('*')):
                destination_lab = os.path.join(self.labels_train, item_label.name)
                shutil.copy2(item_label,destination_lab)
                os.remove(item_label)
        else :
            pass


    def _write_config(self) -> None:
        # Écrit le fichier data.yaml attendu par YOLO
        content = {
            'train': str(self.images_train),
            'val': str(self.images_val),
            'nc': 1,
            'names': ['tree']
        }
        with open(self.data_config, 'w') as f:
            for key, value in content.items():
                f.write(f"{key}: {value}\n")


class MappingHandler:
    # Gère la table de correspondance entre images non annotées et pseudo-labels
    def __init__(self, path: Path):
        self.path = path
        self.processed = self._load()

    def _load(self) -> dict:
        # Charge le mapping existant (si présent)
        if not self.path.exists():
            return {}
        with open(self.path, newline='') as f:
            reader = csv.DictReader(f)
            return {row['unlabel_image']: row for row in reader}

    def update(self, unlabel_image: str, new_image: str, iteration: int, n_boxes: int, thresh: float, needed: int) -> None:
        # Ajoute une nouvelle entrée au mapping et met à jour en mémoire
        entry = {
            'unlabel_image': unlabel_image,
            'new_image': new_image,
            'iteration': iteration,
            'n_boxes': n_boxes,
            'thresh': thresh,
            'needed': needed
        }
        new_index = 0 if not self.path.exists() else sum(1 for _ in open(self.path))
        row = {'index': new_index, **entry}
        exists = self.path.exists()
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['index'] + list(entry))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        self.processed[unlabel_image] = row


class ModelHandler:
    # Charge, entraîne, sauvegarde et évalue le modèle YOLO
    def __init__(self, initial_weights: Path, save_folder: Path):
        self.initial = initial_weights
        self.save_folder = save_folder
        self.model = None

    def load(self, iteration: int) -> None:
        # Charge le modèle initial ou la version obtenue à l'itération précédente
        if iteration == 0:
            self.model = YOLO(str(self.initial))
        else:
            path = self.save_folder / f'tree_detect_it{iteration-1}' / 'weights' / 'best.pt'
            self.model = YOLO(str(path))

    def train(self, data_config: Path, epochs: int, project: Path, name: str, augment: dict) -> None:
        # Entraîne le modèle avec les paramètres spécifiés
        self.model.train(
            data=str(data_config),
            epochs=epochs,
            freeze=False,
            pretrained=True,
            weight_decay=0.0097,
            imgsz=160,
            project=str(project),
            batch=4,
            lr0=0.0001,
            name=name,
            optimizer='AdamW',
            **augment
        )

    def save(self, iteration: int) -> None:
        # Sauvegarde les poids du modèle entraîné
        target = self.save_folder / f'tree_detect_it{iteration}' / 'weights' / 'best.pt'
        target.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(target))

    def evaluate(self, data_config: Path, project: Path ) -> None:
        # Évalue le modèle sur le jeu de validation et affiche le mAP50
        metrics = self.model.val(
            data=str(data_config),
            split='val',
            project=str(project),
            save_json=True
        )

        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f'precision: {metrics.box.mp:.4f}')
        print(f'recall: {metrics.box.mr:.4f}')

        entry = {
        'mAP50': metrics.box.map50,
        'precision': metrics.box.mp,
        'recall': metrics.box.mr,
        'mAP50-95' : metrics.box.map
        }
        new_index = 0 if not self.csv_path.exists() else sum(1 for _ in open(self.csv_path))
        row = {'index': new_index, **entry}
        exists = self.csv_path.exists()
        with open( self.csv_path , 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['index'] + list(entry))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        
        

class SemiSupervisedTrainer:
    # Orchestration du flux d'entraînement et de pseudo-labelisation
    def __init__(
        self,
        initial_weights: Path,
        epochs_per_iter: int,
        thresh_init_min : float,
        thresh_init_max : float,
        max_boxes : int,
        min_boxes: int,
        data_dir: Path,
        save_folder: Path,
        dataset_dir: Path
    ):
        self.save_folder = save_folder
        self.data_manager = DataManager(data_dir, dataset_dir)
        self.data_manager.prepare()
        self.mapping = MappingHandler(dataset_dir / 'mapping.csv')
        self.csv_path = MappingHandler(dataset_dir / 'validation.csv')
        self.model_handler = ModelHandler(initial_weights, save_folder)
        self.thresh_init_min = thresh_init_min
        self.thresh_init_max = thresh_init_max
        self.epochs_per_iter = epochs_per_iter
        self.max_boxes = max_boxes
        self.min_boxes = min_boxes

    def _count_initial_labels(self) -> int:
        # Compte le nombre d'images initiales annotées
        patterns = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        files = [p for p in (self.data_manager.images_train).iterdir() if p.suffix.lower() in patterns]
        count = len(files)
        print(f"Initial labeled images: {count}")
        return count

    def run(self, total_iterations: int, batch_size: int = 8) -> None:
        # Lance les étapes de préparation, entraînement et pseudo-labellisation
        start_time = time.time() # Siva
        lst_prob = [] # Siva
        for iteration in range(total_iterations):
            # Suppression des caches pour forcer la réactualisation
            cache_paths = [self.data_manager.labels_train.with_suffix('.cache'),
                           self.data_manager.labels_val.with_suffix('.cache')]
            for path in cache_paths:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
            print(f"=== Iteration {iteration} start ===")
            input()
            self.model_handler.load(iteration)
            augment = {'flipud': 0.5, 'mosaic': 0.9, 'mixup': 0.2, 'hsv_h' : 0.7, 'hsv_s' :0.7 , 'bgr' : 0.6 } if iteration > 1 else {}
            epochs = self._compute_epochs(iteration, total_iterations)
            self.model_handler.train(
                data_config = self.data_manager.data_config,
                epochs = epochs,
                project = self.save_folder,
                name = f'tree_detect_it{iteration}',
                augment = augment
            )
            self.model_handler.save(iteration)

            thresh_low, thresh_high = self.update_thresh(lst_prob, iteration) # Siva 
            print(f"Iteration: {iteration} -> Thresh_low = {thresh_low:.3f}, Thresh_high = {thresh_high:.3f}") # Siva
            self.thresholds_csv(iteration, thresh_low, thresh_high) # Siva
            preci_max, recal_max, mAP50_max = self.max_precision_recall_mAP50(iteration, Path(self.save_folder)/f'tree_detect_it{iteration}'/'results.csv') # Siva
            print(f"Iteration: {iteration} -> preci_max = {preci_max}, recal_max = {recal_max}, mAP50_max = {mAP50_max}") # Siva

            lst_prob_ite = self._process_unlabeled(total_iterations, iteration, batch_size, lst_prob, thresh_low, thresh_high) # Siva
            lst_prob.append(lst_prob_ite) # Siva
            print("annotate images and transfer in the right folder, then precess 'end' ")
            self.data_manager.transfer_data_folder()
            input()

        #self.model_handler.evaluate(self.data_manager.data_config, self.save_folder)  a corriger
        print(f"Final model at {self.save_folder}/tree_detect_it{total_iterations-1}/weights/best.pt")
        print(f"New dataset at {self.data_manager.target_dir}, added: {len(self.mapping.processed)} images")
        print(f"Mapping file: {self.mapping.path}")

        self.thresholds_graph(Path(self.save_folder/'thresholds.csv')) # Siva
        self.max_precision_recall_mAP50_graph(Path(self.save_folder)/'max_precision_recall_mAP50.csv') # Siva

        end_time = time.time() # Siva
        print(f"Execution took {self.execution_time(total_iterations, start_time, end_time):.2f} seconds") # Siva


    def _compute_epochs(self, iteration: int, total: int) -> int:
        # Ajuste le nombre d'époques pour la première et dernière itération
        if 0 < iteration < (total - 1):
        # if iteration < (total-1):
            return self.epochs_per_iter
        return round(self.epochs_per_iter * 1)
    
    def visualize(self, label, image_path):
        image = cv2.imread(image_path)
        print(image.shape)
        image_height, image_width = image.shape[:2]
        annotations = []

        with open(label, 'r') as file :
            for line in file:
        # Parse the YOLO annotation line
                class_id, center_x, center_y, bbox_width, bbox_height = map(float, line.split())
                annotations.append((class_id, center_x, center_y, bbox_width, bbox_height))

        fig, ax = plt.subplots(figsize=(15, 15))
        test = image / np.max(image)

        brightness_factor = 1.4 # You can adjust this value for more/less brightness
        bright_image = test * brightness_factor

        # Step 3: Clip the pixel values to avoid overflow (ensure values are within [0, 255] for 8-bit images)
        bright_image = np.clip(bright_image, 0, 255)

        ax.imshow(bright_image, cmap='gray')

        # Iterate over annotations and draw rectangles
        for annotation in annotations:
            class_id, center_x, center_y, bbox_width, bbox_height = annotation
            
            # Convert YOLO format to pixel coordinates
            x_min, y_min, x_max, y_max = self.yolo_to_pixel(center_x, center_y, bbox_width, bbox_height, image_width, image_height)
            
            # Create a rectangle patch and add it to the plot
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # Show the final image with bounding boxes
        plt.title("YOLO Annotations Overlaid on png Image")
        plt.show()

    @staticmethod
    def yolo_to_pixel(center_x, center_y, bbox_width, bbox_height, img_width, img_height):
        x_min = int((center_x - bbox_width / 2) * img_width)
        y_min = int((center_y - bbox_height / 2) * img_height)
        x_max = int((center_x + bbox_width / 2) * img_width)
        y_max = int((center_y + bbox_height / 2) * img_height)
        return x_min, y_min, x_max, y_max
    
    def update_thresh(self, lst_prob, iteration) -> tuple: # Siva
        # Met a jour thresh_low et thresh_high
        if iteration == 0:
            return (self.thresh_init_min, self.thresh_init_max)
        thresh_low: float = np.quantile(lst_prob[iteration-1], 0.25)
        thresh_high: float = np.quantile(lst_prob[iteration-1], 0.75)
        return thresh_low, thresh_high
    
    def thresholds_csv(self, iteration, thresh_low, thresh_high) -> None: # Siva
        # Enregriste thresh_low et thresh_high dans un ficher CSV
        csv_path = os.path.join(self.save_folder, "thresholds.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["Iteration", "thresh_low", "thresh_high"])
            writer.writerow([iteration, round(thresh_low, 3), round(thresh_high, 3)])

    def max_precision_recall_mAP50(self, iteration: int, path_resultats_csv: Path) -> tuple: # Siva
        # Retourne et enregistre le max precision(B), recall(B), mAP50(B) dans un fichier CSV
        preci: float
        recal: float
        mAP50: float
        preci_max: float = 0.0
        recal_max: float = 0.0
        mAP50_max: float = 0.0

        with open(path_resultats_csv, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                try:
                    preci =  float(row['metrics/precision(B)'])
                    recal =  float(row['metrics/recall(B)'])
                    mAP50 =  float(row['metrics/mAP50(B)'])

                except ValueError:
                    continue

                if preci > preci_max : 
                    preci_max = preci
                if recal > recal_max : 
                    recal_max = recal
                if mAP50 > mAP50_max : 
                    mAP50_max = mAP50
        
        csv_path = os.path.join(self.save_folder, "max_precision_recall_mAP50.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["Iteration", "max_precision", "max_recall", "max_mAP50"])
            writer.writerow([iteration, round(preci_max, 3), round(recal_max, 3), round(mAP50_max, 3)])
        return preci_max, recal_max, mAP50_max
    
    def execution_time(self,total_iterations, start_time, end_time) -> float: # Siva
        # Retourne et enregistre le temps pris pour l'execution du code run dans un ficher CSV
        tmp_tot = end_time - start_time
        csv_path = os.path.join(self.save_folder, "execution_time.csv")
        write_header = not os.path.exists(csv_path)
        with open (csv_path, mode = 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["Total_iterations", "start_time(seconds)", "end_time(seconds)", "total_time(seconds)"])
            writer.writerow([total_iterations, round(start_time, 3), round(end_time, 3), round(tmp_tot, 3)])
        return tmp_tot

    def thresholds_graph(self, thresholds_csv: Path) -> None: # Siva
        # Affiche et enregistre la progression de thresh_low et thresh_high
        df = pd.read_csv(thresholds_csv)
        plt.plot(df['Iteration'], df['thresh_low'], label = 'thresh_low')
        plt.plot(df['Iteration'], df['thresh_high'], label = 'thresh_high')
        
        plt.title('Dynamic thresholds over iterations')
        plt.xlabel('Iterations')
        plt.legend()
        plt.savefig(Path(self.save_folder)/'Dynamic_thresholds_plot.png')
        plt.show()
        plt.close('all')

    def max_precision_recall_mAP50_graph(self, max_precision_recall_mAP50_csv: Path) -> None: # Siva
        # Affiche et enregistre la progression de maximum de precision, recall, mAP50
        df = pd.read_csv(max_precision_recall_mAP50_csv)
        plt.plot(df['Iteration'], df['max_precision'], label = 'max_precision')
        plt.plot(df['Iteration'], df['max_recall'], label = 'max_recall')
        plt.plot(df['Iteration'], df['max_mAP50'], label = 'max_mAP50')

        plt.title('Maximum precision, recall, mAP50 over iteration')
        plt.xlabel('Iterations')
        plt.legend()
        plt.savefig(Path(self.save_folder)/'Maximum_precision_recall_mAP50_plot.png')
        plt.show()
        plt.close('all')


    def _process_unlabeled(self, total_iterations, iteration: int, batch_size: int, lst_prob, thresh_low, thresh_high) -> List[float]:
        # Gère la pseudo-labellisation des images non annotées
        files = sorted(self.data_manager.unlabeled.glob('*'), key=lambda p: extract_indexed_number(p.name))
        print(f"Unlabeled images: {len(files)}")
        added = 0
        self.next_index = self._count_initial_labels()
        lst_prob = [] # Siva

        for i in range(0, len(files), batch_size): # Siva
            if iteration == total_iterations: # Siva
                break # Siva
            batch = files[i:i + batch_size]
            results = self.model_handler.model.predict(
                source=[str(p) for p in batch], task='segment', verbose=True, iou = 0.4, conf = 0.2, retina_masks =True)
            
            for result in results: # Siva
                confs = result.boxes.conf.tolist()
                lst_prob.extend(confs) # Siva


        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            results = self.model_handler.model.predict(
                source=[str(p) for p in batch], task='segment', verbose=True, iou = 0.4, conf = 0.2, retina_masks =True)
            for result, img_path in zip(results, batch):
                print(img_path)
                name = img_path.name
                if name in self.mapping.processed:
                    continue
                confs = result.boxes.conf.tolist()
                needed = self.min_boxes + (1 if iteration >= 1 else 0) # Siva
                if all(c < thresh_low for c in confs) and len(confs) >= needed: # Siva
                    print(confs,'le numero est :', self.next_index)
                    new_name = f"image{self.next_index}.png"
                    label_file, image_file = self._save_pseudo_label(result, img_path, new_name)
                    #self.visualize(label_file,image_file)
                    input()
                    self.next_index += 1
                    self.mapping.update(name, new_name, iteration, len(confs), thresh_low, needed)
                    added += 1

                elif all(c > thresh_high for c in confs) and len(confs) >= needed : # Siva
                    print(confs)
                    new_name = f"image{self.next_index}.png"
                    _, _ = self._save_pseudo_label(result, img_path, new_name)
                    self.next_index += 1
                    self.mapping.update(name, new_name, iteration, len(confs), thresh_high, needed)
                    added += 1
                
                # elif len(confs) >= self.max_boxes and iteration >= 2 : #condition nouvelle pour verifier
                #     print(confs)
                #     new_name = f"image{self.next_index}.png"
                #     label_2 , image_2 = self._save_pseudo_label(result, img_path, new_name)
                #     #self.visualize(label_2,image_2)
                #     input()
                #     self.next_index += 1
                #     self.mapping.update(name, new_name, iteration, len(confs), thresh_high, needed)
                #     added += 1

        print(f"Iteration {iteration} added {added} images")
        return lst_prob


    def _save_pseudo_label(self, result, img_path: Path, new_name: str) -> None:
        # Sauvegarde le fichier de pseudo-label au format YOLO et l'image correspondante
        w, h = result.orig_shape
        result.show()
        print(f"[DEBUG] Number of predicted boxes: {len(result.boxes)}") # debug
        print(f"[DEBUG] Boxes (xyxy): {result.boxes.xyxy.tolist()}") # debug
        lines = []
        for x1, y1, x2, y2 in result.boxes.xyxy.tolist():
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            wd = (x2 - x1) / w
            ht = (y2 - y1) / h
            lines.append(f"0 {xc} {yc} {wd} {ht}")
        label_file = self.data_manager.folder_label / new_name.replace('.png', '.txt')
        image_file = self.data_manager.folder_image / new_name
        with open(label_file, 'w') as f:
            f.write("\n".join(lines))
        shutil.copy2(img_path, image_file)
        return  label_file, image_file


if __name__ == '__main__':
    base = Path('C:/Users/Sivakumaran/Documents/Stages/Institut francais de Pondichery 2025/EPHEMERE_DL_TREE')
    # base = Path('./')
    trainer = SemiSupervisedTrainer(
        initial_weights = './weights_test_5_epo30_ite10/yolo11l.pt',  # Initial weights
        epochs_per_iter = 30, 
        thresh_init_min= 0.25,                         # Confidence threshold for pseudo-labeling
        thresh_init_max= 0.75,
        max_boxes = 8,
        min_boxes = 1,                                               # Minimum number of boxes to accept pseudo-labeling
        data_dir = base / 'dataset/dataset_Yolo_AL_160px',                    # Directory with initial labeled data
        save_folder = Path('run_v11l_test_5_epo30_ite10'),                          # Directory to save the model and results
        dataset_dir = base / 'run_v11l_test_5_epo30_ite10' / 'new_dataset'          # Directory for the new dataset
    )
    trainer.run(total_iterations=10, batch_size=4)

