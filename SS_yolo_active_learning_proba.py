import os
import re
import numpy as np
import csv
import time
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import keyboard 

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

    def update(self, unlabel_image: str, new_image: str, iteration: int, n_boxes: int, thresh: float, needed: int, time_end : float) -> None:
        # Ajoute une nouvelle entrée au mapping et met à jour en mémoire
        entry = {
            'unlabel_image': unlabel_image,
            'new_image': new_image,
            'iteration': iteration,
            'n_boxes': n_boxes,
            'thresh': thresh,
            'needed': needed,
            'time-end' :time_end,
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
            imgsz=320,
            project=str(project),
            batch=16,
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
        thresh_init_high : float,
        thresh_init_low : float,
        top_value : float,
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
        self.top_value = top_value
        self.thresh_init_high  = thresh_init_high
        self.thresh_init_low  = thresh_init_low
        self.model_handler = ModelHandler(initial_weights, save_folder)
        self.epochs_per_iter = epochs_per_iter
        self.min_boxes = min_boxes

    def _count_initial_labels(self) -> int:
        # Compte le nombre d'images initiales annotées
        patterns = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        files = [p for p in (self.data_manager.images_train).iterdir() if p.suffix.lower() in patterns]
        count = len(files)
        print(f"Initial labeled images: {count}")
        return count

    def run(self, total_iterations: int, batch_size: int = 8) -> None:
        lst_proba = []
        # Lance les étapes de préparation, entraînement et pseudo-labellisation
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
            #input()
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
            # thresh_low, thresh_high = self.update_thresh(lst_proba, iteration)
            # liste = self._process_unlabeled(iteration, batch_size, thresh_low, thresh_high)
            self._process_prediction_proba_based(iteration, batch_size)
            #lst_proba.append(liste)
            #print('le seuil haut est :', thresh_high ,'et le seuil bas est :', thresh_low)
            print("annotate images and transfer in the right folder, then precess 'end' ")
            self.data_manager.transfer_data_folder()
            #input()

        self.model_handler.evaluate(self.data_manager.data_config, self.save_folder)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"The result is: {execution_time}")



        print(f"Final model at {self.save_folder}/tree_detect_it{total_iterations-1}/weights/best.pt")
        print(f"New dataset at {self.data_manager.target_dir}, added: {len(self.mapping.processed)} images")
        print(f"Mapping file: {self.mapping.path}")


    def _compute_epochs(self, iteration: int, total: int) -> int:
        # Ajuste le nombre d'époques pour la première et dernière itération
        if 0 < iteration < (total - 1):
        # if iteration < (total-1):
            return self.epochs_per_iter
        return round(self.epochs_per_iter * 1) #modify multiply by 1
    
    def visualize(self, label, image):
        image = cv2.imread(image)
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


    def update_thresh(self, lst_prob, iteration):
        if iteration == 0:
            return self.thresh_init_low, self.thresh_init_high
        thresh_low = np.quantile(lst_prob[iteration-1],0.25)
        thresh_high = np.quantile(lst_prob[iteration-1],0.75)
        return thresh_low , thresh_high
    
    @staticmethod
    def _prob_score(liste) :
        lec = []
        for value in liste:
            if value != 0 :
                res = (1 - value) / value 
                print(res)
                lec.append(round(res,3))
        return lec


    def _process_prediction_proba_based(self, iteration: int, batch_size: int) -> None:
        liste_proba_score = []
        liste_result = []
        liste_name = []
        # Gère la pseudo-labellisation des images non annotées
        files = sorted(self.data_manager.unlabeled.glob('*'), key=lambda p: extract_indexed_number(p.name))
        print(f"Unlabeled images: {len(files)}")
        added = 0
        self.next_index = self._count_initial_labels()

        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            results = self.model_handler.model.predict(
                source=[str(p) for p in batch], task='segment', verbose=True, iou = 0.3, conf = 0.2, retina_masks =True)
            for result, img_path in zip(results, batch):
                name = img_path.name
                print('le result is :', img_path)
                if name in self.mapping.processed:
                    continue
                confs = result.boxes.conf.tolist()
                if len(confs) != 0:
                    proba_sc = self._prob_score(confs) #on calcule le proba_score avec la methode prediction probability-based
                    liste_name.append(img_path)
                    liste_result.append(result)
                    liste_proba_score.append(proba_sc)

        print( 'la longueur est de la liste de proba :' , len(liste_proba_score), 'et la longueur de la liste des results est :', len(liste_result))
        # 2. Sort the lists together
        combined_list = zip(liste_proba_score, liste_name, liste_result)

        sorted_combined_list = sorted(combined_list, key=lambda x: max(x[0]), reverse=True)
        sorted_liste_proba , reordered_liste_name, reordered_liste_result = zip(*sorted_combined_list)
        low_proba = sorted_liste_proba[:int(len(sorted_liste_proba) * self.top_value)] #on prend les 10% premières valeurs qui correspond aux image ayant des detections les plus incertaines
        low_name = reordered_liste_name[:int(len(reordered_liste_name) * self.top_value )]
        low_result = reordered_liste_result[:int(len(reordered_liste_result) * self.top_value)]

        print('la taille des listes est :', len(low_name), len(low_proba), len(low_result))

        # 3. Save and visualize the lowest inference
        for i in range(len(sorted_combined_list)):
            new_name = f"image{self.next_index}.png"
            print('les nouveaux paths sont :', low_result, low_name)
            label_file , image_file = self._save_pseudo_label(low_result[i], low_name[i], new_name)
            self.visualize(label_file,image_file)
            input()
            self.next_index = self._count_initial_labels()
            end_time = time.time() -start_time

            self.mapping.update(low_name[i].name, new_name, iteration, len(low_proba[i]), max(low_proba[i]), self.min_boxes, end_time)
            added += 1

        print(f"Iteration {iteration} added {added} images")

        


    def _process_unlabeled(self, iteration: int, batch_size: int , thresh_low, thresh_high) -> None:
        liste_conf = []
        # Gère la pseudo-labellisation des images non annotées
        files = sorted(self.data_manager.unlabeled.glob('*'), key=lambda p: extract_indexed_number(p.name))
        print(f"Unlabeled images: {len(files)}")
        added = 0
        self.next_index = self._count_initial_labels()

        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            results = self.model_handler.model.predict(
                source=[str(p) for p in batch], task='segment', verbose=True, iou = 0.3, conf = 0.2, retina_masks =True)
            for result, img_path in zip(results, batch):
                name = img_path.name
                if name in self.mapping.processed:
                    continue
                confs = result.boxes.conf.tolist()
                liste_conf.extend(np.round(confs,3))
                needed = self.min_boxes #+ (1 if iteration >= 2 else 0)
                # if all(c < thresh_low for c in confs) and len(confs) >= needed:
                #     print(confs,'le numero est :', self.next_index)
                #     new_name = f"image{self.next_index}.png"
                #     _, _ = self._save_pseudo_label(result, img_path, new_name)
                #     #self.visualize(label_file,image_file)
                #     self.next_index += 1
                #     self.mapping.update(name, new_name, iteration, len(confs), thresh_low, needed)
                #     added += 1

                if all(c > thresh_high for c in confs) and len(confs) >= needed :
                    print(confs)
                    new_name = f"image{self.next_index}.png"
                    _, _ = self._save_pseudo_label(result, img_path, new_name)
                    self.next_index += 1
                    end_time = time.time() - start_time
                    self.mapping.update(name, new_name, iteration, len(confs), thresh_high, needed, end_time)
                    added += 1
                
        print('la liste ajoutée est :',liste_conf)

        print(f"Iteration {iteration} added {added} images")
        return liste_conf


    def _save_pseudo_label(self, result, img_path: Path, new_name: str) -> None:
        # Sauvegarde le fichier de pseudo-label au format YOLO et l'image correspondante
        w, h = result.orig_shape
        lines = []
        for x1, y1, x2, y2 in result.boxes.xyxy.tolist():
            xc = (x1 + x2) / 2 / w
            ht = (y2 - y1) / h
            lines.append(f"0 {xc} {yc} {wd} {ht}")
        label_file = self.data_manager.labels_train / new_name.replace('.png', '.txt')
        image_file = self.data_manager.images_train / new_name
        with open(label_file, 'w') as f:
            f.write("\n".join(lines))
        shutil.copy2(img_path, image_file)
        return  label_file, image_file


if __name__ == '__main__':
    base = Path('C:/Users/rahim/Deeplearning_oct_2024/DeepLearning_EPHEMERE_Tree/Programme_git/EPHEMERE_DL_TREE')
    start_time = time.time()
    trainer = SemiSupervisedTrainer(
        initial_weights = './weight/yolo11m.pt',  # Initial weights
        epochs_per_iter = 4,                       # Number of epochs per iteration
        thresh_init_high = 0.75,
        thresh_init_low = 0.25,
        top_value = 0.05,
        min_boxes = 2,                                               # Minimum number of boxes to accept pseudo-labeling
        data_dir = base / './Pleiade_yolo_1300/',                    # Directory with initial labeled data
        save_folder = Path('run_v11m_test_1300_proba_test_1'),                          # Directory to save the model and results
        dataset_dir = base / 'run_v11m_test_1300_proba_test_1' / 'new_dataset'          # Directory for the new dataset
    )
    trainer.run(total_iterations=5, batch_size=32)
