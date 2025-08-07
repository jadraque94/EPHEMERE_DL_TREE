from ultralytics import YOLO
import pandas as pd
import os
import shutil
import glob
from tqdm import tqdm
import re
import numpy as np
import SS_YOLO_function 

# idea is to create a semi suprvised loop by using yolo at each iteration and integration not just the label but also the images when the probability of all of the bounding box
# of the image are above a certain threshold
def extract_number(filename):
    match = re.search(r"(\d+)(?=\.\w+$)", filename)  # Finds digits before file extension
    return int(match.group(1)) if match else float('inf')  # Use 'inf' if no number found

name_save = 'runs'

proba_accept = 0.7 #the mAP coefficient should be at 0.7
image_unlabel_path = './dataset/unlabel/*'
image_label_path = './dataset/images/train/*'
txt_label_path = './dataset/labels/train/*'


for i in range(8): #use preferably while loop while 

    if i == 0:
        model_weighted = YOLO("yolo8m.pt")  # Use the `.yaml` config file directly

    else :
        weight = f'./{name_save}/tree_detect{i-1}/weights/best.pt'
        model_weighted = YOLO(weight)  # Use the `.yaml` config file directly and reajusted the weight


    ### we evaluate the model
    model_weighted.train(data="trees.yaml", epochs=15, freeze = False, pretrained =True, weight_decay = 0.0097, imgsz=320, project= name_save,
                      batch = 16, lr0 =0.0001, name = f'tree_detect{i}', optimizer = 'AdamW')

    model_weighted.save(f'./{name_save}/tree_detect{i}/weights/best.pt')

    images_unlabel = sorted(glob.glob(image_unlabel_path), key=extract_number)
    images_label = sorted(glob.glob(image_label_path), key=extract_number)
    image_label_idx = len(images_label)


    for j in tqdm(range(len(images_unlabel))):
        
        results = model_weighted.predict(source = images_unlabel[j], task='segment') # predict box for each unlabeled images

        number_box = results[0].boxes 
        list_conf = number_box.conf.tolist() 
        coef_condition = all(proba_accept < x for x in list_conf)
        coef_condition_2 = all((proba_accept + 0.1) < x for x in list_conf) 
        print( list_conf )


        if i < 2 and coef_condition and len(number_box) > 0 : # Our condition : 70% > for all of the predicted boxes and a minimum of 4 boxes detected

            print('donc ici le nombre est ', i, list_conf)
            img_width, img_height = results[0].orig_shape  # Original image dimensions
            # Extract bounding boxes in [x_min, y_min, x_max, y_max] format
            bboxes_xyxy = results[0].boxes.xyxy  # Tensor with shape (N, 4)
            bboxes_yolo = []
            for bbox in bboxes_xyxy:
                x_min, y_min, x_max, y_max = bbox.tolist()
                
                # Convert to YOLO format
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                yolo_line = f"{0} {x_center} {y_center} {width} {height}"
                bboxes_yolo.append(yolo_line)

                ### FIRST, for observing purposes, we will store thoses images and labels in 
                output_txt = f'./dataset/new_datas/train_box/image{image_label_idx}.txt' # we will
                output_image = f'./dataset/new_datas/train_image/image{image_label_idx}.png' 

                image_label_idx += 1
                print(image_label_idx)
            
            with open(output_txt, 'w') as f:
                for y in bboxes_yolo :
                    f.write(f"{y}\n")

            shutil.move(images_unlabel[j], output_image)

            ## we will count wich images has been    
            with open(f'./dataset/new_datas/train_confidence_yolo_11_{i}.txt', 'a') as f:
                f.write(f"the unlabeled image number {j} has fullfilled the condition and {output_image} and its label are {output_txt} \n")
                

        elif i > 1 and (coef_condition_2) and len(number_box) > 1 : #on augmente pour obtenir 80%
            

            print('donc ici le nombre est ', list_conf )
            img_width, img_height = results[0].orig_shape  # Original image dimensions
            # Extract bounding boxes in [x_min, y_min, x_max, y_max] format
            bboxes_xyxy = results[0].boxes.xyxy  # Tensor with shape (N, 4)
            bboxes_yolo = []
            for bbox in bboxes_xyxy:
                x_min, y_min, x_max, y_max = bbox.tolist()
                
                # Convert to YOLO format
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                yolo_line = f"{0} {x_center} {y_center} {width} {height}"
                bboxes_yolo.append(yolo_line)

                ### FIRST, for observing purposes, we will store thoses images and labels in 
                output_txt = f'./dataset/new_datas/train_box/image{image_label_idx}.txt' # we will
                output_image = f'./dataset/new_datas/train_image/image{image_label_idx}.png' 

                image_label_idx += 1
                print(image_label_idx)
            
            with open(output_txt, 'w') as f:
                for y in bboxes_yolo :
                    f.write(f"{y}\n")

            shutil.move(images_unlabel[j], output_image)

            ## we will count wich images has been    
            with open(f'./dataset/new_datas/train_confidence_yolo_11_{i}.txt', 'a') as f:
                f.write(f"the unlabeled image number {j} has fullfilled the condition and {output_image} and its label are {output_txt} \n")
        

            
    # we will store in the folder new_datas the unlabel datas which has outpaced the condition of SS

    #we will inject the datas which has outpaced our conditions
    SS_YOLO_function.index_image(path_image_output = image_label_path, path_image_input = './dataset/new_datas/train_image/*', path_txt_output = './dataset/labels/train/*', path_txt_input ='./dataset/new_datas/train_box/*' )



    
    metrics_test = model_weighted.val(data = 'trees.yaml', split = 'val')
    #metrics_unlabel = model_weighted.predict(source = './dataset_SS/unlabel')


    prec_test = metrics_test.box.map50

    print(' la difference de de précision générales entre les données tests et non_labélisées', prec_test , '/n')
    with open(f'./dataset/new_datas/train_conf_yolo_11_{i}.txt', 'a') as f:
        f.write(f"the round {i} and the  {prec_test}\n")



                        



