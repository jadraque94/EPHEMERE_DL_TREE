from ultralytics import YOLO
import os
import shutil
import glob
import numpy as np
import re
#image_label_path = './dataset_SS/images/train/*'

def extract_number(filename):
    match = re.search(r"(\d+)(?=\.\w+$)", filename)  # Finds digits before file extension
    return int(match.group(1)) if match else float('inf')  # Use 'inf' if no number found


def index_image(path_image_output, path_image_input, path_txt_output, path_txt_input):

    sort_path_output = sorted(glob.glob(path_image_output), key = extract_number)
    sort_path_input = sorted(glob.glob(path_image_input), key = extract_number)
    sort_txt_output = sorted(glob.glob(path_txt_output), key = extract_number)
    sort_txt_input = sorted(glob.glob(path_txt_input), key = extract_number)

    ### length of image_input and txt_input are the same because there are associated
    assert len(sort_path_input) == len(sort_txt_input)

    ### we take the length of the output folder 

    length_out = len(sort_path_output)

    for j in range(len(sort_txt_input)): #the length of image folder and the labels txt file associated in the text folder is the same
        
        image_end = sort_path_input[j]
        text_end = sort_txt_input[j]
        
        #print(image_end, text_end , '/n' )

        new_image_filename = os.path.join(os.path.dirname(path_image_output),f'image{length_out}.png')
        new_text_filename = os.path.join(os.path.dirname(path_txt_output),f'image{length_out}.txt')
        length_out +=1 

        #print('le nouveau est :', new_image_filename , new_text_filename )
        shutil.move( image_end , new_image_filename) #we will copy every image to the ne folder for train our semisupervised model
        shutil.move( text_end, new_text_filename )

    print(path_image_output)


#index_image(path_image_output = image_label_path, path_image_input = './dataset_SS/new_datas/train_image/*', path_txt_output = './dataset_SS/labels/train/*', path_txt_input ='./dataset_SS/new_datas/train_box/*' )
