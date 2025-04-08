import os
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import box
import shutil
import glob
import geopandas as gpd


name = 'pansharp_vue1'


def index_image(path_image_output, path_image_input, path_txt_input, path_txt_output):

    sort_path_output = sorted(glob.glob(path_image_output), key=os.path.getctime)
    sort_path_input = sorted(glob.glob(path_image_input), key=os.path.getctime)
    sort_txt_input = sorted(glob.glob(path_txt_input), key=os.path.getctime)
    sort_txt_output = sorted(glob.glob(path_txt_output), key=os.path.getctime)


    n_start = len(sort_path_output)
    n_interm = len(sort_path_input)

    n_final = n_start + n_interm # we will add all of the different images of panshapren vue1 vue2 vue3 and ortho image
    print(n_final)
    compteur = 0
    for i in range(n_start, n_final): # the length of image folder and the labels txt file associated in the text folder is the same
        print(i, sort_path_input[compteur])
        image_end = os.path.join(os.path.dirname(sort_path_input[compteur]),f'image{compteur}.png')
        text_end = os.path.join(os.path.dirname(sort_txt_input[compteur]),f'image{compteur}.txt')
        
        print(image_end, text_end ,'/n', )

        new_image_filename = os.path.join(os.path.dirname(sort_path_output[compteur]),f'image{i}.png')
        new_text_filename = os.path.join(os.path.dirname(sort_txt_output[compteur]),f'image{i}.txt')

        print('le nouveau est :', new_image_filename , new_text_filename )
        shutil.copy2( image_end , new_image_filename) #we will copy the 
        shutil.copy2( text_end, new_text_filename )
        compteur += 1

    print(path_image_output)


def index_unlabel_image(path_image_output, path_image_input, number_ite):

    sort_path_output = sorted(glob.glob(path_image_output), key=os.path.getctime)
    sort_path_input = sorted(glob.glob(path_image_input), key=os.path.getctime)

    n_start = len(sort_path_output)
    n_interm = len(sort_path_input)

    n_final = n_start + n_interm # we will add all of the different images of panshapren vue1 vue2 vue3 and ortho image
    print(n_final)
    compteur = 0
    for i in range(n_start, n_final): # the length of image folder and the labels txt file associated in the text folder is the same
        #if compteur < number_ite :

        image_end = os.path.join(os.path.dirname(sort_path_input[compteur]),f'image{compteur}.png')
        
        new_image_filename = os.path.join(os.path.dirname(sort_path_output[compteur]),f'image{i}.png')

        print(i, image_end , new_image_filename)


        shutil.copy2( image_end , new_image_filename) #we will copy the 
        compteur += 1

    print(path_image_output)

### test
#index_unlabel_image(path_image_output = './dataset/unlabel/*', path_image_input = f'./{name}/png_unlabel_{name}/*', number_ite = 1001)# , path_txt_output = './dataset/labels/test/*', 
#            path_txt_input = f'./{name}/test_yolo_{name}/*' )

### train

index_image(path_image_output = './dataset/images/test/*', path_image_input = f'./{name}/png_test_{name}/*' , path_txt_output = './dataset/labels/test/*', 
             path_txt_input = f'./{name}/test_yolo_{name}/*' )


