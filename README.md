# Introduction

This repository presents a method and a code to detect isolated trees from panchromatic images obtained through the satellite Pleiade.
The notion of isolated tree can be defined as a single tree or a small group of trees that stands apart from larger forested areas or continuous vegetation cover. These trees are often found in agricultural landscapes, urban areas or open fields.
Those trees are indeed, by their spatial particularity, play a very important role and has a huge contribution to ecosystems service while being a sensitive indicator to a majority of biological variations.
The worflow provided an active learning strategy using the prediction probability-base selection which used the CNN YoloV11.

In our study site in Berambadi, an agricultural watershed in southern India (State of Karnataka), several Very-High Spatial Panchromatic Tri-Stereo images (50cm of spatial resolution) have been captured those last years on this site. 
Those images have been divided into a two groups : the first group has just been injected into a large pool of unlabeled images and the second one, has been annoted and then injected into a small pool of labeled image for the training and the testing dataset.

This method achieved to obtain a precision of 84,6%, a recall of 84.3% and a MAP50 of 85,7%.

## Installation

    pip install -r requirements.txt

## Description

The structure of the data follows this architecture are 

    dataset/
      |---images/
      |  |---train/
      |  |   |---image1.png
      |  |   |---image2.png
      |  |
      |  |---test/
      |      |---image1.png
      |      |---image2.png     
      |
      |---labels/
      |   |---train/
      |   |   |---image1.png
      |   |   |---image2.png
      |   |
      |   |---test/
      |       |---image1.png
      |       |---image2.png  
      |   
      |
      |---unlabel/
          |---image1.png
          |---image2.png


## Using your own data

To train the model on our Deep Active Learning with your own data, if you didn't preprocess our data into the Yolo format and with the correct number of band, you might need to use the preprocessing function  `pipeline_class.py`.
To use the function  `pipeline_class.py`, you will need to have your annotated data in shapefile format '.shp' and also provide a grid in a shapefile format '.shp' to divide the whole image.
The outputs of the `pipeline_class.py` will give you the train and test images and labels and the unlabel dataset needed to launch the active-learning code `SS_yolo_active_learning.py`

* All the images should have the same size (eg: 320x320, 640x640, etc...).

* The code is currently designed for three-band imagery. The input data supported by the algorithm should be in format '.tif' or 'geotiff' and to handle other formats, you would need to modify the functions in `pipeline_class.py`.

* Create the structure accordingly to the structure above with all of the output files obtained with `pipeline_class.py` and also add the yaml.
## Output

You have in the folder Results, some examples of tree detection images predicted and their probability of distribution.


