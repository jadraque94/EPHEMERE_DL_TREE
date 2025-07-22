# Introduction

This repository presents a method and a code to detect isolated trees from panchromatic images obtained through the satellite Pleiade.
The notion of isolated tree can be defined as a single tree or a small group of trees that stands apart from larger forested areas or continuous vegetation cover. These trees are often found in agricultural landscapes, urban areas or open fields.
Those trees are indeed, by their spatial particularity, play a very important role and has a huge contribution to ecosystems service while being a sensitive indicator to a majority of biological variations.
The worflow provided an active learning strategy using the prediction probability-base selection which used the CNN YoloV11.

In our study site in Berambadi, an agricultural watershed in southern India (State of Karnataka), several Very-High Spatial Panchromatic Tri-Stereo images (50cm of spatial resolution) have been captured those last years on this site. 
Those images have been divided into a two groups : the first group has just been injected into a large pool of unlabeled images and the second one, has been annoted and then injected into a small pool of labeled image for the training and the testing dataset.

This method achieved to obtained a precision of 87,4%, a recall of 80.1% and a MAP50 of 87,9%.

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
      |---new-datas/
      |   |---train/
      |   |---test/
      |
      |---unlabel/
          |---image1.png
          |---image2.png


## Using your own data

To train on your own data, you will need to organize the data into the format expected by pipeline_class.py.

* All the images should have the same size (eg: 320x320, 640x640, etc...).

* The code is currently designed for three-band imagery. To handle less bands, you would need to modify the function in `YOLO_SS_class/pipeline_class.py`.

*Create the structure accordingly to the structure above with all of the output files obtained with `YOLO_SS_class/pipeline_class.py` and also add the yaml.
## Output

You have in the folder Results, some examples of tree detection images predicted and their probability of distribution.


