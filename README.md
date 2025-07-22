# Introduction

This repository presents a method and a code to detect isolated trees from panchromatic images obtained through the satellite Pleiade.
The notion of isolated tree can be defined as a single tree or a small group of trees that stands apart from larger forested areas or continuous vegetation cover. These trees are often found in agricultural landscapes, urban areas or open fields.
Those trees are indeed, by their spatial particularity, play a very important role and has a huge contribution to ecosystems service while being a sensitive indicator to a majority of biological variations.
The worflow provided an active learning strategy using the prediction probability-base selection which used the CNN YoloV11.

In our study site in Berambadi, an agricultural watershed in southern India (State of Karnataka), several Very-High Spatial Panchromatic Tri-Stereo images (50cm of spatial resolution) have been captured those last years on this site. 
Those images have been divided into a two groups : the first group has just been injected into a large pool of unlabeled images and the second one, has been annoted and then injected into a small pool of labeled image for the training and the testing dataset.

This method achieved to obtained a precision of 87,4%, a recall of 80.1% and a MAP50 of 87,9%.

## Installation

`pip install -r requirements.txt`

## Description

The first part consists from this three different inputs (Satellite image, a grid for dividing the satellite image into small images and a batch of trees) to create the following file structure :


![Image](https://github.com/user-attachments/assets/d815961a-3ceb-4c26-ba5d-ccd96e33d16b)



New_datas allow us just to store the images from the unlabeled dataset and in the txt file the score of mAP on the testset at the end of each iteration, the number and the name of the images which overcome the conditions of the detection before being injected in the trainset. 
Thus, the function SS_yolo.py and SS_yolo_function.py allow to train this active learning methpod with YOLOv11.

## Output

You have in the folder Results, some examples of tree detection images predicted and their probability of distribution.


