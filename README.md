# Introduction

This github presents a semi-supervised methodology combined with YOLOV8 in order to detect isolated tree from satellite images.

Thus, we can also define the notion of isolated tree which is a single tree or a small group of trees that stands apart from larger forested areas or continuous vegetation cover. Generally, these trees are often found in agricultural landscapes, urban areas, pastures, or open fields, where they are not connected to dense woodlands or forests. Those trees are indeed, by their spatial particularity, play a very important role and has a huge contribution to ecosystems service while being a sensitive indicator to a majority of biological variations.


## Installation

pip install -r requirements.txt

## Description

The first part consists, from three different inputs (Satellite image, a grid for dividing the satellite image in a small image a polygon and batch of trees) and create the following file structure 



dataset/
   ├── images/
   │   ├── train/
   │   │   ├── image1.jpg
   │   │   └── image2.jpg
   │   └── test/
   │       ├── image101.jpg
   │       └── image102.jpg
   │
   ├── labels/
   │   ├── train/
   │   │   ├── image1.txt
   │   │   └── image2.txt
   │   └── test/
   │       ├── image101.txt
   │       └── image102.txt
   │
   ├── new_datas/
   │   ├── train/
   │   └── test/
   │
   └── unlabel/
       ├── image_unlabeled1.jpg
       └── image_unlabeled2.jpg




Thus, 


