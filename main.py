#!/usr/bin/env python
# -*- coding: utf-8 -*-

from load_image import Load_image
from yolo_v4 import Yolo_v4
from mask_beer import Mask_beer

# set camera define
camera = 0

if __name__ == "__main__":
    picture = Load_image()
    picture_yolo = Yolo_v4()
    filter_mask = Mask_beer()

    picture.capture_image(30, camera) # frame quantity, camera
    name_cup_chopp = picture_yolo.yolo_object_detection()
    name, height, width, volume = picture.data_load_info(name_cup_chopp)
    filter_mask.capture_image(name, height, width, volume, camera) # camera
    
    # Print the docstring of multiply function
    # print(picture.capture_image.__doc__)
    # print(picture.data_load_info.__doc__)
    # print(picture_yolo.yolo_object_detection.__doc__)
