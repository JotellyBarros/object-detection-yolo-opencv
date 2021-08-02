#!/usr/bin/env python
# -*- coding: utf-8 -*-

from load_image import Load_image
from yolo_v4 import Yolo_v4

if __name__ == "__main__":
    picture = Load_image()
    picture_yolo = Yolo_v4()

    # picture.capture_image(30)
    name_cup_chopp = picture_yolo.yolo_object_detection()
    picture.data_load_info(name_cup_chopp)

    # Print the docstring of multiply function
    # print(picture.capture_image.__doc__)
    # print(picture.data_load_info.__doc__)
    # print(picture_yolo.yolo_object_detection.__doc__)
