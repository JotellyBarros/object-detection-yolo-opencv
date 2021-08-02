#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import csv
import numpy as np
import time

class Load_image:
    """Classe load image"""

    def __init__(self, name='picture'):
        self.name = None

    def capture_image(self, nFrames, camera):
        """Get capture of image"""

        file = "./picture_cup.png"
        cap = cv2.VideoCapture(camera)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        for count in range(nFrames + 1):
            ret, frame = cap.read()
            # frame = cv2.resize(frame, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
            image = cv2.putText(frame, "count frame: " + str(count), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Displaying the image
            cv2.imshow('capture_image ', image)

            cv2.waitKey(1)
            if count == nFrames:
                print("Save picture." + str(file))
                cv2.imwrite(file,frame)
                break

        cap.release()
        cv2.destroyAllWindows()

    def data_load_info(self, name_cup):
        """Get info of image"""

        name_file = open('data_cup.csv')
        name,height,width,volume=np.loadtxt('data_cup.csv',
                                    delimiter=';',
                                    unpack=True,
                                    dtype='str')

        for i in range(len(name_file.readlines())):
            if (name[i] == name_cup):
                print("Name: " + name[i], "\nHeight: " + height[i], "\nWidth: " + width[i], "\nVolume: " + volume[i])
                break
