#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import csv
import numpy as np
import time

def capture_image():
    file = "./cup.png"
    nFrames = 0

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
        cv2.imshow('capture_image', frame)

        nFrames = nFrames + 1

        cv2.waitKey(1)
        if nFrames == 30:
            cv2.imwrite(file,frame)
            break

    cap.release()
    cv2.destroyAllWindows()

def data_load_info(name_cup):
    arquivo = open('data_cup.csv')
    name,height,width,volume=np.loadtxt('data_cup.csv',
                                delimiter=';',
                                unpack=True,
                                dtype='str')

    for i in range(len(arquivo.readlines())):
        if (name[i] == name_cup):
            print("Name: " + name[i], "\nHeight: " + height[i], "\nWidth: " + width[i], "\nVolume: " + volume[i])
            break

def yolo_object_detection():
    return "Chopp_Tulipa"

def main():
    capture_image()
    name_cup_chopp = yolo_object_detection()
    data_load_info(name_cup_chopp) # Chopp_Tulipa

if __name__ == '__main__':
    main()
    # sys.exit(main(sys.argv))