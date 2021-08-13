#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import the necessary packages
from collections import deque
import numpy as np
import argparse
# import imutils
import cv2

# Global variables
AreaContornoLimiteMin = 300
# posX  = 0
# posY  = 0
# lastX = 0
# lastY = 0
pixel = 394
cmSize = 13.5

# Measured cup per cm
altCup = 80 # Measured cup height
sizeCupMl = 980 # Measured cup 150 ml

class Mask_beer:
    """Classe mask beer"""
    
    # Build a mask with the color reference, finding the range of beer color int the image
    def CreateMask(self, type_frame, lowerColor, upperColor):
        maskColor = cv2.inRange(type_frame, lowerColor, upperColor)
        maskColor = cv2.GaussianBlur(maskColor, (9, 9), 0)
        maskColor = cv2.erode(maskColor, None, iterations=2)
        maskColor = cv2.dilate(maskColor, None, iterations=2)
        return maskColor

    # Find the largest contour of the mask
    def Findcontour(self, frame, cntColor, R, G, B):
        cColor = max(cntColor, key=cv2.contourArea)
        rectColor = cv2.minAreaRect(cColor)
        boxColor = cv2.boxPoints(rectColor)
        boxColor = np.int0(boxColor)
        MColor = cv2.moments(cColor)
        centerColor = (int(MColor["m10"] / MColor["m00"]), int(MColor["m01"] / MColor["m00"]))
        cv2.circle(frame, (centerColor), 7, (255, 255, 255), -1)
        # cv2.drawContours(frame, [boxColor], -1, (R, G, B), 2)
        cv2.drawContours(frame, contours=cntColor, contourIdx=-1, color=(R, G, B), thickness=-1, lineType=cv2.LINE_AA)

    # Scans all found contours
    def SweepContours(self, typeColor, name_frame, cnt, R,G,B):
        cont = 0
        for contour in cnt:
            # Smaller area outlines are ignored.
            if cv2.contourArea(contour) < AreaContornoLimiteMin:
                continue
            # Get contour coordinates
            (x, y, w, h) = cv2.boundingRect(contour) #x e y: coordinates of the upper left vertex
                                                    #w e h: respectively width and height of the rectangle
            cv2.rectangle(name_frame,(x,y),(x+w,y+h),(0,255,0),2)
            # cmX = (w*cmSize)/pixel # (w * (measured on the ruler)) / (pixel amount)
            # cmY = (h*cmSize)/pixel # (h * (measured on the ruler)) / (pixel amount)
            
            # Displays pixel quantity of the XY axis
            # cv2.putText(name_frame, str(w), (x+w+10, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1) # Area in pixel X
            # cv2.putText(name_frame, str(h), (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),1) # Area in pixel Y

            cmX = round(((float(w)*float(cmSize))/pixel),2) # (w * (measured on the ruler)) / (pixel amount) Eixo X / width
            cmY = round(((float(h)*float(cmSize))/pixel),2) # (h * (measured on the ruler)) / (pixel amount) Eixo Y / height

            # Cm to ml estimated total
            vMlCmY = (sizeCupMl / altCup)
            mlEstimated = (vMlCmY * cmY)

            # print(str(mlEstimated) + " ml")
            cv2.putText(name_frame, str(mlEstimated) + " ml", (x + 90, y+h-100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),2) # ml estimated > height
            cv2.putText(name_frame, str(h) + " px", (x + 120, y+h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) # Area in pixel Y
            
            # Displays name type
            cv2.putText(name_frame,typeColor,(x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (R, G, B))
            
        #     if(cmX > 1 and cmY > 1):
        #         cont = w + cont
           
        #         # Displays pixel measure of the XY axis
        #         cv2.putText(name_frame, "{:.2f}".format(cmX) + " cm", (x+w+10, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2) # Eixo X
        #         cv2.putText(name_frame, "{:.2f}".format(cmY) + " cm", (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),2) # Eixo Y

        #         # Displays name type
        #         cv2.putText(name_frame,typeColor,(x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (R, G, B))
        # print(cont)
    
    def capture_image(self, name, height, width, volume, camera):
        """Get capture of image"""

        print("Get capture of image mask_beer...")
        cap = cv2.VideoCapture(camera)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        while True:
            ret, frame = cap.read()            
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # -----------------------------------------------------------
            # Define the range upper and lower limits of each color
            # Defining the Range of Beer color
            beerLower=np.array([6, 120, 50],np.uint8)
            beerUpper=np.array([50, 255, 255],np.uint8)

            # Defining the Range of Spume color
            spumeLower=np.array([69, 7, 197],np.uint8) # 10, 0, 145
            spumeUpper=np.array([164, 193, 255],np.uint8) # 180, 56, 255

            # Build a mask with the color reference
            maskbeer = self.CreateMask(hsv_frame, beerLower, beerUpper)
            beer = cv2.bitwise_and(hsv_frame, hsv_frame, mask = maskbeer)

            maskspume = self.CreateMask(hsv_frame, spumeLower, spumeUpper)
            spume = cv2.bitwise_and(hsv_frame, hsv_frame, mask = maskspume)

            # Encontrar contornos da mascara e inicializar a corrente
            cntbeer = cv2.findContours(maskbeer.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            centerbeer = None
            cntspume = cv2.findContours(maskspume.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            centerspume = None

            if len(cntbeer) > 0:
                # Find the largest contour of the mask
                self.Findcontour(frame, cntbeer, 0, 255, 0)

                # Scans all found contours
                self.SweepContours("Beer", frame, cntbeer, 0, 255, 0)
            # -----------------------------------------------------------

            # Displaying the image
            cv2.imshow(name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
