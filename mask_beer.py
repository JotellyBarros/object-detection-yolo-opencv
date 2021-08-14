#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import the necessary packages
from collections import deque
import numpy as np
import argparse
import cv2

# Global variables
AreaContornoLimiteMin = 3000
pixel = 364
cmSize = 15.5

class Mask_beer:
    """Classe mask beer"""
    
    # Build a mask with the color reference, finding the range of beer color int the image
    def CreateMask(self, type_frame, lowerColor, upperColor):
        maskColor = cv2.inRange(type_frame, lowerColor, upperColor)
        maskColor = cv2.GaussianBlur(maskColor, (9, 9), 0)
        maskColor = cv2.erode(maskColor, None, iterations=2)
        maskColor = cv2.dilate(maskColor, None, iterations=2)
        return maskColor

    # Scans all found contours
    def SweepContours(self, typeColor, name_frame, cnt, height, volume, R,G,B):
        cont = 0
        for contour in cnt:
            # Smaller area outlines are ignored.
            if cv2.contourArea(contour) > AreaContornoLimiteMin:
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

                vMlCmY = (volume / height) # Cm to ml estimated total
                mlEstimated = (vMlCmY * cmY) # Measured cup per cm
                print(str(mlEstimated) + " ml")
                print(str(volume) + " ml")

                if(mlEstimated > 5):
                    cv2.putText(name_frame, str(round(float(mlEstimated), 2)) + " ml", (x + 90, y+h-100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),2) # ml estimated > height
                    # cv2.putText(name_frame, str(h) + " px", (x + 120, y+h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) # Area in pixel Y
                    
                    # Displays name type
                    cv2.putText(name_frame,typeColor,(x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (R, G, B), 1, cv2.LINE_AA)

                    # if len(cnt) != 0:
                    #     cv2.drawContours(name_frame, cnt, contourIdx=-1, color=(R, G, B), thickness=-1, lineType=cv2.LINE_AA)

                    if (mlEstimated >= volume):
                        image = cv2.putText(name_frame, "Enjoy with moderation...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                        print("Enjoy with moderation...")
                        return True
        return False

                
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

        beerServed = False
        spumeServed = False

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
            
            # Scans all found contours
            if ((len(cntbeer) > 0) and (beerServed == False)):
                beerServed = self.SweepContours("Beer", frame, cntbeer, height, volume, 0, 255, 0)
                cv2.imshow(name, frame)
   
            if (len(cntspume) > 0) and (spumeServed == False):
                spumeServed = self.SweepContours("Spume", frame, cntspume, height, volume, 0, 255, 0)
                cv2.imshow(name, frame)
            # -----------------------------------------------------------

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
