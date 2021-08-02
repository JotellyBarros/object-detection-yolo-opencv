from collections import deque
import numpy as np
import argparse
# import imutils
import cv2

# Global variables
AreaContornoLimiteMin = 300
posX  = 0
posY  = 0
lastX = 0
lastY = 0
pixel = 426
cmSize = 8

cv2.namedWindow("window")
cv2.namedWindow("window_Canny")

# Import video camera
cap = cv2.VideoCapture(2)

def mause(pos):
    pass

# Setting Camera Parameters
cv2.createTrackbar("CONTRAST", "window" , 50, 100, mause)
cv2.createTrackbar("BRIGHTNESS", "window" , 50, 100, mause)

# Setting Canny Parameters
cv2.createTrackbar("Canny_min", "window_Canny" , 150, 300, mause)
cv2.createTrackbar("Canny_max", "window_Canny" , 200, 300, mause)

# Define the range upper and lower limits of each color
# Defining the Range of Beer color
beerLower=np.array([128, 115, 150],np.uint8)
beerUpper=np.array([255, 255, 255],np.uint8)

# Defining the Range of Spume color
spumeLower=np.array([69, 7, 197],np.uint8) # 10, 0, 145
spumeUpper=np.array([164, 193, 255],np.uint8) # 180, 56, 255

# Find the largest contour of the mask
def SweepRadiusContours(cntColor, name_frame, R, G, B):
    cColor = max(cntColor, key=cv2.contourArea)
    # rectColor = cv2.minAreaRect(cColor)
    # boxColor = cv2.boxPoints(rectColor)
    # boxColor = np.int0(boxColor)
    MColor = cv2.moments(cColor)
    centerColor = (int(MColor["m10"] / MColor["m00"]), int(MColor["m01"] / MColor["m00"]))
    ((x, y), radius) = cv2.minEnclosingCircle(cColor)
    center = (int (x), int (y))
    radius = int (radius)
    if radius > 5:
        circle = cv2.circle (name_frame, center, radius - 5, (R, G, B), 2)
        circle = cv2.circle (name_frame, center, radius + 5, (R, G, B), 2)
        # cv2.drawContours (name_frame, [boxColor], -1, (R, G, B), 2)

        # Displays name type
        # cv2.putText(name_frame, "Radius:" + str(radius),(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (R, G, B))
        # cv2.putText(name_frame,"("+str(center[0])+","+str(center[1])+")", (center[0]+10,center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(R, G, B), 2, cv2.LINE_AA)
        cv2.putText(name_frame, "Diametro:" + str((radius * cmSize * 2)/pixel) + "mm",(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (R, G, B), 2)
        cv2.putText(name_frame, "Raio:" + str(((radius * cmSize * 2)/pixel)/2) + "mm",(int(x+30), int(y+30)),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (R, G, B), 2)

# Scans all found contours
def SweepContours(typeColor, name_frame, cnt, R,G,B):
    for contour in cnt:
        # Smaller area outlines are ignored.
        if cv2.contourArea(contour) < AreaContornoLimiteMin:
            continue
        # Get contour coordinates
        (x, y, w, h) = cv2.boundingRect(contour) #x e y: coordinates of the upper left vertex
                                                #w e h: respectively width and height of the rectangle
        cv2.rectangle(name_frame,(x,y),(x+w,y+h),(0,255,0),2)
        cmX = (w*cmSize)/pixel # (w * (measured on the ruler)) / (pixel amount)
        cmY = (h*cmSize)/pixel # (h * (measured on the ruler)) / (pixel amount)
 
        # Displays pixel quantity of the XY axis
        # cv2.putText(name_frame, str(w), (x+w+10, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1) # Area in pixel X
        # cv2.putText(name_frame, str(h), (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),1) # Area in pixel Y
        
        # Displays pixel measure of the XY axis
        cv2.putText(name_frame, str(int(cmX)) + " cm", (x+w+10, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2) # Eixo X
        cv2.putText(name_frame, str(int(cmY)) + " cm", (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),2) # Eixo Y
        # Displays name type
        cv2.putText(name_frame,typeColor,(x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (R, G, B))

# Build a mask with the color reference, finding the range of beer color int the image
def CreateMask(type_frame, lowerColor, upperColor):
    maskColor = cv2.inRange(type_frame, lowerColor, upperColor)
    maskColor = cv2.GaussianBlur(maskColor, (9, 9), 1)
    # maskColor = cv2.erode(maskColor, None, iterations=2)
    maskColor = cv2.dilate(maskColor, None, iterations=2)
    return maskColor

while(cap.isOpened()):
    success, img = cap.read()
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Build a mask with the color reference
    maskbeer = CreateMask(hsv_frame, beerLower, beerUpper)
    beer = cv2.bitwise_and(hsv_frame, hsv_frame, mask = maskbeer)

    # Encontrar contornos da mascara e inicializar a corrente
    cntbeer = cv2.findContours(maskbeer.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cntbeer) > 0:
        # Scans all found contours
        SweepContours("tam", img, cntbeer, 0, 255, 0)

        # Find the largest contour of the mask
        # SweepRadiusContours(cntbeer, img, 0, 255, 255)

    contrast = cv2.getTrackbarPos("CONTRAST", 'window')
    brightness = cv2.getTrackbarPos("BRIGHTNESS", 'window')

    canny_min = cv2.getTrackbarPos("Canny_min", 'window_Canny')
    canny_max = cv2.getTrackbarPos("Canny_max", 'window_Canny')

    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness) # BRIGHTNESS
    cap.set(cv2.CAP_PROP_CONTRAST, contrast) # CONTRAST
    
    # Convert Color to Grayscale
    imgGray = cv2.cvtColor(beer, cv2.COLOR_BGR2GRAY)

    string = "CONTRAST " + str(contrast)+" BRIGHTNESS " + str(brightness)
    cv2.putText(beer, string, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255))
   
    # Canny Edge Detection in OpenCV
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, canny_min, canny_max)
    
    string_canny = "Canny_min " + str(canny_min)+" Canny_max " + str(canny_max)
    cv2.putText(imgCanny, string_canny, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 255, 255))
    
    imgCanny = cv2.dilate(imgCanny, None, iterations=1)

    # Show image
    cv2.imshow("window", img)
    cv2.imshow("window_Canny", imgCanny)

    # Control edge thresholds
    edge_vis = img.copy()
    edge_vis = np.uint8(edge_vis/3.)
    edge_vis[imgCanny != 0] = (0, 255, 0)
    cv2.imshow('window_Canny_edge', edge_vis)

    # Wainting key "q"
    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break

