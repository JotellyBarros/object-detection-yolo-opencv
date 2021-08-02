#!/usr/bin/env python
# -*- coding: utf-8 -*-
# YOLO v4 object detection

import cv2
import numpy as np 
import time

class Yolo_v4:
    """Classe Yolo v4"""

    def load_argument(self):
        input_file = '/content/object-detection-yolo-opencv/cup.png'
        output_file='None.jpg'
        labels_file='/content/darknet/data/obj.names' #'data/coco.names'
        config_file='/content/darknet/cfg/custom-yolov4-detector.cfg' #'cfg/yolov3-tiny.cfg'
        weights_file='/content/darknet/backup/custom-yolov4-detector_final.weights' #'yolov3-tiny.weights'

        return input_file, output_file, labels_file, config_file, weights_file

    def load_yolo(self, config_file, weights_file, labels_file):
        net = cv2.dnn.readNet(config_file, weights_file)
        classes = []
        with open(labels_file, "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layers_names = net.getLayerNames()
        output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        return net, classes, colors, output_layers

    def load_image(self, input_file):
        img = cv2.imread(input_file)
        img = cv2.resize(img, None, fx=0.9, fy=0.9)
        height, width, channels = img.shape
        return img, height, width, channels

    def detect_objects(self, img, net, outputLayers):			
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(608, 608), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(outputLayers)
        return blob, outputs
    
    def get_box_dimensions(self, outputs, height, width):
        boxes = []
        confs = []
        class_ids = []
        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.3:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
        return boxes, confs, class_ids

    def draw_labels(self, boxes, confs, colors, class_ids, classes, img): 
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                text = "{}: {:.4f}".format(classes[class_ids[i]], confs[i])
                cv2.putText(img, text, (x, y - 5), font, 1, color, 1)
                print(label)
        cv2.imshow("Image", img)
        return label

    def image_detect(self, input_file, output_file, labels_file, config_file, weights_file):
        model, classes, colors, output_layers = self.load_yolo(config_file, weights_file, labels_file)
        image, height, width, channels = self.load_image(input_file)
        blob, outputs = self.detect_objects(image, model, output_layers)
        boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)
        name_label = self.draw_labels(boxes, confs, colors, class_ids, classes, image)
        
        while True:
            key = cv2.waitKey(1)
            if key == 27:
                break
            
        return name_label

    def yolo_object_detection(self):
        print("Yolo object detection loading...")
        
        input_file, output_file, labels_file, config_file, weights_file = self.load_argument()

        # print("input_file:  "  + input_file)
        # print("labels_file: "  + labels_file)
        # print("config_file: "  + config_file)
        # print("weights_file: " + weights_file)

        name_label = self.image_detect(input_file, output_file, labels_file, config_file, weights_file)
        return name_label
