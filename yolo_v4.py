#!/usr/bin/env python
# -*- coding: utf-8 -*-
# YOLO v4 object detection

import cv2
import numpy as np 
import time

class Yolo_v4:
    """Classe Yolo v4"""

    def load_argument(self):
        """ load all arguments"""
        input_file = '/content/object-detection-yolo-opencv/picture_cup.png'
        output_file='None.jpg'
        labels_file='/content/object-detection-yolo-opencv/data/obj.names' #'data/coco.names'
        config_file='/content/object-detection-yolo-opencv/cfg/custom-yolov4-detector.cfg' #'cfg/yolov3-tiny.cfg'
        weights_file='/content/object-detection-yolo-opencv/weight/custom-yolov4-detector_final.weights' #'yolov3-tiny.weights'

        return input_file, output_file, labels_file, config_file, weights_file

    def post_process(self, labels_file, img, outputs, conf):
        """ Load names of classes and get random colors """

        classes = open(labels_file).read().strip().split('\n')
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

        H, W = img.shape[:2]

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            scores = output[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w//2), int(y - h//2)
                p1 = int(x + w//2), int(y + h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                # cv2.rectangle(img, p0, p1, (255, 255, 255), 1)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                label = str(classes[class_ids[i]])
                text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
                cv2.putText(img, text, (x + 15, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                print(text)
                return label

    def load_image(self, input_file, output_file, labels_file, config_file, weights_file):
        global img, img0, outputs, ln

        # Give the configuration and weight files for the model and load the network.
        net = cv2.dnn.readNet(config_file, weights_file)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        # determine the output layer
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        img0 = cv2.imread(input_file)
        img0 = cv2.resize(img0, None, fx=0.9, fy=0.9)
        img = img0.copy()
        
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)
        t0 = time.time()
        outputs = net.forward(ln)
        t = time.time() - t0

        outputs = np.vstack(outputs)

        name_label = self.post_process(labels_file, img, outputs, 0.5)
        cv2.imshow('Detection window',  img)
        # cv2.waitKey(0)
        return name_label

    def yolo_object_detection(self):
        """ Yolo object detection loading """

        input_file, output_file, labels_file, config_file, weights_file = self.load_argument()
        name_label = self.load_image(input_file, output_file, labels_file, config_file, weights_file)
        return name_label
