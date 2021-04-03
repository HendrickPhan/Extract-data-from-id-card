import cv2
import matplotlib.pyplot as plt
import numpy as np
net = cv2.dnn.readNet("./models/detectbb/detectbb.weights",
                      "./models/detectbb/detectbb.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(255, 255), scale=1/255)
def detectBB(frame):
    classes, scores, boxes = model.detect(frame, 0.7, NMS_THRESHOLD)
    (h, w, d) = frame.shape
    for (classid, score, box) in zip(classes, scores, boxes):
        box1=box.copy()
        x0=box1[0]
        x1=box1[1]
        x2=box1[2]
        x3=box1[3]
        if box[1]+box[3]<h-25:
            x3=box[1]+box[3]+25
        if box[1]>25:
            x1=box[1]-25
        if box[0]+box[2]<w-25:
            x2=box[0]+box[2]+25
        if box[0]>25:
            x0=box[0]-25
        crop_img = frame[x1:x3, x0:x2]
    return crop_img
def detect_blur_fft(image, size=60, thresh=10, vis=False):
    canny = cv2.Canny(image, 50,250)
    mean=cv2.mean(canny)[0]
    return (mean, mean <= thresh)
