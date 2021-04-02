import requests
from flask import Flask, render_template, request, redirect, url_for, send_from_directory,jsonify
from werkzeug.utils import secure_filename
import numpy as np
import os
import sys
import json
import cv2
from Crop4CornerVer2 import Crop_img,ExtractIDAndName
import imutils
from pyimagesearch.blur_detector import detect_blur_fft
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = [ 'IDCard']
net = cv2.dnn.readNet("./DataIBEModelNewest/custom-yolov4-tiny-detector_final.weights",
                      "./DataIBEModelNewest/custom-yolov4-tiny-detector.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(255, 255), scale=1/255)
sys.path.append("..")





def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DOWNLOAD_FOLDER'] = 'dowloads/'
app.config['IDCARD_FOLDER']='IDCard/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def Process(image_path,filename):
    info={}
    frame = cv2.imread(image_path)
    classes, scores, boxes = model.detect(frame, 0.7, NMS_THRESHOLD)
    if len(classes)>=1:
        for (classid, score, box) in zip(classes, scores, boxes):
            crop_img = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            frame1 = imutils.resize(crop_img, width=500)
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            (mean, blurry) = detect_blur_fft(gray, size=60,
                                             thresh=19, vis=False)
            if blurry:
                return {
                    "blur": True,
                    "id_number": None,
                    "name": None
                }
            else:
                IDCard=Crop_img(frame)
                cv2.imwrite(app.config['IDCARD_FOLDER']+'IDCard_Of_' + filename,IDCard)
                if type(IDCard) != None:
                    text=ExtractIDAndName(IDCard)
                break
        info["blur"]=False
        for i,content in enumerate(text):
            if i==0:
                info["id_number"]=content
            if i==2 or i==1:
                if content !=None:
                    info["name"]=content
        with open('./OutputJson/info.json', 'a',encoding='utf8') as json_file:
            json.dump(info, json_file,ensure_ascii=False)
        return info
    else:
        return jsonify('Fail Cannot find any IDCard in image '+filename)
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    info={}
    for image_path in TEST_IMAGE_PATHS:
        return (Process(image_path,filename))

@app.route('/upload/url', methods=['POST'])
def url():
    url = request.json
    filename = url['url'].split('/')[-1]
    if allowed_file(filename):
        with open(app.config['DOWNLOAD_FOLDER']+filename, 'wb') as f:
            f.write(requests.get(url["url"]).content)
        return Process(app.config['DOWNLOAD_FOLDER']+filename,filename)
    else:
        return jsonify('Fail incorrect type Image'+filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
