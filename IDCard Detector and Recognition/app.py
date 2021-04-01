from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import os
import sys
import json
import tensorflow as tf
from PIL import Image
import cv2
import time
from imutils.video import VideoStream
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
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file',
                                filename=filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    IMAGE_SIZE = (12, 8)
    info={}
    for image_path in TEST_IMAGE_PATHS:
        frame = cv2.imread(image_path)
        #images = [f.path for f in os.scandir("/content/gdrive/My Drive/id-card-detector/test_images/")]
        #for path in images:
        #frame = cv2.imread(path)
        start = time.time()
        classes, scores, boxes = model.detect(frame, 0.7, NMS_THRESHOLD)
        end = time.time()
        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_names[classid[0]], score)
            crop_img = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            cv2.rectangle(frame, box, color, 2)
            #cv2.imwrite("./9.jpg",crop_img)
            frame1 = imutils.resize(crop_img, width=500)
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            (mean, blurry) = detect_blur_fft(gray, size=60,
                                             thresh=19, vis=False)
            # draw on the frame, indicating whether or not it is blurry
            color = (0, 0, 255) if blurry else (0, 255, 0)
            #text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
            if blurry:
                text = "Blurry ({:.4f})"
            else:
                cv2.imwrite("./CropImage.jpg",crop_img)
                IDCard=Crop_img(crop_img)
                if type(IDCard) != None:
                    text=ExtractIDAndName(IDCard)
                cv2.waitKey(0)
                break
        im = Image.fromarray(frame)
        im.save('uploads/' + filename)
        print(text)
        for i,content in enumerate(text):
            if i==0:
                info["ID"]=content
            if i==2 or i==1:
                if content !=None:
                    info["Name"]=content
        print(info)
        with open('./uploads/info.json', 'a',encoding='utf8') as json_file:
                json.dump(info, json_file,ensure_ascii=False)
    cv2.destroyAllWindows()
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
