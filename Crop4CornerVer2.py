import time
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
from unidecode import unidecode
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = './models/transformerocr.pth'
#config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False
detector = Predictor(config)
fname = './common_last_name.txt'
with open(fname,encoding="utf8") as f:
    last_name_list = f.readlines()
last_name_list = [x.strip().upper() for x in last_name_list]
last_name_decode = [unidecode(x) for x in last_name_list]
def detect_blur_fft(image, size=60, thresh=10, vis=False):
    canny = cv2.Canny(image, 50,250)
    mean=cv2.mean(canny)[0]
    return (mean, mean <= thresh)
def get_center_point(box):
    if len(box)==0:
        return ()
    xmin, ymin, w, h = box
    return (xmin + xmin+w) // 2, (ymin + ymin+h) // 2
def perspective_transoform(image, source_points):
    dest_points = np.float32([[0,0], [950,0], [950,510], [0,510]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (950, 510))

    return dst
def find_miss_corner(classes):
    a=[]
    count=0
    mark=-1
    for i in classes:
        a.append(i[0])
    for i in range(0,4):
        if i not in a:
            count+=1
            mark=i
    if count==1:
        return mark
    elif count==2:
        return -1
    else:
        return None
def calculate_missed_coord_corner(coordinate_dict,miss_corner):
    thresh = 0

    index = miss_corner

    # calculate missed corner coordinate
    # case 1: missed corner is "top_left"
    if index == 2:
        midpoint = np.add(coordinate_dict['Top_right'], coordinate_dict['Bottom_left']) / 2
        y = 2 * midpoint[1] - coordinate_dict['Bottom_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['Bottom_right'][0] - thresh
        coordinate_dict['Top_left'] = (x, y)
    elif index == 3:  # "top_right"
        midpoint = np.add(coordinate_dict['Top_left'], coordinate_dict['Bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['Bottom_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['Bottom_left'][0] - thresh
        coordinate_dict['Top_right'] = (x, y)
    elif index == 0:  # "bottom_left"
        midpoint = np.add(coordinate_dict['Top_left'], coordinate_dict['Bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['Top_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['Top_right'][0] - thresh
        coordinate_dict['Bottom_left'] = (x, y)
    elif index == 1:  # "bottom_right"
        midpoint = np.add(coordinate_dict['Bottom_left'], coordinate_dict['Top_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['Top_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['Top_left'][0] - thresh
        coordinate_dict['Bottom_right'] = (x, y)

    return coordinate_dict
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int"),pick
def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = [ 'Bottom_left', 'Bottom_right','Top_left', 'Top_right']


#net = cv2.dnn.readNet("D:/linh_tinh/4Corner/custom-yolov4-tiny-detector_final.weights",
#                      "D:/linh_tinh/4Corner/custom-yolov4-tiny-detector.cfg")
net = cv2.dnn.readNet("./models/detect4corner/detect4corner.weights",
                      "./models/detect4corner/detect4corner.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

time_start= time.time()
def Crop_img(frame):
    classes, scores, boxes = model.detect(frame, 0.1, NMS_THRESHOLD)
    end = time.time()
    frame1=frame.copy()
    start_drawing = time.time()
    if len(boxes)<3:
        return None
    if len(boxes)>4:
        new_boxes=[]
        for i in boxes:
            x1=i[0];y1=i[1];x2=i[0]+i[2];y2=i[1]+i[3]
            new_boxes.append([x1,y1,x2,y2])
        new_boxes,pick=non_max_suppression_fast(np.array(new_boxes),0.5)
        new_classes=[]
        new_boxes=[]
        pick=selection_sort(pick)
        for i in range(0,len(pick)):
            new_classes.append(classes[pick[i]])
            new_boxes.append(boxes[pick[i]])
        classes=np.array(new_classes)
        boxes=new_boxes
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()
    a=classes.copy()
    for i in range(1,len(a)):
        if a[i]==a[i-1]:
            if a[i]==0:
                if boxes[i][1]<boxes[i-1][1]:
                    a[i]=2
                else:
                    a[i-1]=2
            elif a[i]==1:
                if abs(boxes[i][1]-boxes[0][1])<100 or abs(boxes[i][0]-boxes[0][0])<100:
                    a[i-1]=3
                else:
                    a[i]=3

    news=[[],[],[],[]]
    for i in range(0,len(a)):
        news[a[i][0]]=boxes[i]

    final_points = list(map(get_center_point, news))
    label_boxes = dict(zip(class_names, final_points))

    miss_corner=find_miss_corner(a)
    if miss_corner!=None:
        label_boxes=calculate_missed_coord_corner(label_boxes,miss_corner)
    if miss_corner==-1:
        return None
    source_points = np.float32([
        label_boxes['Top_left'], label_boxes['Top_right'], label_boxes['Bottom_right'], label_boxes['Bottom_left']
    ])

    # Transform
    crop = perspective_transoform(frame1, source_points)
    #cv2.imshow("IDCard after align",crop)
    #cv2.waitKey(0)
    return crop
    #print(aha)
    #break
Size_Text=[[115,170,465,900],[165,235,395,950],[225,280,295,950],[280,326,455,950],[318,375,555,950],[375,425,295,950],[420,465,625,950],[465,510,295,950]]
def ExtractInfo(img,count):
    im_pil = Image.fromarray(img)
    s = detector.predict(im_pil)
    if count==0:
        a=re.findall(r'\d+', s)
        for j in a:
            if len(j)==9 or len(j)==12:
                return j
            if len(j)>9 and len(j)<12:
                return j[len(j)-9:]
    if count==1 or count==2:
        words = s.split()
        if len(words)>7:
            return ""
        for j in range(0,len(words)):
            #print(unidecode(words[j]).upper())
            if unidecode(words[j]).upper() in last_name_decode:
                if unidecode(words[j]).upper()=="THI" :
                    continue
                text = ' '.join(words[j:])
                return text

def ExtractIDAndName(img):
    count=0
    text=[]
    for i in Size_Text[:3]:
        crop_img = img[i[0]:i[1],i[2]:i[3]]
        text.append(ExtractInfo(crop_img,count))
        count+=1
    return text
