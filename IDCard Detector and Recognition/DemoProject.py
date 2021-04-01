import cv2
import time
import os
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = [ 'bottom_left', 'bottom_right','top_left', 'top_right']


vc = cv2.VideoCapture("C:/Users/anhco/Desktop/VideoTest/DemoIDCard.mp4")
fps = vc.get(cv2.CAP_PROP_FPS)
frame_width = int(vc.get(3))
frame_height = int(vc.get(4))
print(frame_width,frame_height)
out = cv2.VideoWriter('C:/Users/anhco/Desktop/VideoTest/OutPutDemoIDCard.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
net = cv2.dnn.readNet("D:/linh_tinh/DataIBEModelNewest/custom-yolov4-tiny-detector_final.weights",
                      "D:/linh_tinh/DataIBEModelNewest/custom-yolov4-tiny-detector.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

thresh=int(round((fps/2.5),0))
time_start= time.time()
count=0
while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        time_end= time.time()
        print("time processing is:",time_end-time_start)
        break
    if count==0:
        #images = [f.path for f in os.scandir("/content/gdrive/My Drive/id-card-detector/test_images/")]
        #for path in images:
        #frame = cv2.imread(path)
        start = time.time()
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()

        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_names[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        end_drawing = time.time()

        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        out.write(frame)
        count+=1
    elif count>0 and count<thresh:
        count+=1
        out.write(frame)
    elif count==thresh:
        count=0
        out.write(frame)
vc.release()
out.release()

print("The video was successfully saved")