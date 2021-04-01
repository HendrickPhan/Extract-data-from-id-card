import cv2
import time
from imutils.video import VideoStream
from Crop4Corner import Crop_img,ExtractIDAndName
import imutils
from pyimagesearch.blur_detector import detect_blur_fft
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = [ 'IDCard']


#vc = cv2.VideoCapture("C:/Users/anhco/Desktop/VideoTest/20210312_112017.mp4")
vc = VideoStream(src=0).start()
#fps = vc.get(cv2.CAP_PROP_FPS)
#frame_width = int(vc.get(3))
#frame_height = int(vc.get(4))
#print(frame_width,frame_height)
#out = cv2.VideoWriter('C:/Users/anhco/Desktop/VideoTest/OutPutDemoIDCard.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
net = cv2.dnn.readNet("./DataIBEModelNewest/custom-yolov4-tiny-detector_final.weights",
                      "./DataIBEModelNewest/custom-yolov4-tiny-detector.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(255, 255), scale=1/255)

time_start= time.time()
count=0
frame = vc.read()
print(frame.shape)
while cv2.waitKey(1) < 1:
    frame = vc.read()
    #frame=cv2.imread("C:/Users/anhco/Desktop/DataIBE/2.jpg")
    if count==0:
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
            cv2.imshow("crop_img",crop_img)
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
                cv2.imshow("IDCard",crop_img)
                IDCard=Crop_img(crop_img)
                if type(IDCard) != None:
                    ExtractIDAndName(IDCard)
                cv2.waitKey(0)
                break
            text = text.format(mean)
            cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        end_drawing = time.time()
        count+=0
        fps_label = "FPS: %.2f " % (1 / (end - start))
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.imshow("IDCard",crop_img)
cv2.waitKey(0)
#vc.release()
#out.release()

print("The video was successfully saved")
