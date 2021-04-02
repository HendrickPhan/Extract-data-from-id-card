import cv2
import matplotlib.pyplot as plt
import numpy as np
net = cv2.dnn.readNet("D:/linh_tinh/DataIBEModelNewest/custom-yolov4-tiny-detector_final.weights",
                      "D:/linh_tinh/DataIBEModelNewest/custom-yolov4-tiny-detector.cfg")
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
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # check to see if we are visualizing our output
    if vis:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))

        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("dowloads")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        # show our plots
        plt.show()

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean <= thresh
