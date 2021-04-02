import cv2
from DetectBBandCheckBlur import detectBB
image=cv2.imread("C:/Users/anhco/Desktop/DataIBE/8.jpg")
cv2.imshow("result",detectBB(image))
cv2.waitKey(0)