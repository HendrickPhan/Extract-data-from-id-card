import argparse
import cv2
from pyimagesearch.detector import detect_info
import matplotlib.pyplot as plt
import numpy as np
import sys


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def plot_img(img):
    plt.imshow(img)
    plt.show()




img = cv2.imread("./73.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plot_img(img)

warped = img


if warped is None:
    print('Cant find id card in image')
    sys.exit()

try:
    face, number_img, name_img, dob_img, gender_img, nation_img, \
    country_img, address_img, country_img_list, address_img_list = detect_info(
        warped)
except:
    print('Cant find id card in image')
    sys.exit()


list_image = [face, number_img, name_img, dob_img,
              gender_img, nation_img, country_img, address_img]

for y in range(len(list_image)):
    plt.subplot(len(list_image), 1, y+1)
    plt.imshow(list_image[y])
plt.show()

