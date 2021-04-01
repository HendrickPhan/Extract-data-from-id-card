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



img = cv2.imread("./10.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plot_img(img)





face, number_img, name_img, dob_img, gender_img, nation_img, \
country_img, address_img, country_img_list, address_img_list = detect_info(
    img)



list_image = [face, number_img, name_img, dob_img,
              gender_img, nation_img, country_img, address_img]
j=0
list_label=["face", "number_img", "name_img", "dob_img","gender_img", "nation_img", "country_img", "address_img"]
for i in list_image:
    print(list_label[j])
    j=j+1
    show_img(i)


