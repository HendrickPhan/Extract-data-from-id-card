import cv2
import matplotlib.pyplot as plt
import imutils
from pyimagesearch.blur_detector import detect_blur_fft
img=cv2.imread("D:/Result/0.jpg")
print(img.shape)
#img_resize=cv2.resize(img,(950,510))
Size_Text=[[115,170,370,900],[165,235,265,950],[229,270,295,925],[270,333,265,925],[330,385,255,950],[380,430,255,950],[420,475,255,950],[465,510,255,950]]
#crop_img = img_resize[280:316,302:903]
temp=0
for j in range(0,999):
    print(j)
    img=cv2.imread("D:/Result/"+str(j)+".jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (mean, blurry) = detect_blur_fft(gray, size=60,
                                     thresh=20, vis=-1 > 0)
    if not blurry:
        for i in Size_Text[:2]:
            #img_resize=cv2.rectangle(img_resize,(i[2],i[0]),(i[3],i[1]), (255,0,0), 2)
            crop_img = img[i[0]:i[1],i[2]:i[3]]
            #cv2.imshow("cropped", crop_img)
            cv2.imwrite("./aha"+str(temp)+".jpg",crop_img)
            temp+=1
            cv2.waitKey(0)
    print(temp)

plt.show() # Show the image windowq