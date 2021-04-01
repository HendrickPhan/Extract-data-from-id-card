import cv2
import pytesseract
import numpy as np
import re
def ocr(image):
    #ret,image = cv2.threshold(image,150,250,cv2.THRESH_BINARY)
    #cv2.imshow("a",image)
    #cv2.waitKey(0)
    t = pytesseract.image_to_string(image, lang='Vietnamese',config=custom_config)
    return t
custom_config = r'-c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ --psm 13'
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

#img=cv2.imread("./gray.jpg")
#t=ocr((img))
#print(t)
for i in range(0,59):
    print(i)
    im_gray = cv2.imread('./aha'+str(i)+'.jpg', cv2.IMREAD_GRAYSCALE)
    im_gray = cv2.GaussianBlur(im_gray,(1,1),0)
    (thresh, im_bw) = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh=thresh*0.9
    print(thresh)
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('bw_image.png', im_bw)
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(im_bw,kernel,iterations = 1)
    cv2.imshow("dilation",dilation)
    erosion = cv2.erode(dilation,kernel,iterations = 1)
    cv2.imshow("erosion",erosion)

    #opening = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)
    #cv2.imshow("opening",opening)
    #sharpened_image = unsharp_mask(im_bw)
    #cv2.imshow("sharpened image",sharpened_image)
    #sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    #sharped_img = cv2.filter2D(im_bw, -1, sharpen_filter)
    #cv2.imshow("sharped image",sharped_img)
    t=ocr((erosion))
    print(t)
    #a=re.findall(r'\d+', t)
    #print(a)
    cv2.waitKey(0)