
import pytesseract
import cv2
def ocr(image):
    ret,image = cv2.threshold(image,150,250,cv2.THRESH_BINARY)
    cv2.imshow("a",image)
    cv2.waitKey(0)
    t = pytesseract.image_to_string(image, lang='vie',config="--psm 12 --oem 1")
    return t
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

img=cv2.imread("./gray.jpg")
t=ocr((img))
print(t)
