import pytesseract
import cv2
import matplotlib.pyplot as plt
import os 
import sys
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
img = cv2.imread('./c.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (445,95))
text=pytesseract.image_to_string(img, lang="eng")
print(text)

hImg, wImg, _ = img.shape
boxes = pytesseract.image_to_boxes(img)
ocr = []
for b in boxes.splitlines():
    b = b.split(' ')
    print(b[0])
    ocr.append(b[0])
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (50, 50, 255), 1)
    cv2.putText(img, b[0], (x, hImg - y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 205, 50), 1)
cv2.imshow('crop',img)
plt.imshow(img)
plt.show()
#img_text = pytesseract.image_to_string(Image.open(filename))
#cv2.imwrite('./teser1.png', img)
#pytesseract.image_to_boxes(img)
#ocr'''
#pytesseract.image_to_boxes(img)