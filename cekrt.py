
import time
import cv2
import mss
import numpy
import os
import numpy as np
import cv2
from sys import modules
import torch
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
model = torch.hub.load('ultralytics/yolov5', 'custom', path='modelbaru\\best.pt') 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

title = "uji coba screen graber"
start_time = time.time()
display_time = 1
fps = 0
sct = mss.mss()
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
mon = (0, 40, 800, 640)



def screen_recordMSS():
    global fps, start_time
    while True:
        img = numpy.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = model(frame[..., ::-1])
        results = detections.pandas().xyxy[0].to_dict(orient="records")
        print(results)
        for result in results:
                con = result['confidence']
                cs = result['name']
                x1 = int(result['xmin'])
                y1 = int(result['ymin'])
                x2 = int(result['xmax'])
                y2 = int(result['ymax'])
               
                #tesseract
                crop = frame[y1:y2, x1:x2]
                img2 = cv2.resize(crop, (445,95))

                hImg, wImg, _ = img2.shape
                text=pytesseract.image_to_string(img2, lang="eng")
                print(text)
                boxes = pytesseract.image_to_boxes(img2)
                ocr= []
                for b in boxes.splitlines():
                    b = b.split(' ')
                    print(b[0])
                    ocr.append(b[0])
                    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                    cv2.rectangle(img2, (x, hImg - y), (w, hImg - h), (50, 50, 255), 1)
                    cv2.putText(img2, b[0], (x, hImg - y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 205, 50), 1)
                
                cv2.putText(frame, str(cs) + " " , (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 2,(0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 255, 0), 2)
           

       
        # Display the resulting frame
        cv2.imshow('frame',frame)
        cv2.imshow('crop',img2)

        
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

screen_recordMSS()