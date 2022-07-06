import numpy as np
import cv2
from sys import modules
import torch
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') 
cap = cv2.VideoCapture('rtsp://192.168.43.109:8554/Streaming/Channels/101')


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.resize(frame,(640,640))
    #print(frame.shape)
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
                img = cv2.resize(crop, (445,95))

                hImg, wImg, _ = img.shape
                text=pytesseract.image_to_string(img, lang="eng")
                print(text)
                boxes = pytesseract.image_to_boxes(img)
                ocr= []
                for b in boxes.splitlines():
                    b = b.split(' ')
                    print(b[0])
                    ocr.append(b[0])
                    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])

                    cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (50, 50, 255), 1)
                    cv2.putText(img, b[0], (x, hImg - y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 205, 50), 1)
                
                cv2.putText(frame, str(cs) + " " , (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 2,(0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 255, 0), 2)
           

       
    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('crop',img)

  
    #results.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

