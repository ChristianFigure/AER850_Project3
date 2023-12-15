from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
import torch


model = YOLO("yolov8n.pt")
results = model.train(data=r'D:\4th Year File (7th Semester)\AER 850\Project3\Project 3 Data\data\data.yaml', 
                      epochs= 30,
                      batch= 4,
                      imgsz = 1200)

img = cv2.imread(r'D:\4th Year File (7th Semester)\AER 850\Project3\Project 3 Data\data\evaluation\ardmega.jpg')
img = cv2.resize(img,(724,543))
pred = model.predict(img)
pred_array = pred[0].plot()
cv2.imshow("maskedFinal", pred_array)
cv2.waitKey(0)

img = cv2.imread(r'D:\4th Year File (7th Semester)\AER 850\Project3\Project 3 Data\data\evaluation\arduno.jpg')
img = cv2.resize(img,(724,543))
pred = model.predict(img)
pred_array = pred[0].plot()
cv2.imshow("maskedFinal", pred_array)
cv2.waitKey(0)

img = cv2.imread(r'D:\4th Year File (7th Semester)\AER 850\Project3\Project 3 Data\data\evaluation\rasppi.jpg')
img = cv2.resize(img,(724,543))
pred = model.predict(img)
pred_array = pred[0].plot()
cv2.imshow("maskedFinal", pred_array)
cv2.waitKey(0)



