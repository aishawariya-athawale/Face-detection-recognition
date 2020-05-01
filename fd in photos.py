import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

face_csc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv2.imread("IMG20190924133808.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_csc.detectMultiScale(gray, 1.3, 5)
for (x,y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h), (255,255,255),1)
        roi_gray = img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
#cv2.imshow("image", img)        
cv2.waitKey(0)
cv2.destroyAllWindows()     
