import cv2
from time import time
import numpy as np


face_csc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
VC = cv2.VideoCapture(0)
while True:
    ret, img = VC.read()
    start = time()
    if ret is True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        continue
    faces = face_csc.detectMultiScale(gray, 1.3, 5) 
    print(time()-start)

    for (x,y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h), (15,150,100),3)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
    cv2.imshow("image", img)
    cv2.waitKey(5)

VC.release()
cv2.destroyAllWindows()