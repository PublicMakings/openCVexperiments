import cv2
import numpy as np

'''requires massive xml files refering to an objext'''

eye_Cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#catface_Cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = eye_Cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
    cv2.imshow('facecam', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
