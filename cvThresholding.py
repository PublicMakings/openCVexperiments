import cv2
import numpy as py

img = cv2.imread('images/quartz.jpg')
retval, threshold = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gauze = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 1)



cv2.imshow('original', img)
cv2.imshow('threshold', threshold)
cv2.imshow('gauze', gauze)
cv2.waitKey(0)
cv2.destroyAllWindows
