
import cv2
import numpy as np


def templateMatching():
    pass

template = cv2.imread('a.png', 0)
img = cv2.imread('alphabet.jpg')

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

w, h = template.shape[::-1]

res = cv2.matchTemplate(imgGray, template, cv2.TM_CCOEFF_NORMED)
threshold= 0.5
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (200,0,225), 2)


#laplacian = cv2.Laplacian(img, cv2.CV_64F)
#sobelx = cv2.Sobel(img, cv2.CV_64F, 1,0, ksize=5)
#sobely = cv2.Sobel(img, cv2.CV_64F, 0,1,ksize=5)
#edges = cv2.Canny(img, 150,200 )


#cv2.imshow('boof',laplacian)
#cv2.imshow('sobel2',sobelx)
#cv2.imshow('sobel',sobely)
#cv2.imshow('outline',edges)
cv2.imshow('detected', img)

cv2.waitKey(0)
cv2.destroyAllWindows


