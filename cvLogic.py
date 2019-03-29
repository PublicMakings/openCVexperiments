
#!/usr/bin/env python3
import cv2
import numpy as np


#images should be same size

img1= cv2.imread('images/mushroom.jpg',cv2.IMREAD_COLOR)
img2 = cv2.imread('images/quartz.jpg', cv2.IMREAD_COLOR)
#added = img1 + img2 
#added = cv2.add(img1,img2) #adds pixel values
#weighted = cv2.addWeighted(img1, 0.6, img2 0.4, 0) 


rows,cols,channels = img1.shape
roi = img2[0:rows,0:cols]

img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img1gray, 120, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask) ## bitwise not

img2_bg = cv2.bitwise_and(roi,roi, mask=mask_inv) ## bitwise and
img1_fg = cv2.bitwise_and(img1, img1, mask=mask)

dst = cv2.add(img1_fg, img2_bg)
img2[0:rows,0:cols] = dst



cv2.imshow('result.....', img2)



cv2.waitKey(0)
cv2.destroyAllWindwos()

