import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('quartz.jpg')
cv2.imshow('a crystal',img)
cv2.waitKey(0)
cv2.destroyAllWindows

mask = np.zeros(img.shape[:2], np.uint8)

bgmodel = np.zeros((1,65), np.float64)
fgmodel = np.zeros((1,65), np.float64)

print(bgmodel)

rect = (100,400,50,80)

cv2.grabCut(img, mask, rect, bgmodel, fgmodel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img = img*mask2[:,:,np.newaxis]


plt.imshow(img)
plt.colorbar()
plt.show()
