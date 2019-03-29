#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

img= cv2.imread('images/mushroom.jpg',cv2.IMREAD_COLOR)
#IMREAD_GRAYSCALE = 0
#IMREAD_COLOR = 1
#IMREAD_UNCHANGED = -1


### let's do some stuff to manipulate pixels
def manipulatePixels(img):
    img[5,5]=[255,255,0]
    
    #Region of Image
    img[30:500, 40:200] = [255,200,50]
    
    mushroomROI = img[37:111, 107:194]
    img[0:74,0:87]=mushroomROI
    
    
    


def pltOutput(img):
    ### output in plt
    ### because cv2 is BGR the image is inverted color layerwise
    #plt.imshow(img,cmap='gray', interpolation='bicubic')
    #plt.imshow(img, interpolation='bicubic')
    #plt.plot([5,100],[70,100],'c',linewidth=5)
    #plt.show()
    pass
    #savethe image

def makeSomeShapes(img):
    cv2.line(img,(0,0), (150,150),(205,40,120),15)
    cv2.rectangle(img, (15,15),(200,200), (100,20,255),5)
    cv2.circle(img, (90,400), 30, (0,0,10), -1)

    pts = np.array([[300,300],[100,30],[50,400]], np.int32)
    #pts = pts.reshape((-1,1,2))
    cv2.polylines(img, [pts], True, (100,255,80),2)

def makeSomeTxt(img):
    font= cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'woooordszzz', (10,50), font, 1.2, (80,80,200), 2, cv2.LINE_AA)

def saveAlot(img):
    cv2.imwrite('images/graybody.png', img)

if __name__ == "__main__":
    makeSomeShapes(img)
    makeSomeTxt(img)
    cv2.imshow('fruiting bodies', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
