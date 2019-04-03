import numpy as np
import cv2
import imutils
import pickle

def splitROI():

#    filename = input("filename?")
    filename = 'alphabet.jpg'

    img = cv2.imread(filename)
    im = imutils.resize(img, width=1600)
    cv2.imshow('resized',im)
    cv2.waitKey(0)
    imGray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imGray,110,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    count = 0
    for cnt in contours: 
        x,y,w,h = cv2.boundingRect(cnt)
        if w < 15 or h < 20:
            continue
        rois.append([x,y,w,h])
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,120,120),2)
        imCrop = imGray[y:y+h, x:x+w]

#could just crop to array of imgs
        cv2.imwrite('./letters/'+str(count)+'.jpg',imCrop)
        count+=1




    cv2.imshow('roi 20', cropped)

    cv2.waitKey(0)
    
    ##compare rois


    cv2.imshow("rects",im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    splitROI()
