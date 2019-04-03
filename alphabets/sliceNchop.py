import numpy as np
import cv2
import bkgdDelete as bkd

import argparse
#import glob #some sorta file path module


imgfile = 'alphabet.jpg'

def main():
    ## open image, grayscale, and reduce size
    ## find some contours

    im = bkd.imageCrop(imgfile, 20)
    full = cv2.imread(imgfile)
    im2 = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
    
    roi = im2[400:600,1100:1300]
    cv2.imshow('a?',roi)
   ## remake img to color

    img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
 
    #kernel = np.ones((5,5),np.float32)/25 #not in use
    #dst = cv2.filter2D(im,-1,kernel) #not in use
    dst = cv2.GaussianBlur(im,(7,7),0)
#    ret,thresh = cv2.threshold(dst,100,255,cv2.THRESH_BINARY)
#    cv2.imshow('threshed', thresh)
#    cv2.imshow('blurred',dst)
    edged = cv2.Canny(roi,112,255)

#    cv2.imshow('canny',edged)
    ret, thresh = cv2.threshold(im, 100, 255, 0)
    edged2 = cv2.Canny(thresh, 0,10)
    cv2.imshow('edged after threshold',edged2)
    cv2.imshow('thresholded',thresh)


# adjust to see if I can find better matches

    contours, hierarchy = cv2.findContours(edged2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# testing out some contours
    print(contours[2:5])
    for cnt in contours[2:5]:
        cv2.drawContours(img, cnt, 4,(0,255,200),4)
    showOutput(img)


    for cnt in contours:
        cv2.drawContours(img, cnt, -1,(0,150,200),3)
    showOutput(img)
    ## make some bounding boxes
    
    cnts = [] #make array for good contours
    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w<5 or h <10:
        #remake countours list to remove small ones
            continue
        if w>im.shape[1]/3 or h>im.shape[0]/3:
            continue
        cnts.append(cnt)
        cv2.drawContours(img,cnt,-1,(150,190,255),2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,120,120),2)

    showOutput(img)


    ## find some matches
'''    for c1 in cnts:
        cv2.drawContours(img, c1, -1, (150,10,255), 1)
        for c2 in cnts:
            ret = cv2.matchShapes(c1,c2,1,0.0)
            if ret < 1:
#                print('match',ret)
                cv2.drawContours(img, c2, -1, (255,255,0), 2)
        else:
#            print('nope,nope,nope')
            cv2.drawContours(img, c2,-1,(100,250,20), 2)
    showOutput(img)
'''

def auto_canny(image, sigma=0.33):
    '''determins upper and lower image thresholds
        from https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    '''
        
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def makeVectors():
    '''utilize a mix of find outermost points and offset from contours 
       to create vectors of handwriting'''
    pass
    

def OCR():
    '''train ocr from continually adding handwriting'''
    pass


def showOutput(image):

    cv2.imshow("a picture of...", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
