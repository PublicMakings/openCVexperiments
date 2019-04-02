import numpy as np
import cv2
import bkgdDelete as bkd


#import argparse
#import glob


imgfile = 'alphabet.jpg'

def main():
    ## open image, grayscale, and reduce size
    ## find some contours

    im = bkd.imageCrop(imgfile, 20)
    
    kernel = np.ones((5,5),np.float32)/25
    #dst = cv2.filter2D(im,-1,kernel)
    dst = cv2.GaussianBlur(im,(5,5),0)
    cv2.imshow('blurred',dst)
    edged = cv2.Canny(dst,112,255)

#    img = auto_canny(im)
#    showOutput(img)
    cv2.imshow('canny',edged)
    ret, thresh = cv2.threshold(im, 112, 255, 0)
#    cv2.imshow('thresholded',thresh)


# adjust to see if I can find better matches

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ## remake img to color

    img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    
    ## make some bounding boxes

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w<5 or h <10:
        #remake countours list to remove small ones
            continue
        if w>im.shape[1]/3 or h>im.shape[0]/3:
            continue

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,120,120),2)

    showOutput(img)


    ## find some matches
    prevCnt = cnt[0]
    for c1 in contours:
        for c2 in contours:
            ret = cv2.matchShapes(c1,c2,1,0.0)
            if ret < .01:
                print('match',ret)
                cv2.drawContours(img, cnt, -2, (255,255,0), 3)
#        else:
            #print('nope,nope,nope')
    showOutput(img)


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
