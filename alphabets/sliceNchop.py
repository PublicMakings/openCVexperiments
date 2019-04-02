[0]
import numpy as np
import cv2
import bkgdDelete as bkd

imgfile = 'alphabet.jpg'

def main():
    ## open image, grayscale, and reduce size
    ## find some contours

    im = bkd.imageCrop(imgfile, 20)
#    showOutput(im)
    

    ret, thresh = cv2.threshold(im, 127, 255, 0)
#    showOutput(thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ## remake img to color

    img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    
    ## make some bounding boxes

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w<5 or h <10:
            continue
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,120,120),2)

    showOutput(img)


    ## find some matches
    
'''    for cnt in contours:
        
    #### Syntax cv2.matchShapes(InputArray 1, InputArray 2, 
    ####                       int method, double parameter)
    ret = cv2.matchShapes(contours1, contours2, 1, 0.0)
'''

    #cv2.drawContours(imgraySM, contours, -2, (255,255,0), 3)


def showOutput(image):

    cv2.imshow("a picture of...", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
