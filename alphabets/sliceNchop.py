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
        #remake countours list to remove small ones
            continue


        cv2.rectangle(img,(x,y),(x+w,y+h),(0,120,120),2)

    showOutput(img)


    ## find some matches
    prevCnt = cnt[0]
    for cnt in contours[420:540]:
        ret = cv2.matchShapes(prevCnt,cnt,1,0.0)
        if ret < .002:
            print('match',str(cnt))
            cv2.drawContours(img, cnt, -2, (255,255,0), 3)
        else:
            print('nope,nope,nope')
    showOutput(img)

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
