import numpy as np
import cv2

file = 'alphabet.jpg'

def main():
    ## open image, grayscale, and reduce size
    im = cv2.imread(file)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    scale_percent = 30
    width = int(im.shape[1] * scale_percent/100)
    height = int(im.shape[0] * scale_percent/100)
    dim = (width, height)
    imgraySM = cv2.resize(imgray, dim, cv2.INTER_AREA)

    showOutput(imgraySM)


    ## find some contours


    ret, thresh = cv2.threshold(imgraySM, 127, 255, 0)
    showOutput(thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    ## make some bounding boxes

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(imgraySM,(x,y),(x+w,y+h),(0,120,120),2)


    #cv2.drawContours(imgraySM, contours, -2, (255,255,0), 3)
    showOutput(imgraySM)    


def showOutput(image):

    cv2.imshow("a picture of...", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
