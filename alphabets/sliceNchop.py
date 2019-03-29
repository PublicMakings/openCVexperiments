[0]
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
    
   

    ## remake img to color

    img = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
    imgSM = cv2.resize(img, dim, cv2.INTER_AREA)

 ## crop to outermost  hierarchy
    
    
    #get the coordinates of outermost hierarchy

    hierarchy = hierarchy[0]    

    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        x,y,w,h = cv2.boundingRect(currentContour)
        if currentHierarchy[2] < 0:
            #inner elements
            cv2.rectangle(imgSM,(x,y),(x+w,y+h),(255,0,130),2)
        elif currentHierarchy[3] < 0:
            cv2.rectangle(imgSM,(x,y),(x+w,y+h),(0,200,130),2)

    #cropGray = imGray[y:y+h, x:x+w].copy() # makes a copy


    showOutput(imgSM)
#    cv2.rectangle(img)





    ## make some bounding boxes

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
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
