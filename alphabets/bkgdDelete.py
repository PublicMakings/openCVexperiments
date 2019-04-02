import cv2
import numpy as np


def imageCrop(file,scale):
    '''reads in a file and returns

        scaled down and cropped image
    '''
    im = cv2.imread(file)
    
    scale_percent = scale
    width = int(im.shape[1] * scale_percent/100)
    height = int(im.shape[0] * scale_percent/100)
    dim = (width, height)
    img = cv2.resize(im, dim, cv2.INTER_AREA)
    
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
    
    
    ret, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    #cv2.imshow('binary',thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[1]
    #print(cnt)
    
    x,y,w,h = cv2.boundingRect(cnt)
    
#    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),5) #show the crop
    cropped = img[y:y+h,x:x+w]
    
    #cv2.imshow('a title',img)
    #cv2.imshow('cropped',cropped)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    imgray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    return imgray

if __name__ == "__main__":

    file = 'alphabet.jpg'    
    imageCrop(file,20)
