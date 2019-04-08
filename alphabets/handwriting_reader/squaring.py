import cv2
import numpy as np
#import module to read files from directory
import os 


def squareD():
#    get the files
    letters = os.listdir('./letterz')
    count = 0
#    os.mkdir("./letterz/square/")

    for image in letters:
        img = cv2.imread("./letterz/"+image)
        cv2.imshow('letter',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        i = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if i.shape[0] > i.shape[1]:
            new_blank = np.zeros((i.shape[0],i.shape[0]-i.shape[1]), np.uint8)
            new_blank[:] = (255)
            cv2.imshow('Blank White', new_blank)
            new_image = np.hstack((i,new_blank))
    
        else:
            new_blank = np.zeros((i.shape[1]-i.shape[0],i.shape[1]), np.uint8)
            new_blank[:] = (255)
            cv2.imshow('Blank White', new_blank)
            new_image = np.vstack((i,new_blank))

        cv2.imshow('combined', new_image)

        cv2.imwrite("./letters/square/letter"+str(count)+".jpg",new_image)
        count+=1
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def blankImage(h,w):

    pass
    #blank_image[:,0:width//2] = (255,0,0)      # (B, G, R)
    #blank_image[:,width//2:width] = (0,255,0)




if __name__ == "__main__":
    squareD()
