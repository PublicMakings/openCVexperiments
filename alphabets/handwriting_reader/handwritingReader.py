import numpy as np
import cv2
import imutils
import pickle
import cnnCharacter as cnn
import os

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

    count = 0
    mrgn = 5
    for cnt in contours: 
        x,y,w,h = cv2.boundingRect(cnt)
        if w < 15 or h < 20:
            continue
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,120,120),2)
        imCrop = imGray[y-mrgn:y+h+mrgn, x-mrgn:x+w+mrgn]
        cv2.imwrite("./letterz"+str(count)+".jpg",imCrop)
        count+=1

    cv2.imshow("rects",im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#def interpretNsort(image):
def do_ocr():
   #from flask app 

    """Add two numbers server side, ridiculous but well..."""
    app.logger.debug("Accessed _do_ocr page with image data")
    # flash('Just hit the _add_numbers function')
    # a = json.loads(request.args.get('a', 0, type=str))
    data = request.args.get('imgURI', 0, type=str)
    app.logger.debug("Data looks like " + data)
    index = request.args.get('index', 0, type=int)
    vocab = json.loads(request.args.get('vocab',0,type=str))

    pp = img_prep(fn="dataset.txt")
    ocr = LiteOCR(fn="app/model/alpha_weights.pkl")
    char_prediction= ocr.predict(pp.preprocess(data))

    result = "You entered a: " + char_prediction

    app.logger.debug("Recognized a character")
    return jsonify(result=result)



def interpretNsort(iMage):
    '''data is a filename?'''
    img = cv2.imread(iMage)
    colimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("letter",colimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pp = cnn.img_prep(fn="dataset.txt")
    ocr = cnn.LiteOCR(fn="alpha_weights.pkl")
    char_prediction= ocr.predict(pp.preprocess(iMage))

    print(char_prediction)

if __name__ == "__main__":
#    splitROI()
#    interpretNsort("./letterz/letterz40.jpg")
    letters = os.listdir("./letters/square")
    for letter in letters:
        interpretNsort("./letters/square/"+letter)
