import os
import subprocess
import pytesseract
from PIL import Image
import cv2

letters = os.listdir("./letters/square")
#dir = "home/ayo/repos/cv/alphabets/handwriting_reader/letters/square/"
dir = "./letters/square/"
print(letters)
ltrs=[]
for letter in letters:
   im = cv2.imread(dir+letter)
   ret, thresh = cv2.threshold(im, 100,255,0)
   cv2.imshow('letter', thresh)
   ltr = pytesseract.image_to_string(thresh)
   ltrs+=ltr
print(ltrs)
cv2.waitKey(0)
cv2.destroyAllWindows() 
