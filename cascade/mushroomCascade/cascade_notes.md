# How to Train a Haar Cascade

## Outline based on Sentdex

* get thousands of negative images w/o object
* get thousands of images w/ object or manually create them
    * image net {like wordnet}
* create a positive vector file by stitching together all positives
* train cascade  
---
the negative and postive images need description files
usually bg.txt containing pth to image by line. \n
positive images often called _info_ or _pos.txt_ 
contains path to images, how many objects and where they are located.
    * image, num objects, start point, rectangle coordinates

---
negative images should generally be larger than positive

gemerally small images 100X100 for negatives 50x50 for positives...
 when training the images will be even smaller.

~ double the amount of positive images to negative for training



#### specifics for setndex

* [sentdex tutorial](https://www.youtube.com/watch?v=jG3bu0tjFbk&list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq&index=17)
    has positive image is 50x50 pixels
* creating samples by placing positive images on negative images
