try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import os

def tesseract():
    # If you don't have tesseract executable in your PATH, include the following:
    #pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_exec$
    # Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
    
    # Simple image to string
    
    #.encode('utf8') #add to print statement to make encoding work
    
    print(pytesseract.image_to_string(Image.open(imgFile)).encode('utf8'))
    
    # Get bounding box estimates
    print(pytesseract.image_to_boxes(Image.open(imgFile)))
    
    # Get verbose data including boxes, confidences, line and page numbers
    print(pytesseract.image_to_data(Image.open(imgFile)))
    
    # Get information about orientation and script detection
    print(pytesseract.image_to_osd(Image.open(imgFile)))
    
    # In order to bypass the internal image conversions, just use relative or ab$
    # NOTE: If you don't use supported images, tesseract will return error
    print(pytesseract.image_to_string(imgFile))
    
    # get a searchable PDF
    pdf = pytesseract.image_to_pdf_or_hocr(imgFile, extension='pdf')
    
    # get HOCR output
    hocr = pytesseract.image_to_pdf_or_hocr(imgFile, extension='hocr')

letters = os.listdir("./letters/square")

#def reader(images):
for letter in letters:
    print(pytesseract.image_to_string(Image.open(letter).encode('utf8'))

#if __name__ == "__main__":
#reader(letters)
