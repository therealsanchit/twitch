from PIL import Image
from pytesseract import image_to_string


print "pre ocr"
im = Image.open("crop.png")
im.show()
try:
    text = image_to_string(im)
    print text
except Exception as e:
    raise e