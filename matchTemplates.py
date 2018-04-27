import cv2
import numpy as np
import time
import os

from PIL import Image
import pytesseract




fourcc = cv2.VideoWriter_fourcc(*'XVID')

bitOut = cv2.VideoWriter('./outputs/bits.avi',fourcc, 20.0, (1280,720))
subOut = cv2.VideoWriter('./outputs/subs.avi',fourcc, 20.0, (1280,720))
donOut = cv2.VideoWriter('./outputs/donations.avi',fourcc, 20.0, (1280,720))
notOut = cv2.VideoWriter('./outputs/nothing.avi',fourcc, 20.0, (1280,720))
overallOut = cv2.VideoWriter('./outputs/everything.avi',fourcc, 20.0, (1280,720))
outlineOverallOut = cv2.VideoWriter('./outputs/outlineEverything.avi',fourcc, 20.0, (1280,720))

subscriberTemplate = cv2.imread('./templates/tsmMythSubscriber.png',0)
donationTemplate = cv2.imread('./templates/tsmMythDonation.png',0)
bitsTemplate = cv2.imread('./templates/bits.png',0)

ROI = (865, 61, 141, 115)

def main(first):
    startReadingFrames("myth.mpg", first)

def startReadingFrames(filename, first):

    cam = cv2.VideoCapture(filename)
    r = 0
    while True:
        ret, img = cam.read()
        try:
            if ret:
                threshold = .95
                findTemplate(threshold, img)
        except:
            print "buggin"
            continue

            if (0xFF & cv2.waitKey(5) == 27):
                out.release()
                cv2.destroyAllWindows()
                break
            if(img.size == 0):
                continue




def findTemplate(threshold, ogimg):
    img = cv2.cvtColor(ogimg, cv2.COLOR_BGR2GRAY)
    cv2.imshow("ogToGray", img)
    res = cv2.matchTemplate(img,subscriberTemplate,cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    if(len(loc[0]) == 0):
        res = cv2.matchTemplate(img,donationTemplate,cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        if(len(loc[0]) == 0):
            res = cv2.matchTemplate(img,bitsTemplate,cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            if(len(loc[0]) == 0):
                writeToFile(img, 0, "nothing")
                return -2;
            else:
                writeToFile(img, loc, "bits")
                return 1;
        else:
            getROI(ogimg);
            writeToFile(img, loc, "donation")
            return 1
    else:
        writeToFile(img, loc, "subscriber")
        return 1
    return -1;



def findGoodImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    _, cnts, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 100:
            break

    ## (4) Create mask and do bitwise-op
    mask = np.zeros(img.shape[:2],np.uint8)
    cv2.drawContours(mask, [cnt],-1, 255, -1)
    dst = cv2.bitwise_and(img, img, mask=mask)

    ## Save it
    #cv2.imshow("mask.png", mask)
    #cv2.imshow("dst.png", dst);
    cv2.imshow("gray.png", gray)
    cv2.waitKey()

def getROI(img):
    r = cv2.selectROI(img)
    imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    print r
    first = False
    #cv2.imshow("1Image", imCrop)
    cv2.waitKey(0)
    cv2.imwrite('crop.png',imCrop)
    processOCR(imCrop)
    print r


def processOCR(img):
    print "pre ocr"
    im = Image.open("crop.png")
    text = pytesseract.image_to_string(im)
    print text
    cv2.waitKey()

def writeToFile(image, loc, label):
    print label
    img = image
    if(label == 'nothing'):
        notOut.write(img)
    elif(label == 'donation'):
        w, h = donationTemplate.shape[::-1]
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        donOut.write(img)
    elif(label == 'bits'):
        w, h = bitsTemplate.shape[::-1]
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        bitOut.write(img)
    else:
        w, h = subscriberTemplate.shape[::-1]
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        subOut.write(img)

    overallOut.write(img)

    img = cv2.Canny(image, threshold1 = 200, threshold2 = 300)
    try:
        outlineOverallOut.write(img)
    except Exception as e:
        raise e
    cv2.imwrite('res.png',img)
    return


if __name__ == "__main__":
    main(str)