import cv2
import numpy as np
import time
import os
from livestreamer import Livestreamer
from matplotlib import pyplot as plt



fourcc = cv2.VideoWriter_fourcc(*'XVID')

bitOut = cv2.VideoWriter('./outputs/bits.avi',fourcc, 20.0, (1280,720))
subOut = cv2.VideoWriter('./outputs/subs.avi',fourcc, 20.0, (1280,720))
donOut = cv2.VideoWriter('./outputs/donations.avi',fourcc, 20.0, (1280,720))
notOut = cv2.VideoWriter('./outputs/nothing.avi',fourcc, 20.0, (1280,720))
overallOut = cv2.VideoWriter('./outputs/everything.avi',fourcc, 20.0, (1280,720))
outlineOverallOut = cv2.VideoWriter('./outputs/outlineEverything.avi',fourcc, 20.0, (1280,720))

subscriberTemplate = cv2.imread('./templates/tsmMythSubscriber.png',0)
donationTemplate = cv2.imread('./templates/tsmMythDonation2.png',0)
bitsTemplate = cv2.imread('./templates/bits.png',0)

def main():
    session = Livestreamer()
    session.set_option("http-headers", "Client-ID=jzkbprff40iqj646a697cyrvl0zt2m6")
    streams = session.streams("twitch.tv/tsm_myth")
    stream = streams['720p60']
    fname = "downloading.mpg"
    vid_file = open(fname,"wb")
    fd = stream.open()
    new_bytes = 0

    for i in range(0,8*1024):
        new_bytes = fd.read(2048)
        vid_file.write(new_bytes)
    print "Done buffering."

    startReadingFrames(fname, vid_file, fd)


def startReadingFrames(filename, videoFile, streamData):
    cam = cv2.VideoCapture(filename)
    while True:
        ret, img = cam.read()
        try:
            if ret:
                threshold = 0.95
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

        videoFile.write(streamData.read(1024*16))
        time.sleep(0.05)


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
    cv2.imwrite('res.png',img)
    return

def getROI(img):
    r = (596, 103, 683, 64)
    imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.imwrite('crop.png',imCrop)
    processOCR(imCrop)
    print r


def processOCR(img):
    print "pre ocr"
    im = Image.open("crop.png")
    text = pytesseract.image_to_string(im)
    print text
    cv2.waitKey()

if __name__ == "__main__":
    main()
