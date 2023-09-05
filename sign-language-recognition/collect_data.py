import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

folder = "Data/A"

def collect_data():
    counter = 0
    cap = cv2.VideoCapture(0)
    # Creating an instance of the HandDetector class with the parameter maxHands set to 1. 
    # This will allow detection and tracking of a single hand.
    detector = HandDetector(maxHands=1)

    # Sets the offset value used for cropping the hand region.
    offset = 20
    # Sets the size of the square image used for classification.
    imgSize = 300
    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                if imgCrop.any():
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                if imgCrop.any():
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize


        cv2.imshow("Image", img)
        key = cv2.waitKey(1)

        # Saves the image on selected folder
        if key == ord("s"):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
            print(counter)

        # Closes the prompt window
        if key == ord("q"):
            break

if __name__ == "__main__":
    collect_data()