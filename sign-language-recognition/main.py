# -*- coding: utf-8 -*-
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from PIL import ImageFont, ImageDraw, Image



def model():
    cap = cv2.VideoCapture(0)
    # Creating an instance of the HandDetector class with the parameter maxHands set to 1. 
    # This will allow detection and tracking of a single hand.
    detector = HandDetector(maxHands=1)
    # Creating an instance of the Classifier class, 
    # specifying the path to a pre-trained Keras model file for image classification.
    classifier = Classifier("Model/keras_model.h5")

    # Sets the offset value used for cropping the hand region.
    offset = 20
    # Sets the size of the square image used for classification.
    imgSize = 300

    # Creates a list of labels corresponding to the classes predicted by the classifier.
    labels = ['Α','Β','Γ','Δ','Ε','Ζ','Η','Θ','Ι','Κ','Λ','Μ','Ν','Ξ','Ο','Π','Ρ','Σ','Τ','Υ','Φ','Χ','Ψ','Ω']

    while True:
        success, img = cap.read()
        imgOutput = img.copy()
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
                    # Calls the getPrediction method of the Classifier class
                    # to classify the hand region image (imgWhite).
                    # It returns the prediction probabilities for all classes and
                    # the index of the predicted class with the highest probability.
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                # Calls the getPrediction method of the Classifier class
                # to classify the hand region image (imgWhite).
                # It returns the prediction probabilities for all classes and
                # the index of the predicted class with the highest probability.
                prediction, index = classifier.getPrediction(imgWhite, draw=False)


            cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                          (x - offset+190, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x + w+offset, y + h+offset), (255, 0, 255), 4)

            # Using Arial font in order to display Greek Letters
            # and shows the predicted letter
            fontpath = "Font/arial.ttf"
            font = ImageFont.truetype(fontpath, 32)
            img_pil = Image.fromarray(imgOutput)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x, y-60), str(labels[index]) + " " + str(round(prediction[index] * 100,2)) + " %" , font=font, fill=(255,255,255))
            imgOutput = np.array(img_pil)

        cv2.imshow("Image", imgOutput)
        key = cv2.waitKey(1)

        # Closes the prompt window
        if key == ord("q"):
            break

if __name__ == "__main__":
    model()