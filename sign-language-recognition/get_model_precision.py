# -*- coding: utf-8 -*-
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from PIL import ImageFont, ImageDraw, Image



def get_model_precision():
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

    acc_a,acc_b,acc_c,acc_d,acc_e,acc_z,acc_h,acc_th,acc_i,acc_k,acc_l,acc_m,acc_n,acc_ks,acc_o,acc_p \
    ,acc_r,acc_s,acc_t,acc_y,acc_f,acc_x,acc_ps,acc_w = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

    labels_dict = {
        'Α': acc_a,
        'Β': acc_b,
        'Γ': acc_c,
        'Δ': acc_d,
        'Ε': acc_e,
        'Ζ': acc_z,
        'Η': acc_h,
        'Θ': acc_th,
        'Ι': acc_i,
        'Κ': acc_k,
        'Λ': acc_l,
        'Μ': acc_m,
        'Ν': acc_n,
        'Ξ': acc_ks,
        'Ο': acc_o,
        'Π': acc_p,
        'Ρ': acc_r,
        'Σ': acc_s,
        'Τ': acc_t,
        'Υ': acc_y,
        'Φ': acc_f,
        'Χ': acc_x,
        'Ψ': acc_ps,
        'Ω': acc_w
    }

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
                if imgCrop.any():
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

            for index, label in enumerate(labels):
                if label in labels_dict and len(labels_dict[label]) <= 100 and round(prediction[index] * 100, 2) >= 90:
                    labels_dict[label].append( round(prediction[index] * 100, 2) )

        cv2.imshow("Image", imgOutput)
        key = cv2.waitKey(1)

        # Closes the prompt window
        if key == ord("q"):
            total = 0
            for label, acc_list in labels_dict.items():
                print(label + " | " + str(round(sum(acc_list) / len(acc_list),2)) + "%")
                total += round(sum(acc_list) / len(acc_list),2)
            print( "TOTAL | " + str(round(total / 24,2)) )
            break

if __name__ == "__main__":
    get_model_precision()