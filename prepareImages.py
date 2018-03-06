import numpy as np
import cv2
import os, os.path
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib

def prepareImages(path):
    IMAGE_DIR = path  # specify your path here
    NEW_FOLDER = 'edited_images/' + IMAGE_DIR
    DELAY_MILISECONDS = 1

    print(NEW_FOLDER)
    image_path_list = []
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]  # specify your vald extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # create a list all files in directory and
    # append files with a vaild extention to image_path_list
    for file in os.listdir(IMAGE_DIR):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(IMAGE_DIR, file))

    # loop through image_path_list to open each image
    i = 1
    for imagePath in image_path_list:
        image = cv2.imread(imagePath)
        # display the image on screen with imshow()
        # after checking that it loaded
        if image is not None:
            img = cv2.imread(imagePath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = imutils.resize(img, width=800)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 2)

            for rect in rects:
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                (x, y, w, h) = rect_to_bb(rect)
                img = fa.align(img, gray, rect)

                # display the output images
            if not os.path.exists(NEW_FOLDER):
                os.makedirs(NEW_FOLDER)

            print(imagePath)
            file_name = NEW_FOLDER + imagePath.split('/')[2]
            print(file_name)
            cv2.imwrite(file_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(DELAY_MILISECONDS)
            print(i)
            i += 1

    cv2.destroyAllWindows()