# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os, os.path
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

def findFace(file_path):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	fa = FaceAligner(predictor, desiredFaceWidth=256)
	image = cv2.imread(file_path)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)


	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		print(shape)
		for (x, y) in shape[36:48]:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	cv2.imshow("Output", image)
	cv2.waitKey(2000)