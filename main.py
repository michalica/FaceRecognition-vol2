from prepareImages import prepareImages
from facial_landmarks import findFace
import os, os.path
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib

def main():
    dir_path_list = []
    new_dir_path_list = []
    PROCCES_IMAGES = False
    DETECT_FACES = False

    prepareImages('gt_db/ja/')
    if PROCCES_IMAGES:
        for dir in os.listdir('gt_db/'):
            extension = os.path.splitext(dir)[1]
            dir_path_list.append(os.path.join('gt_db/', dir))

        for path in dir_path_list:
            print(path + '/')
            prepareImages(path + '/')

    if DETECT_FACES:
        for dir in os.listdir('edited_images/gt_db/'):
            extension = os.path.splitext(dir)[1]
            new_dir_path_list.append(os.path.join('gt_db/', dir))
        print(new_dir_path_list)
        for dir in new_dir_path_list:
            for file in os.listdir(dir):
                file_path = 'edited_images/' + dir + '/' + file;
                print(file_path)
                findFace(file_path)

if __name__ == "__main__":
    main()
