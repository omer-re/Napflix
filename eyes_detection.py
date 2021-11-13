from operator import itemgetter

import cv2
import winsound
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QMessageBox
import pyttsx3
import sys
import data_base
import datetime
from PyQt5.QtGui import QTextCursor
import numpy as np

flag2 = True

# function to convert dlib.full_object_detection to numpy array
def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords


def check_eyes():
    eye_cascPath = r'haarcascade_eye_tree_eyeglasses.xml'  # eye detect model
    face_cascPath = r'haarcascade_frontalface_alt.xml'  # face detect model
    faceCascade = cv2.CascadeClassifier(face_cascPath)
    eyeCascade = cv2.CascadeClassifier(eye_cascPath)
    flag = True
    #cap = cv2.VideoCapture('http://192.168.0.113:4747/video')
    cap = cv2.VideoCapture(1)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    print(cap.isOpened())
    while flag2:
        ret, img = cap.read()
        cv2.imshow('preview', img)

        if ret:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )
            # print("Found {0} faces!".format(len(faces)))
            if len(faces) > 0:
                #   choose the closest face
                persons = [(x, y, w, h, w * h) for (x, y, w, h) in faces]
                #  Draw a rectangle around the faces
                (x, y, w, h) = max(persons, key=itemgetter(4))[:-1]
                # Draw a rectangle around the faces

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
                frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
                eyes = eyeCascade.detectMultiScale( frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                    # flags = cv2.CV_HAAR_SCALE_IMAGE
                )
                for (ex, ey, ew, eh) in eyes:
                    #cv2.rectangle(frame_tmp, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    cv2.circle(frame_tmp, (int(ex+0.5*ew), int(ey+0.5*eh)), int(0.5*ew), (0, 255, 0), 2, 8, 0)
                    cv2.circle(frame_tmp, (int(ex+0.5*ew), int(ey+0.5*eh)), 3, (150, 0, 150), -1)


                if len(eyes) == 0:
                    flag += 1
                    print('no')
                    if flag > 7:
                        frequency = 2500  # Set Frequency To 2500 Hertz
                        duration = 1000  # Set Duration To 1000 ms == 1 second
                        winsound.Beep(frequency, duration)
                        #speak(gender, volume, "hey, wake up")
                        flag = 0
                else:
                    flag = 0
                    print('yes')

                frame_tmp = cv2.resize(frame_tmp, (400, 400), interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Face Recognition', frame_tmp)

            waitkey = cv2.waitKey(27)
            if waitkey == ord('q') or waitkey == ord('Q'):
                cv2.destroyAllWindows()
                break




def countWarning():
    recentWarnings = []
    #warningsList = data_base.get_user_data(user_id)
    d = datetime.datetime.now() - datetime.timedelta(days = 50)
    # for w in warningsList:
    #     w = datetime.datetime.fromisoformat(w)
    #     if w > d:
    #         recentWarnings.append(w)
    return 10
    #return len(recentWarnings)



if __name__ == "__main__":
    print('start')
    check_eyes()
    print('end')




