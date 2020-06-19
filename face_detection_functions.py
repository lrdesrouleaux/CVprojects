# working with arrays
import image_functions
import imutils
from imutils.video import VideoStream
import time
import numpy as np
# using computer vision library
import cv2 as cv
from image_functions import retgray
# facial detection in images and video

# detect faces using deep neural networks in images
# pass in paths to the video,.prototxt file and caffee model, along with the minimum confidence
def dnnfacedetectimg(imagepath, protopath, modelpath, minconfidence):
    # todo change coded defaults to take in terminal arguments
    # reading in dnn
    net = cv.dnn.readNetFromCaffe(
        protopath, modelpath)
    # reading in image
    image = cv.imread(imagepath)
    # records height and with of the picture [:2] means read the list until the 2 index
    (height, width) = image.shape[:2]
    # creates the blob from the image
    blob = cv.dnn.blobFromImage(
        cv.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    # passes the blob into the dnn
    net.setInput(blob)
    # detect faces
    detections = net.forward()
    # draw boxes and label confidence on found faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # if the confidence surpasses our threshold
        if confidence > minconfidence:
            # calculate x-y coordinates of box
            box = detections[0, 0, i, 3:7] * \
                np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            # format the text string that will be displayed
            text = "{:.2f}%".format(confidence * 100)
            # shift down 10 pixels if text is going to go off screen
            y = startY - 10 if startY - 10 > 10 else startY + 10
            # actually draw the rectangle
            cv.rectangle(image, (startX, startY), (endX, endY),
                         (0, 0, 255), 2)
            # display text over the rectangle
            cv.putText(image, text, (startX, y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv.imshow("output", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

# detect faces in a video feed

# detect faces using deep neural networks in video
# pass in paths to the video,.prototxt file and caffee model, along with the minimum confidence
def dnnfacedetectvideo(videopath, protopath, modelpath, minconfidence):
    # opencv video capture method
    vidcap = cv.VideoCapture(videopath)
    # loop while the video stream is opened
    net = cv.dnn.readNetFromCaffe(
        protopath, modelpath)
    while vidcap.isOpened():
        # capture each frame
        ret, frame = vidcap.read()
        # exit if you can't get ret
        if not ret:
            print('Can not get frame exiting')
            break
            # resize each frame
        frame = imutils.resize(frame, width=400)
        # capture height and with from each frame
        (h, w) = frame.shape[:2]
        # make a blob out of the frame
        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300),
                                    (104.0, 177.0, 123.0))
        # feed the blob into the dnn
        net.setInput(blob)
        # capture detections
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # if the confidence surpasses our threshold
            if confidence > minconfidence:
                # calculate x-y coordinates of box
                box = detections[0, 0, i, 3:7] * \
                    np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # format the text string that will be displayed
                text = "{:.2f}%".format(confidence * 100)
                # shift down 10 pixels if text is going to go off screen
                y = startY - 10 if startY - 10 > 10 else startY + 10
                # actually draw the rectangle
                cv.rectangle(frame, (startX, startY), (endX, endY),
                             (0, 0, 255), 2)
                # display text over the rectangle
                cv.putText(frame, text, (startX, y),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv.imshow("output", frame)
        if cv.waitKey(1) == ord('q'):
            break
    vidcap.release()
    cv.destroyAllWindows()


# scans a image and displays all faces found using rectangles using haar cascades


def haarfacedetectimage(imagepath, cascadepath):
    face_cascade = cv.CascadeClassifier(cascadepath)
    image = cv.imread(imagepath)
    gray = retgray(image)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    print("Found {0} faces!".format(len(faces)))
    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.imshow("Faces found", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# scans a video frame by frame and labels faces with rectangles using haar cascades


def haarfacedetectvideo(video, cascadepath):
    face_cascade = cv.CascadeClassifier(cascadepath)
    vidcapture = cv.VideoCapture(video)
    while vidcapture.isOpened():
        ret, frame = vidcapture.read()

        if not ret:
            print("Can't get frame exiting")
            break
        gray = retgray(frame)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    vidcapture.release()
    cv.destroyAllWindows()
