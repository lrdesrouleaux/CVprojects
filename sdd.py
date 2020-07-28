from configure import sddcongfig as config
from detectpeople import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import time
import cv2 as cv
import os
import imutils
# parse arguments passed and sets default parameters for the models and files
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="pedestrians.mp4",
                help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="output.avi",
                help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not output frame should be displayed")
args = vars(ap.parse_args())
labelspath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelspath).read().strip().split("\n")

wpath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configpath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
print("[INFO] loading YOLO from disk...")
net = cv.dnn.readNetFromDarknet(configpath, wpath)

ln = net.getLayerNames()
ln = [ln[i[0]-1]for i in net.getUnconnectedOutLayers()]
print("[INFO] accessing video stream...")
vs = cv.VideoCapture(args["input"])
# writer is used to write to the ouput file
writer = None
while True:
    # grab the frame
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
        # resize the image~
    frame = imutils.resize(frame, width=700)
    #detect people using yolov3 
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
    #initialize a empty set of violaters
    violate = set()
    #if there are more than 2 people calculate their euclidean distance
    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                #if they're both within the minimum distance add them to the violater set
                if D[i, j] < config.MIN_DISTANCE:
                    violate.add(i)
                    violate.add(j)
    #color non violaters green                
    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startx, starty, endx, endy) = bbox
        (cx, cy) = centroid
        color = (0, 255, 0)
    #color violaters red
        if i in violate:
            color = (0, 0, 255)
        #add rectangles to all
        cv.rectangle(frame, (startx, starty), (endx, endy), color, 2)
        #add circle aroung their centroids
        cv.circle(frame, (cx, cy), 5, color, 1)
    #print number of violations on screen
    text = "Social Distancing Violations: {}".format(len(violate))
    cv.putText(frame, text, (10, frame.shape[0] - 25),
               cv.FONT_HERSHEY_SIMPLEX, 0.86, (0, 0, 255), 3)
    if args["display"] > 0:
        cv.imshow("Frame", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
            #write to output file
    if args["output"] != "" and writer is None:
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter(args["output"], fourcc, 25,
                                (frame.shape[1], frame.shape[0]), True)
    if writer is not None:
        writer.write(frame)
