from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2 as cv
import os
ap=argparse.ArgumentParser()
ap.add_argument("-m", "--model",default="vggbird.model" ,required=False,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin",default="lb.pickle" ,required=False,
	help="path to label binarizer")
ap.add_argument("-i", "--image",default="/Users/lens/Downloads/f.webp" ,
            required=False,
	help="path to input image")
args = vars(ap.parse_args())
image=cv.imread(args["image"])
output=image.copy()

image=cv.resize(image,(96,96))
image=image.astype("float")/255.0
image=img_to_array(image)
image=np.expand_dims(image,axis=0)
print("<info> loading model...")
model=load_model(args["model"])
lb=pickle.loads(open(args["labelbin"],"rb").read())
print("<info> classifying image")
proba=model.predict(image)[0]
idx=np.argmax(proba)
label=lb.classes_[idx]
filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
output = imutils.resize(output, width=400)
cv.putText(output, label, (10, 25),  cv.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
print("[INFO] {}".format(label))
cv.imshow("Output", output)
cv.waitKey(0)