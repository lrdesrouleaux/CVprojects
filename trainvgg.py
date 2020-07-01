import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#using agg backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
#by tansforming data through rotations shearing etc.
#I can generate more training data with a lower amount of pictures
from keras.preprocessing.image import ImageDataGenerator
#optimizer method
from keras.optimizers import Adam
#self explanatory
from keras.preprocessing.image import img_to_array
#allows human readable classes and encodes them into vectors 
# and transform int class labels back to human readable
from sklearn.preprocessing import LabelBinarizer
#create training and testing splits
from sklearn.model_selection import train_test_split
#my model
from smallervgg.smallervggnet import smallervggnet
#already used everything else
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2 as cv
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",default="/Users/lens/Documents/CVproject/dataset"
 ,required=False,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", default="vggbird.model",required=False,
	help="path to output model")
ap.add_argument("-l", "--labelbin",default="lb.pickle",required=False,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())
#initialize number of epochs and learning rate
EPOCHS=100
INIT_LR=1E-3
BS=32
IMAGE_DIMS=(96,96,3)

data=[]
labels=[]
print("<info> loading images...")
imagepaths=sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagepaths)
for imagepath in imagepaths:
    #input dataset images into the data
    image=cv.imread(imagepath)
    image=cv.resize(image,(IMAGE_DIMS[1],IMAGE_DIMS[0]))
    image=img_to_array(image)
    data.append(image)
    #find class label from the image path and update the list
    label=imagepath.split(os.path.sep)[-2]
    labels.append(label)
data=np.array(data,dtype="float")/255.0
labels=np.array(labels)
print("<info> data matrix: {:.2f}MB".format(data.nbytes/(1024*1000)))
#binarize the labels
lb=LabelBinarizer()
labels=lb.fit_transform(labels)

#part data into training and testing at a 80-20 ratio
(trainx,testx,trainy,testy)=train_test_split(data,labels,test_size=.2,random_state=42)
#fabricate image generator
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
print("<info> compiling model...")
model=smallervggnet.build(width=IMAGE_DIMS[1],
height=IMAGE_DIMS[0],depth=IMAGE_DIMS[2],classes=len(lb.classes_))
opt=Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
#train the network
print("<info> training network...")
H=model.fit_generator(
    aug.flow(trainx,trainy,batch_size=BS),
    epochs=EPOCHS,verbose=1,validation_data=(testx,testy),
    steps_per_epoch=len(trainx)//BS)
print("<info> serializing network...")
model.save(args["model"])
#save the label bin to disk
print("<info> serializing label binarizer")
f=open(args["labelbin"],"wb")
f.write(pickle.dumps(lb))
f.close()
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])