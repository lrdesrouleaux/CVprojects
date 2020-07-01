import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as k
class smallervggnet():
    @staticmethod

    def build(width,height,depth,classes):
        #initialize model and input shape
        model=Sequential()
        inputshape=(height,width,depth)
        chandim=-1
        #change formation if using channels first
        if k.image_data_format=="channels_first":
            inputshape=(depth,height,width)
            chandim=1
        #32 filters with a 3*3 kernal
        model.add(Conv2D(32,(3,3),padding="same",input_shape=inputshape))
        #using rectifier activation function
        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chandim))

        model.add(MaxPooling2D(pool_size=(3,3)))
        #using dropout to build in redudancy
        model.add(Dropout(.25))
        #64 filters with a 3*3 kernal
        model.add(Conv2D(64,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chandim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chandim))
        #reduce pooling to ensure I don't reduce dimensionality too quickly
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        #128 filters with a 3*3 kernal
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chandim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chandim))
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #reduce overfitting with dropout again
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        print("<info> model compiled returning")
        return model
