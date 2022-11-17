import os
import gdown
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, Activation
from imgtovar.utils import functions

## This model reconstruction works for the heavier .h5 formatted model saves I, the deep face models should use their own architecture
## I need to add a step to download this in the right place


def loadModel(
    url="https://github.com/dvdimitrov13/thesis_modelzoo/releases/download/v.1.0.0/age_cls.h5",
):

    # Replace code snippet with new download_asset in functions
    home = functions.get_imgtovar_home()

    if os.path.isfile(home + "/.imgtovar/weights/age_cls.h5") != True:
        print("age_cls.h5 will be downloaded...")

        output = home + "/.imgtovar/weights/age_cls.h5"
        gdown.download(url, output, quiet=False)
    ############################################################

    #VGG-Face model
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Convolution2D(101, (1, 1), name='predictions'))
    model.add(Flatten())
    model.add(Activation('softmax'))

    base_model_output = Sequential()
    base_model_output = Dropout(0.5)(model.layers[-4].output)
    base_model_output = Convolution2D(5, (1, 1), name='predictions')(base_model_output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)   

    age_model = Model(inputs=model.input, outputs=base_model_output)

    age_model.load_weights(
        home + "/.imgtovar/weights/age_cls.h5"
    )

    return age_model
