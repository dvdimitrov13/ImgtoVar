## This classifiers identifies inverted images due to the way they have been scrapped from pdf files,
## trained on images from pdfs, use with caution for other contexts
import os
import gdown
import numpy as np
from imgtovar.utils import functions

from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D


def loadModel(
    url="https://github.com/dvdimitrov13/thesis_modelzoo/releases/download/v.1.0.0/nature_cls.h5",
):

    ## Define home
    home = functions.get_imgtovar_home()

    if os.path.isfile(home + "/.imgtovar/weights/nature_cls.h5") != True:
        print("nature_cls.h5 will be downloaded...")

        output = home + "/.imgtovar/weights/nature_cls.h5"
        gdown.download(url, output, quiet=False)
    
    ## Rebuild model
    base_model = applications.ResNet50V2(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))

    for layer in base_model.layers:
        layer.trainable=False

    input = Input(shape=(224, 224, 3), name = 'image_input')
    output_base = base_model(input)

    x = BatchNormalization()(output_base)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(3, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)

    print(model.summary())

    model.load_weights(
        home + "/.imgtovar/weights/nature_cls.h5"
    )
    # print(model.summary())

    return model
