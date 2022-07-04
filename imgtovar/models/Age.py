import os
import gdown
import numpy as np
from tensorflow import keras
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

    reconstructed_model = keras.models.load_model(
        home + "/.imgtovar/weights/age_cls.h5"
    )

    # print(reconstructed_model.summary())

    return reconstructed_model
