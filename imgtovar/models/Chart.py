import os
import gdown
import numpy as np
from tensorflow import keras
from imgtovar.utils import functions


def loadModel(
    url="https://github.com/dvdimitrov13/thesis_modelzoo/releases/download/v.1.0.0/chart_cls.h5",
):

    home = functions.get_imgtovar_home()

    if os.path.isfile(home + "/.imgtovar/weights/chart_cls.h5") != True:
        print("chart_cls.h5 will be downloaded...")

        output = home + "/.imgtovar/weights/chart_cls.h5"
        gdown.download(url, output, quiet=False)

    reconstructed_model = keras.models.load_model(
        home + "/.imgtovar/weights/chart_cls.h5"
    )

    # print(reconstructed_model.summary())

    return reconstructed_model

