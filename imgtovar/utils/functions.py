from fileinput import filename
import os
import re
import operator
import functools

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import cv2
import glob
import base64
from pathlib import Path
from PIL import Image
import requests
import gdown

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorsys


import tensorflow as tf

from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image

from deepface.commons import functions
from deepface.detectors import FaceDetector
from deepface.detectors import (
    OpenCvWrapper,
    SsdWrapper,
    DlibWrapper,
    MtcnnWrapper,
    RetinaFaceWrapper,
    MediapipeWrapper,
)

# --------------------------------------------------
def initialize_input(data):

    if os.path.isfile(data):
        print("Operating in mode 'single file'")
        return ".", [[data]]

    elif os.path.isdir(data):
        check = [1 if os.path.isdir(os.path.join(data, x)) else 0 for x in os.listdir(data)]
        if functools.reduce(operator.mul, check, 1) == 0:
            print("Operating in mode 'directory of files'")
            return data, [[os.path.join(data, x) for x in os.listdir(data)]]
        print("Operating in mode 'directory of directories'")
        return data, [os.path.join(data, x) for x in os.listdir(data)]
        

    else:
        raise ValueError("Confirm that ", data, " exists")


def initialize_folder():
    home = get_imgtovar_home()

    if not os.path.exists(home + "/.imgtovar"):
        os.makedirs(home + "/.imgtovar")
        print("Directory ", home, "/.imgtovar created")

    if not os.path.exists(home + "/.imgtovar/weights"):
        os.makedirs(home + "/.imgtovar/weights")
        print("Directory ", home, "/.imgtovar/weights created")


def get_imgtovar_home():
    return str(os.getenv("IMGTOVAR_HOME", default=Path.home()))


def loadBase64Img(uri):
    encoded_data = uri.split(",")[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def load_image(img, BGR=False):
    exact_image = False
    base64_img = False
    url_img = False

    if type(img).__module__ == np.__name__:
        exact_image = True

    elif len(img) > 11 and img[0:11] == "data:image/":
        base64_img = True

    elif len(img) > 11 and img.startswith("http"):
        url_img = True

    # ---------------------------

    if base64_img == True:
        img = loadBase64Img(img)

    elif url_img:
        img = np.array(Image.open(requests.get(img, stream=True).raw))

    elif exact_image != True:  # image path passed as input
        if os.path.isfile(img) != True:
            raise ValueError("Confirm that ", img, " exists")

        img = cv2.imread(img)
        if not BGR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def download_asset(url):

    home = functions.get_deepface_home()

    asset_name = url.split("/")[-1]

    if os.path.isfile(home + "/.imgtovar/weights/" + asset_name) != True:
        print("{} will be downloaded...".format(asset_name))

        output = home + "/.imgtovar/weights/" + asset_name
        gdown.download(url, output, quiet=False)


def detect_faces(img, detector_backend, align=True, enforce_detection=True):

    if isinstance(img, str):
        print("Converting!")
        img = load_image(img=img, BGR=True)

    backends = {
        "opencv": OpenCvWrapper.detect_face,
        "ssd": SsdWrapper.detect_face,
        "dlib": DlibWrapper.detect_face,
        "mtcnn": MtcnnWrapper.detect_face,
        "retinaface": RetinaFaceWrapper.detect_face,
        "mediapipe": MediapipeWrapper.detect_face,
    }

    detect_face = backends.get(detector_backend)

    # detector stored in a global variable in FaceDetector object.
    # this call should be completed very fast because it will return found in memory
    # it will not build face detector model in each call (consider for loops)
    face_detector = FaceDetector.build_model(detector_backend)

    if detect_face:
        obj = detect_face(face_detector, img, align)
        # obj stores list of detected_face and region pair

        if len(obj) == 0 and enforce_detection == True:
            raise ValueError(
                "No faces were detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False."
            )

        return obj

    else:
        raise ValueError("invalid detector_backend passed - " + detector_backend)


def preprocess_img(img, target_size=(224, 224), grayscale=False):
    # post-processing
    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.expand_dims(
        cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST) / 255, axis=0
    )


def extract_number(f):
    s = re.findall("\d+$", f)
    return (int(s[0]) if s else -1, f)


def build_yolo_df(label_names, output_dir):
    final = pd.DataFrame(
        columns=["label", "x", "y", "width", "height", "confidence", "filename"]
    )

    folders = [os.path.join(output_dir, x) for x in os.listdir(output_dir)]
    for folder in tqdm(folders, desc='Aggregating yolo prediction results into a DataFrame'):
        if os.path.isdir(folder):
            for file in os.listdir(folder + "/labels"):
                fname = file.split(".")[0]
                temp = pd.read_csv(
                    os.path.join(folder + "/labels", file),
                    skipinitialspace=True,
                    header=None,
                    names=["label", "x", "y", "width", "height", "confidence"],
                    delim_whitespace=True,
                )
                if temp.shape[0] > 0:
                    temp["label"] = temp["label"].apply(lambda x: label_names[int(x)])
                    temp["filename"] = fname
                    final = pd.concat([final, temp], ignore_index=True)

    return final


def image_histogram(
    img_path,
    plot=True,
    plot_raw=True,
    max_intensity=100000,
    print_threshold_diagnostics=True,
    color_width=200,
    return_proportion=False,
):
    """ Takes in an image, and plots a histogram of color population against (hue,lightness) pairs. 
    If an image has a population of n (n = color_width) pixels with a specific 
    (hue,lightness) pair that make up more than pct_threshold of an image, we tag that image as artificial. 

    Parameters
    ----------
    img_folder: Path or str
        Fold with images to examine.
    plot: bool
        Plot histogram or not.
    plot_raw: bool
        Plot raw images or not.
    max_intensity: int
        Ceiling value for the hue:lightness pair populations. 
        This value will affect the pixel proportion if too low. 
    print_threshold_diagnostics: bool
        Prints diagnositic information (number of hue/lightness pairs, 
        how many are accounted for in calculating the proportion of the final image.)
    color_width: int
        How many of the most populous hue:lightness pairs to sum together to 
        determine the proportion of the final image they occupy. 
    return_proportion: bool
        Should the function return the color proportion coverage value. 

    Returns:
    ----------
    Color proportion coverage value (if return_proportion=True)
        Histogram (if plot=True)
        Raw Image (if plot_raw=True)
    """

    # Open image and get dimensions
    img = cv2.imread(img_path)
    # You may need to convert the color.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_file = Image.fromarray(img)
    # Resize images in order to bring final scores to similar scale
    img_file = img_file.resize((600, 600))
    img = img_file.load()
    [xs, ys] = img_file.size
    max_intensity = max_intensity
    hues = {}

    # For each pixel in the image file
    for x in range(0, xs):
        for y in range(0, ys):
            # Get the RGB color of the pixel
            [r, g, b] = img[x, y]
            # Normalize pixel color values
            r = r / 255.0
            g = g / 255.0
            b = b / 255.0
            # Convert RGB color to HSL
            [h, l, s] = colorsys.rgb_to_hls(r, g, b)
            # Count how many pixels have matching (h, l)
            if h not in hues:
                hues[h] = {}
            if l not in hues[h]:
                hues[h][l] = 1
            else:
                if hues[h][l] < max_intensity:
                    hues[h][l] += 1

    # Decompose the hues object into a set of one dimensional arrays
    h_ = []
    l_ = []
    i = []
    colours = []

    for h in hues:
        for l in hues[h]:
            h_.append(h)
            l_.append(l)
            i.append(hues[h][l])
            [r, g, b] = colorsys.hls_to_rgb(h, l, 1)
            colours.append([r, g, b])

    # Plot if wanted
    raw_image = np.asarray(img_file)
    if plot == True:
        fig = plt.figure(figsize=(12, 5))
        fig.set_facecolor("white")
        ax = plt.subplot2grid((2, 6), (0, 0), colspan=4, rowspan=2, projection="3d")
        ax.scatter(h_, l_, i, s=30, c=colours, lw=0.5, edgecolors="black")
        ax.set_xlabel("Hue")
        ax.set_ylabel("Lightness")
        ax.set_zlabel("Population")

        # Plot raw image if wanted
        if plot_raw == True:
            ax2 = plt.subplot2grid((2, 6), (0, 4), colspan=2, rowspan=2)
            ax2.imshow(raw_image)
            ax2.title.set_text(f"Raw Image: {img_path}")
        plt.tight_layout()
        plt.show()

    # Determine if the image we're examining is artificially generated
    n_greatest = sum(sorted(i, reverse=True)[:color_width])
    picture_size = xs * ys
    if print_threshold_diagnostics == True:
        print(f"There are {len(i)} hue/lightness pairs in this image.")
        print(
            f"Population of {color_width} hue/lightness pairs with the largest populations = {n_greatest}"
        )
        print(
            f"This represents {n_greatest/picture_size*100:.2f}% of the total pixels in the image."
        )

    if return_proportion == True:
        return len(i), n_greatest / picture_size

