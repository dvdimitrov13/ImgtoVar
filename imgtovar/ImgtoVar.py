import io
import os
import shutil
from cProfile import label
from multiprocessing.sharedctypes import Value
from unicodedata import name

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from deepface import DeepFace
from deepface.extendedmodels import Emotion, Gender, Race  # DeepFace models
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from tqdm import tqdm
from yolov5 import detect

from imgtovar.models import Age, Chart, Inverted, Nature  # Custom model
from imgtovar.utils import functions


def build_model(model_name):

    # We store models in a global variable to avoid reloading within the same session
    global model_obj  # singleton design pattern

    models = {
        "Chart": Chart.loadModel,
        "Inverted": Inverted.loadModel,
        "Nature": Nature.loadModel,
        "Emotion": Emotion.loadModel,
        "Age": Age.loadModel,
        "Gender": Gender.loadModel,
        "Race": Race.loadModel,
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj.keys():
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
            print(model_name, " built")
        else:
            raise ValueError("Invalid model_name passed - {}".format(model_name))

    return model_obj[model_name]


def extract(data, mode="PDF"):

    DIR, files = functions.initialize_input(data)

    if mode.lower() == "pdf":
        for path in files:
            if ".pdf" in path:
                doc = fitz.Document((os.path.join(DIR, path)))

                for page_index in tqdm(range(len(doc)), desc="pages"):
                    for image_index, img in enumerate(doc.get_page_images(page_index)):
                        # get the XREF of the image
                        xref = img[0]
                        # extract the image bytes
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        # get the image extension
                        image_ext = base_image["ext"]
                        # load it to PIL
                        image = Image.open(io.BytesIO(image_bytes))
                        # Create output_dir
                        if not os.path.exists("./extract_output"):
                            os.mkdir("./extract_output")
                        # Save to output folder
                        image.save(
                            open(
                                f"./extract_output/image{page_index+1}_{image_index}.{image_ext}",
                                "wb",
                            )
                        )
    else:
        raise ValueError("Provide a valid mode: currently supporting only 'pdf'")


def color_analysis(
    data, extract_artificial=False, color_width=200, max_intensity=100000, threshold=0.5
):
    """ Finds images in an img_folder with excessively shallow color profiles that will be poor examples of a class. 

    Parameters
    ----------
    img_folder: Path or str
    Fold with images to examine.
    return_all: bool
    Determines if the function should return the entire dataset.
    color_width: int
    How many of the most populous hue:lightness pairs to sum together to determine the proportion of the final image they occupy. 
    threshold: float, 0 < threshold < 1 
    What percent of the image is acceptable for 1000 hue:lightness pairs to occupy, more than this is tagged for removal.

    Returns:
    ----------
    DataFrame
    DataFrame with image paths to remove, and reason (pixel proportion).
    """

    # Initialize input
    DIR, files = functions.initialize_input(data)

    # Initialize DataFrame list
    d = []

    # Loop over folder
    for image in tqdm(files, desc="Analysing colors"):
        total_pairs, proportion = functions.image_histogram(
            os.path.join(DIR, image),
            plot=False,
            plot_raw=False,
            return_proportion=True,
            print_threshold_diagnostics=False,
            color_width=color_width,
            max_intensity=max_intensity,
        )

        # Append to dataframe
        d.append(
            {
                "filename": image,
                "H/L_pairs": total_pairs,
                "{}_dominant_pairs_%".format(color_width): proportion,
            }
        )

    df_mt = pd.DataFrame(d)

    if extract_artificial:
        df_mt = df_mt[df_mt["{}_dominant_pairs_%".format(color_width)] > threshold]
        print(f"{len(df_mt)} artificial images found")
        return df_mt

    else:
        return df_mt


def detect_infographics(data, model=None, run_on_cpu=False, filter_infographics=False):

    if run_on_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    DIR, files = functions.initialize_input(data)

    # We build the model
    if not model:
        model = build_model("Chart")

    d = []

    for file in tqdm(files, "Analysing images"):

        img = functions.load_image(img=os.path.join(DIR, file), BGR=False)

        chart_labels = [
            "AreaGraph",
            "BarGraph",
            "BoxPlot",
            "BubbleChart",
            "FlowChart",
            "LineGraph",
            "Map",
            "NetworkDiagram",
            "Not_chart",
            "ParetoChart",
            "PieChart",
            "ScatterGraph",
            "Table",
            "TreeDiagram",
            "VennDiagram",
        ]

        prep_img = functions.preprocess_img(img)
        prediction = int(np.argmax(model.predict(prep_img), axis=1))

        d.append({"filename": file, "chart_type": chart_labels[prediction]})

    df = pd.DataFrame(d)

    if filter_infographics:
        prompt = input(
            "This action will make changes to your DATA directory, make sure to work with a copy of the original! Proceed ([y]/n)? "
        ).lower()

        if prompt == "y":
            plots = df[df["chart_type"] != "Not_chart"]
            plotlist = plots["filename"].tolist()

            if not os.path.exists("./Infographics"):
                os.mkdir("./Infographics")

            for file in tqdm(files, desc="Moving infographics to ./Infographics"):
                if file in plotlist:
                    shutil.move(os.path.join(DIR, file), "./Infographics")

            return df

        else:
            print("No changes were made!")
            return df

    return pd.DataFrame(d)


def detect_invertedImg(data, model=None, run_on_cpu=False, filter_inverted=False):

    if run_on_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    DIR, files = functions.initialize_input(data)

    # We build the model
    if not model:
        model = build_model("Inverted")

    d = []

    for file in tqdm(files), "Analysing images":

        img = functions.load_image(img=os.path.join(DIR, file), BGR=False)

        # nature_labels = ["True", "Inverted"]

        prep_img = functions.preprocess_img(img)
        prediction = int(np.argmax(model.predict(prep_img), axis=1))

        d.append({"filename": file, "inverted": prediction == 1})

    df = pd.DataFrame(d)

    if filter_inverted:
        prompt = input(
            "This action will make changes to your DATA directory, make sure to work with a copy of the original! Proceed ([y]/n)? "
        ).lower()

        if prompt == "y":
            inverted = df[df["inverted"] == True]
            invlist = inverted["filename"].tolist()

            if not os.path.exists("./Inverted_imgs"):
                os.mkdir("./Inverted_imgs")

            for file in tqdm(files, desc="Moving inveted images to ./Inverted_imgs"):
                if file in invlist:
                    shutil.move(os.path.join(DIR, file), "./Inverted_imgs")

            return df

        else:
            print("No changes were made!")
            return df

    return pd.DataFrame(d)


def background_analysis(data, model=None, run_on_cpu=False):
    if run_on_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    DIR, files = functions.initialize_input(data)

    # We build the model
    if not model:
        model = build_model("Nature")

    d = []

    for file in tqdm(files, desc="Analysing Background"):

        img = functions.load_image(img=os.path.join(DIR, file), BGR=False)

        nature_labels = ["Natural", "Artificial", "Other"]

        prep_img = functions.preprocess_img(img)
        prediction = int(np.argmax(model.predict(prep_img), axis=1))

        d.append({"filename": file, "background": nature_labels[prediction]})

    return pd.DataFrame(d)


def face_analysis(
    data,
    align=False,
    actions=("emotion", "age", "gender", "race"),
    models=None,
    detector_backend="retinaface",
    extract_faces=False,
    enforce_detection=True,
    return_raw=False,
    custom_model_channel="RGB",
    custom_target_size=(224, 224),
    run_on_cpu=False,
):
    if run_on_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    DIR, files = functions.initialize_input(data)

    # Now we build the models to speed up the inference process

    # First lets initailize an empty object
    if not models:
        models = {}

    # ---------------------------------
    # Next we check if the user has provided pre-built models

    built_models = list(models.keys())

    # ---------------------------------
    # We update the actions list if model was passed but not in actions
    if len(built_models) > 0:
        if "emotion" in built_models and "emotion" not in actions:
            actions.append("emotion")

        if "age" in built_models and "age" not in actions:
            actions.append("age")

        if "gender" in built_models and "gender" not in actions:
            actions.append("gender")

        if "race" in built_models and "race" not in actions:
            actions.append("race")

        if "custom" in built_models and "emotion" not in actions:
            actions.append("custom")

    # ---------------------------------
    # Finally we build the models included with the module if none were passed
    if "emotion" in actions and "emotion" not in built_models:
        models["emotion"] = build_model("Emotion")

    if "age" in actions and "age" not in built_models:
        models["age"] = build_model("Age")

    if "gender" in actions and "gender" not in built_models:
        models["gender"] = build_model("Gender")

    if "race" in actions and "race" not in built_models:
        models["race"] = build_model("Race")

    if extract_faces != True:
        print(
            "Extracted faces will not be saved, consider setting extract_faces to True if you prefer saving that output."
        )

    d = []

    for file in tqdm(files, desc="Analysing faces"):

        img = functions.load_image(img=os.path.join(DIR, file), BGR=True)

        deepface_ms = models.copy()
        deepface_ms.pop("age", None)

        faces = functions.detect_faces(
            face_detector=None,
            detector_backend=detector_backend,
            img=img,
            align=align,
            enforce_detection=enforce_detection,
        )

        for i, (face, region) in enumerate(faces):

            if extract_faces:
                if not os.path.exists("./faces_output"):
                    os.makedirs("./faces_output")
                    print("Directory  ./faces_output created")

                cv2.imwrite(
                    "./faces_output/{}_{}.jpg".format(file, i),
                    cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
                )

            DF_resp = DeepFace.analyze(
                face, detector_backend="skip", models=deepface_ms, prog_bar=False
            )

            if "age" in actions:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                age_labels = ["1-18", "19-26", "27-39", "40-60", "60+"]

                # Predict age
                prep_face = functions.preprocess_img(face)
                age_group = int(np.argmax(models["age"].predict(prep_face), axis=1))

                DF_resp["age"] = age_labels[age_group]

            # Adding an option for custom model
            if "custom" in actions:
                if custom_model_channel == "RGB":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif custom_model_channel == "BGR":
                    pass
                else:
                    raise ValueError(
                        "Invalid custom_model_channel passed - {}. Options: RBG or BGR".format(
                            custom_model_channel
                        )
                    )

                # Predict
                prep_face = functions.preprocess_img(
                    face, target_size=custom_target_size
                )
                prediction = int(np.argmax(models["custom"].predict(prep_face), axis=1))

                DF_resp["custom"] = prediction

            # Structure data
            d.append(
                {
                    "filename": file,
                    "face_number": i,
                    "face_size": "{}x{}".format(face.shape[0], face.shape[1]),
                    "face_x": region[0],
                    "face_y": region[1],
                    "face_w": region[2],
                    "face_h": region[3],
                    "predicted_age": DF_resp["age"] if "age" in actions else None,
                    "predicted_gender": DF_resp["gender"]
                    if "gender" in actions
                    else None,
                    "predicted_race": DF_resp["dominant_race"]
                    if "race" in actions
                    else None,
                    "predicted_emotion": DF_resp["dominant_emotion"]
                    if "emotion" in actions
                    else None,
                    "custom_prediction": DF_resp["custom"]
                    if "custom" in actions
                    else None,
                }
            )

    if return_raw:
        return DF_resp
    else:
        return pd.DataFrame(d)


def detect_objects(
    data,
    model="custom",
    weights="yolov5l.pt",
    conf_thres=0.25,
    imgsz=640,
    save_imgs=False,
    labels=None,
):

    # Validate data file format
    if os.path.isfile(data):
        pass

    elif os.path.isdir(data):
        pass

    else:
        raise ValueError("Confirm that ", data, " exists and is a file or directory")

    if model == "custom":

        name = str(weights).split(".")[0] + "_result"

        detect.run(
            source=data,
            weights=weights,
            conf_thres=conf_thres,
            imgsz=imgsz,
            save_txt=True,
            name=name,
            save_conf=True,
            nosave=not save_imgs,
        )

        if labels == None:
            labels = [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            ]

        return functions.build_yolo_df(labels)

    if model == "sub_open_images":

        functions.download_asset(
            url="https://github.com/dvdimitrov13/thesis_modelzoo/releases/download/v.1.0.0/sub_oid_5l.pt"
        )

        weights = (
            functions.get_deepface_home() + "/.imgtovar/weights/" + "sub_oid_5l.pt"
        )
        name = "chart_cls" + "_result"

        detect.run(
            source=data,
            weights=weights,
            conf_thres=conf_thres,
            imgsz=imgsz,
            save_txt=True,
            name=name,
            save_conf=True,
            nosave=not save_imgs,
        )

        labels = [
            "Animal",
            "Tree",
            "Plant",
            "Flower",
            "Fruit",
            "Suitcase",
            "Motorcycle",
            "Helicopter",
            "Sports_equipment",
            "Office_building",
            "Tool",
            "Medical_equipment",
            "Mug",
            "Sunglasses",
            "Headphones",
            "Swimwear",
            "Suit",
            "Dress",
            "Shirt",
            "Desk",
            "Whiteboard",
            "Jeans",
            "Helmet",
            "Building",
        ]

        return functions.build_yolo_df(labels, name)

    if model == "c_energy":

        functions.download_asset(
            url="https://github.com/dvdimitrov13/thesis_modelzoo/releases/download/v.1.0.0/c_energy_5l.pt"
        )

        weights = (
            functions.get_deepface_home() + "/.imgtovar/weights/" + "c_energy_5l.pt"
        )

        name = "c_energy_5l" + "_result"

        detect.run(
            source=data,
            weights=weights,
            conf_thres=conf_thres,
            imgsz=imgsz,
            save_txt=True,
            name=name,
            save_conf=True,
            nosave=not save_imgs,
        )

        labels = [
            "BUFFER",
            "Crane",
            "Wind turbine",
            "farm equipment",
            "oil pumps",
            "plant chimney",
            "solar panels",
        ]
        return functions.build_yolo_df(labels, name)

    else:
        raise ValueError(
            "Provide a valid model: 'c_energy', 'open_images' or 'custom' with appropriate weights"
        )


functions.initialize_folder()

