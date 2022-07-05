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

from imgtovar.models import Age, Chart, Inverted, Background  # Custom model
from imgtovar.utils import functions


def build_model(model_name):

    """
	This function builds the models included with ImgtoVar
	Parameters:
		model_name (string): data clean, background classification or facial attributes model
            Chart, Inverted for data clean
            Background for background classification
			Age, Gender, Emotion, Race for facial attributes
	Returns:
		built imgtovar model
	"""

    # We store models in a global variable to avoid reloading within the same session
    global model_obj  # singleton design pattern

    models = {
        "Chart": Chart.loadModel,
        "Inverted": Inverted.loadModel,
        "Background": Background.loadModel,
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
    """
	This function extracts all images from a  PDF file and stores them in a dir named: ./extract_output
	Parameters:
		data: data_dir or exact image path could be passed.
		mode (string): Specifies the extraction mode, set to PDF by default, currently only PDF mode supported.

	Returns:
		None
        Creates a dir named ./extract_output where all extracted images are saved.
    """

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
    """ 
    This function finds images in an img_folder with excessively shallow color profiles that will be poor examples of a class. 

    Parameters:
		data: data_dir or exact image path could be passed.
        extract_artificial (bool): Determines if the images classified as artificial based on the predefined threshold will be moved from the original data dir to a new dir called ./Artificial_images
        color_width (int): How many of the most populous hue:lightness pairs to sum together to determine the proportion of the final image they occupy. 
        threshold (float): 0 < threshold < 1, what percent of the image is acceptable for color_width hue:lightness pairs to occupy, more than this is tagged as artificial.
        max_intensity (int): Ceiling value for the hue:lightness pair populations. This value will affect the pixel proportion if too low.

    Returns:
    ----------
    DataFrame
    DataFrame with image paths, the total H/L pairs detected and the proportion of the dominant population.
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
        prompt = input(
            "This action will make changes to your DATA directory, make sure to work with a copy of the original! Proceed ([y]/n)? "
        ).lower()

        if prompt == "y":
            artificial = df_mt[
                df_mt["{}_dominant_pairs_%".format(color_width)] > threshold
            ]
            print(f"{len(df_mt)} artificial images found")
            artlist = artificial["filename"].tolist()

            if not os.path.exists("./Artificial_images"):
                os.mkdir("./Artificial_images")

            for file in tqdm(
                files, desc="Moving Artificial immages to ./Artificial_images"
            ):
                if file in artlist:
                    shutil.move(os.path.join(DIR, file), "./Artificial_images")

            return df_mt

        else:
            print("No changes were made!")
            return df_mt

    return df_mt


def detect_infographics(data, model=None, run_on_gpu=False, extract_infographics=False):
    """
	This function analyzes images to detect if they are Infographics and what type, useful before applying color_analysis
	Parameters:
		data: data_dir or exact image path could be passed.
		model: (Optional keras model object): model = ImgtoVar.build_model("Chart"), the model can be passed as a prebuilt model for increased performance especially in loops.
		extract_inforgraphics (boolean): Determines if the images classified as infographics will be saved in a new dir with the following name: ./Infographics
        run_on_gpu (bool): Determines if tensorflow should use GPU acceleration, default is set to False for stability, if enabled you could run into OOM error depending on available GPU VRAM.

	Returns:
		DataFrame
        DataFrame with image paths, 
                       the face number with respect to any given image,
                       the predicted chart type including 'Not_chart' to indicate that no Infographic was detected.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if run_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    if extract_infographics:
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


def detect_invertedImg(data, model=None, run_on_gpu=False, extract_inverted=False):
    """
	This function analyzes images to detected those with inverted and distorted colors.
	Parameters:
		data: data_dir or exact image path could be passed.
		model: (Optional keras model object): model = ImgtoVar.build_model("Inverted"), the model can be passed as a prebuilt model for increased performance especially in loops.
		extract_inverted (boolean): Determines if the images classified as inverted will be saved in a new dir with the following name: ./Inverted_imgs
        run_on_gpu (bool): Determines if tensorflow should use GPU acceleration, default is set to False for stability, if enabled you could run into OOM error depending on available GPU VRAM.

	Returns:
		DataFrame
        DataFrame with image paths, 
                       the face number with respect to any given image,
                       the predicted chart type including 'Not_chart' to indicate that no Infographic was detected.
        """

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if run_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    DIR, files = functions.initialize_input(data)

    # We build the model
    if not model:
        model = build_model("Inverted")

    d = []

    for file in tqdm(files, "Analysing images"):

        img = functions.load_image(img=os.path.join(DIR, file), BGR=False)

        prep_img = functions.preprocess_img(img)
        prediction = int(np.argmax(model.predict(prep_img), axis=1))

        d.append({"filename": file, "inverted": prediction == 1})

    df = pd.DataFrame(d)

    if extract_inverted:
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


def background_analysis(data, model=None, run_on_gpu=False):
    """
	This function analyzes the image background to identify if it is natural or man-made (Artificial), it also has a label 'Other' capturing images with no background or otherwise non-classifiable images.
	Parameters:
		data: data_dir or exact image path could be passed.
		model: (Optional keras model object): model = ImgtoVar.build_model("Background"), the model can be passed as a prebuilt model for increased performance especially in loops.
        run_on_gpu (bool): Determines if tensorflow should use GPU acceleration, default is set to False for stability, if enabled you could run into OOM error depending on available GPU VRAM.

	Returns:
		DataFrame
        DataFrame with image paths, 
                       the face number with respect to any given image,
                       the predicted chart type including 'Not_chart' to indicate that no Infographic was detected.
        """

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if run_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    DIR, files = functions.initialize_input(data)

    # We build the model
    if not model:
        model = build_model("Background")

    d = []

    for file in tqdm(files, desc="Analysing Background"):

        img = functions.load_image(img=os.path.join(DIR, file), BGR=False)

        background_labels = ["Natural", "Artificial", "Other"]

        prep_img = functions.preprocess_img(img)
        prediction = int(np.argmax(model.predict(prep_img), axis=1))

        d.append({"filename": file, "background": background_labels[prediction]})

    return pd.DataFrame(d)


def face_analysis(
    data,
    align=False,
    actions=("emotion", "age", "gender", "race"),
    models=None,
    detector_backend="retinaface",
    extract_faces=False,
    enforce_detection=True,
    return_JSON=False,
    custom_model_channel="BGR",
    custom_target_size=(224, 224),
    run_on_gpu=False,
):

    """
	This function analyzes facial attributes including age, gender, emotion and race
	Parameters:
		data: data_dir or exact image path could be passed.
		actions (tuple): The default is ('age', 'gender', 'emotion', 'race'). You can drop some of those attributes.
		models: (Optional[dict]) facial attribute analysis models are built in every call of analyze function. You can pass pre-built models to speed the function up or to use a custom model.
            N.B. -- If using a custom model you need to pass it under a key == 'custom' (ex. models['custom'] = {custom_keras_model})
			models = {}
			models['age'] = ImgtoVar.build_model('Age') -- custom model that classifies in 5 age groups
			models['gender'] = ImgtoVar.build_model('Gender') -- model built based on DeepFace
			models['emotion'] = ImgtoVar.build_model('Emotion') -- model built based on DeepFace
			models['race'] = ImgtoVar.build_model('Race') -- model built based on DeepFace
		enforce_detection (boolean): The function throws exception if no faces were detected. Set to False by default.
		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib.
		extract_faces (boolean): Determines if the faces extracted form the images will be saved in a new dir with the following name: ./faces_output
        custom_model_channel (string): The channel used to train your custom model default is BGR set to "RGB" if needed
        custom_target_size (tuple): The input size of the custom model, default is (224, 224)
        run_on_gpu (bool): Determines if tensorflow should use GPU acceleration, default is set to False for stability, if enabled you could run into OOM error depending on available GPU VRAM.
        return_JSON (boolean): Determines the return type, set to False by default.
                               Set to true in order to get a ditionary with more detailed information, with the following format:
                               {
                                    "region": {'x': 230, 'y': 120, 'w': 36, 'h': 45},
                                    "age": '20-26',
                                    "dominant_gender": "Woman",
                                    "gender": {
                                        'Woman': 99.99407529830933,
                                        'Man': 0.005928758764639497,
                                    }
                                    "dominant_emotion": "neutral",
                                    "emotion": {
                                        'sad': 37.65260875225067,
                                        'angry': 0.15512987738475204,
                                        'surprise': 0.0022171278033056296,
                                        'fear': 1.2489334680140018,
                                        'happy': 4.609785228967667,
                                        'disgust': 9.698561953541684e-07,
                                        'neutral': 56.33133053779602
                                    }
                                    "dominant_race": "white",
                                    "race": {
                                        'indian': 0.5480832420289516,
                                        'asian': 0.7830780930817127,
                                        'latino hispanic': 2.0677512511610985,
                                        'black': 0.06337375962175429,
                                        'middle eastern': 3.088453598320484,
                                        'white': 93.44925880432129
                                    }
                                }

	Returns:
		DataFrame
        DataFrame with image paths, 
                       the face number with respect to any given image,
                       the pixel size of the detected face,
                       the coordinates of the detected face in the original image (xywh format),
                       and the predicted attributes based on the actions given.
        """

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if run_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

        if "custom" in built_models and "custom" not in actions:
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
            img=img,
            detector_backend=detector_backend,
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

    if return_JSON:
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
    """
	This function is a wrapper for object detection based on the YoloV5 architecture by Ultralytics.
    By default it uses the COCO pre-trained weights provided by the original YoloV5 package.
    Additionally it allows for the specification of two custom models trained specifically for ImgtoVars or finally a custom model trained by the user. 
	Parameters:
		data: data_dir or exact image path could be passed.
		model (string): specifies model options. By default it is set to "csutom" which allows the setting of custom weights.
               model can also be set to:
                    * 'c_energy' -- a custom model trained to detected the following objects: ["Crane", "Wind turbine", "farm equipment", "oil pumps", "plant chimney", "solar panels"]
                    * 'sub_open_images' -- a custom model trained on a subset of google Open Images dataset included labels are: [ "Animal", "Tree", "Plant", "Flower", "Fruit", "Suitcase", "Motorcycle", "Helicopter", "Sports Equipment", "Office Building", "Tool", "Medical Equipment", "Mug", "Sunglasses", "Headphones", "Swimwear", "Suit", "Dress", "Shirt", "Desk", "Whiteboard", "Jeans", "Helmet", "Building"]
        weights (path): determines the weights used if model is set to 'custom' by default COCO pre-trained weights are used, specify labels parameter if you are using a custom model
        conf_thres (float): 0 < conf_thresh < 1, the detection confidence cutoff
        imgsz (int): The image size for detection, default set to 640, best results are achieved if the img size is the same as the one used for training for more information chech yolov5 documentation.
        save_imgs (bool): Determines if the images on which detection was ran should be saved with the predictions overlayed. Set to False by default.
        labels (list): A list with the labels used in custom prediction, must be set when using with custom model.

	Returns:
		DataFrame
        DataFrame with image paths, 
                       the predicted label,
                       the coordinates of the detected object (xywh format),
                       the confidence level of the prediction,
    """

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

        return functions.build_yolo_df(labels, name)

    if model == "sub_open_images":

        functions.download_asset(
            url="https://github.com/dvdimitrov13/thesis_modelzoo/releases/download/v.1.0.0/sub_oid_5l.pt"
        )

        weights = (
            functions.get_imgtovar_home() + "/.imgtovar/weights/" + "sub_oid_5l.pt"
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
            functions.get_imgtovar_home() + "/.imgtovar/weights/" + "c_energy_5l.pt"
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

