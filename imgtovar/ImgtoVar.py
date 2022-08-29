import io
import os
import shutil
import pickle
import functools
import operator
import glob
from cProfile import label
from multiprocessing.sharedctypes import Value
import string
from tkinter.font import names
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
		data: a pdf file or a directory fo pdf files.
		mode (string): Specifies the extraction mode, set to PDF by default, currently only PDF mode supported.

	Returns:
		None
        Creates a dir named ./extract_output where all extracted images are saved.
    """

    DIR, files = functions.initialize_input(data)
    # Create output_dir
    if not os.path.exists("./extract_output"):
        os.mkdir("./extract_output")

    if mode.lower() == "pdf":
        for folder in files:
            for path in folder:
                if ".pdf" in path:
                    doc_name = os.path.basename(path).split(".")[0]
                    doc = fitz.Document((os.path.join(DIR, path)))

                    # Create output dirs
                    if not os.path.exists(f"./extract_output/{doc_name}"):
                        os.mkdir(f"./extract_output/{doc_name}")

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
                            # Convert to RGB 
                            image = image.convert("RGB")

                            # Save to output folder as JPG
                            image.save(
                                open(
                                    f"./extract_output/{doc_name}/{doc_name}_page{page_index+1}_image{image_index}.jpg",
                                    "wb",
                                )
                            )
    else:
        raise ValueError("Provide a valid mode: currently supporting only 'pdf'")


def color_analysis(
    data, extract=False, color_width=200, max_intensity=100000, threshold=0.5, resume = False
):
    """ 
    This function finds images in an img_folder with excessively shallow color profiles that will be poor examples of a class. 

    Parameters:
		data: Three modes of operation: 'Single file', 'Directory of files', 'Directory of directories'
        extract(bool): Determines if the images classified as artificial based on the predefined threshold will be moved from the original data dir to a new dir called ./Artificial_images
        color_width (int): How many of the most populous hue:lightness pairs to sum together to determine the proportion of the final image they occupy. 
        threshold (float): 0 < threshold < 1, what percent of the image is acceptable for color_width hue:lightness pairs to occupy, more than this is tagged as artificial.
        max_intensity (int): Ceiling value for the hue:lightness pair populations. This value will affect the pixel proportion if too low.
        resume (bool): Determines if the a new experiment will be run or a previous one is resumed. When the data directory contains a folder for each pdf a checkpoint is made after each folder!

    Returns:
    ----------
    DataFrame
    DataFrame with image paths, the total H/L pairs detected and the proportion of the dominant population.
    """

    # Get permition to alter file structure if extract set True
    if extract:
        prompt = input(
            "This action will make changes to your DATA directory, make sure to work with a copy of the original! Proceed ([y]/n)? "
        ).lower()

        if prompt != 'y':
            print(f"No changes will be made! Answer: {prompt} (make sure to answer with 'y' or 'Y' if you wanted to agree)")

    # Create output folder for checkpoints
    if not os.path.exists("./Output/color_analysis/exp_1"):
        os.makedirs("./Output/color_analysis/exp_1")
        output_dir = "./Output/color_analysis/exp_1"
    else:
        numbers = [int(name.split("_")[-1]) for name in os.listdir("./Output/color_analysis")]
        number = max(numbers)
        if resume:
            output_dir = f"./Output/color_analysis/exp_{number}"
        else:
            os.makedirs(f"./Output/color_analysis/exp_{number + 1}")
            output_dir = f"./Output/color_analysis/exp_{number + 1}"

    # Initialize input
    DIR, intput_folders = functions.initialize_input(data)

    # Initialize DataFrame list
    d = []
    extraction = []
    progress = []

    # Loop over either files or folder structure
    for i, folder in enumerate(intput_folders):
        # Check for checkpoints
        if os.path.exists(output_dir + '/checkpoints'):
            with open(output_dir + '/checkpoints', "rb") as fp:   # Unpickling
                checkpoints = pickle.load(fp)

        else:
            checkpoints = []
            
        if folder in checkpoints:
            print(f"Folder {folder}, already analysed!")
            continue

        if isinstance(folder, list):
            filenames = folder
            folder= data
        else:
            filenames = [os.path.join(folder, x) for x in os.listdir(folder)]
        

        for image in tqdm(filenames, desc="Analysing folder {}/{} -- {}".format(i+1, len(intput_folders), folder)):

            total_pairs, proportion = functions.image_histogram(
                image,
                plot=False,
                plot_raw=False,
                return_proportion=True,
                print_threshold_diagnostics=False,
                color_width=color_width,
                max_intensity=max_intensity,
            )

            if extract:
                if proportion > threshold:
                    extraction.append(image)

            # Append to dataframe
            d.append(
                {   
                    "id": os.path.basename(os.path.splitext(image)[0]),
                    "filename": image,
                    "H/L_pairs": total_pairs,
                    "{}_dominant_pairs_%".format(color_width): proportion,
                }
            )


        df_mt = pd.DataFrame(d)        
        df_mt.to_csv(output_dir + "/color_analysis.csv")

        if extract:
            if prompt == "y":
                print(f"{len(extraction)} artificial images found")

                destination = output_dir + os.path.join( "/Artificial/", os.path.basename(os.path.normpath(folder)))
                if not os.path.exists(destination):
                    os.makedirs(destination)
            
                for image in tqdm(extraction, desc="Moving artifical images to ./Output/color_analysis/Artificial"):
                    shutil.move(image, destination)
        
        progress.append(folder)
        with open(output_dir + "/checkpoints", "wb") as fp:   #Pickling
            pickle.dump(progress, fp)

    return pd.read_csv(output_dir + "/color_analysis.csv").iloc[:,1:]


def detect_infographics(data, model=None, run_on_gpu=False, extract=False, resume=False):
    """
	This function analyzes images to detect if they are Infographics and what type, useful before applying color_analysis
	Parameters:
        data: Three modes of operation: 'Single file', 'Directory of files', 'Directory of directories'
		model: (Optional keras model object): model = ImgtoVar.build_model("Chart"), the model can be passed as a prebuilt model for increased performance especially in loops.
		extract (boolean): Determines if the images classified as infographics will be saved in a new dir with the following name: ./Infographics
        run_on_gpu (bool): Determines if tensorflow should use GPU acceleration, default is set to False for stability, if enabled you could run into OOM error depending on available GPU VRAM.
        resume (bool): Determines if the a new experiment will be run or a previous one is resumed. When the data directory contains a folder for each pdf a checkpoint is made after each folder!


	Returns:
		DataFrame
        DataFrame with image paths, 
                       the face number with respect to any given image,
                       the predicted chart type including 'Not_chart' to indicate that no Infographic was detected.
    """

    # Set up what is the code going to be running on
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if run_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # We build the model
    if not model:
        model = build_model("Chart")
    
    # Get permition to alter file structure if extract set True
    if extract:
        prompt = input(
            "This action will make changes to your DATA directory, make sure to work with a copy of the original! Proceed ([y]/n)? "
        ).lower()

        if prompt != 'y':
            print(f"No changes will be made! Answer: {prompt} (make sure to answer with 'y' or 'Y' if you wanted to agree)")

    # Create output folder for checkpoints
    if not os.path.exists("./Output/infographics/exp_1"):
        os.makedirs("./Output/infographics/exp_1")
        output_dir = "./Output/infographics/exp_1"
    else:
        numbers = [int(name.split("_")[-1]) for name in os.listdir("./Output/infographics")]
        number = max(numbers)
        if resume:
            output_dir = f"./Output/infographics/exp_{number}"
        else:
            os.makedirs(f"./Output/infographics/exp_{number + 1}")
            output_dir = f"./Output/infographics/exp_{number + 1}"

    # Initialize input
    DIR, intput_folders = functions.initialize_input(data)

    # Initialize DataFrame list
    d = []
    progress = []

    # Loop over either files or folder structure
    for i, folder in enumerate(intput_folders):
        # Check for checkpoints
        if os.path.exists(output_dir + '/checkpoints'):
            with open(output_dir + '/checkpoints', "rb") as fp:   # Unpickling
                checkpoints = pickle.load(fp)

        else:
            checkpoints = []
            
        if folder in checkpoints:
            print(f"Folder {folder}, already analysed!")
            continue

        if isinstance(folder, list):
            filenames = folder
            folder= data
        else:
            filenames = [os.path.join(folder, x) for x in os.listdir(folder)]
        

        for file in tqdm(filenames, desc="Analysing folder {}/{} -- {}".format(i+1, len(intput_folders), folder)):

            img = functions.load_image(img=file, BGR=False)

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

            d.append({"id": os.path.basename(os.path.splitext(file)[0]), "filename": file, "chart_type": chart_labels[prediction]})

        df = pd.DataFrame(d)        
        df.to_csv(output_dir + "/infographics.csv")

        if extract:
            if prompt == "y":
                plots = df[df["chart_type"] != "Not_chart"]
                extraction = plots["filename"].tolist()

                destination = output_dir + os.path.join( "/Infographics/", os.path.basename(os.path.normpath(folder)))
                if not os.path.exists(destination):
                    os.makedirs(destination)
            
                for image in tqdm(extraction, desc="Moving artifical images to ./Output/infographics/Infographics"):
                    shutil.move(image, destination)
        
        progress.append(folder)
        with open(output_dir + "/checkpoints", "wb") as fp:   #Pickling
            pickle.dump(progress, fp)

    return pd.read_csv(output_dir + "/infographics.csv").iloc[:,1:]


def detect_invertedImg(data, model=None, run_on_gpu=False, extract=False, resume=False):
    """
	This function analyzes images to detected those with inverted and distorted colors.
	Parameters:
		data: Three modes of operation: 'Single file', 'Directory of files', 'Directory of directories'
		model: (Optional keras model object): model = ImgtoVar.build_model("Inverted"), the model can be passed as a prebuilt model for increased performance especially in loops.
		extract (boolean): Determines if the images classified as inverted will be moved to a new dir 
        run_on_gpu (bool): Determines if tensorflow should use GPU acceleration, default is set to False for stability, if enabled you could run into OOM error depending on available GPU VRAM.
        resume (bool): Determines if the a new experiment will be run or a previous one is resumed. When the data directory contains a folder for each pdf a checkpoint is made after each folder!
	
    Returns:
		DataFrame
        DataFrame with image paths, 
                       the face number with respect to any given image,
                       the predicted chart type including 'Not_chart' to indicate that no Infographic was detected.
        """

    # Set up what is the code going to be running on
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if run_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # We build the model
    if not model:
        model = build_model("Inverted")
    
    # Get permition to alter file structure if extract set True
    if extract:
        prompt = input(
            "This action will make changes to your DATA directory, make sure to work with a copy of the original! Proceed ([y]/n)? "
        ).lower()

        if prompt != 'y':
            print(f"No changes will be made! Answer: {prompt} (make sure to answer with 'y' or 'Y' if you wanted to agree)")

    # Create output folder for checkpoints
    if not os.path.exists("./Output/inverted_images/exp_1"):
        os.makedirs("./Output/inverted_images/exp_1")
        output_dir = "./Output/inverted_images/exp_1"
    else:
        numbers = [int(name.split("_")[-1]) for name in os.listdir("./Output/inverted_images")]
        number = max(numbers)
        if resume:
            output_dir = f"./Output/inverted_images/exp_{number}"
        else:
            os.makedirs(f"./Output/inverted_images/exp_{number + 1}")
            output_dir = f"./Output/inverted_images/exp_{number + 1}"

    # Initialize input
    DIR, intput_folders = functions.initialize_input(data)

    # Initialize DataFrame list
    d = []
    progress = []

    # Loop over either files or folder structure
    for i, folder in enumerate(intput_folders):
        # Check for checkpoints
        if os.path.exists(output_dir + '/checkpoints'):
            with open(output_dir + '/checkpoints', "rb") as fp:   # Unpickling
                checkpoints = pickle.load(fp)

        else:
            checkpoints = []
            
        if folder in checkpoints:
            print(f"Folder {folder}, already analysed!")
            continue

        if isinstance(folder, list):
            filenames = folder
            folder= data
        else:
            filenames = [os.path.join(folder, x) for x in os.listdir(folder)]
        

        for file in tqdm(filenames, desc="Analysing folder {}/{} -- {}".format(i+1, len(intput_folders), folder)):

            img = functions.load_image(img=file, BGR=False)

            prep_img = functions.preprocess_img(img)
            prediction = int(np.argmax(model.predict(prep_img), axis=1))

            d.append({"id": os.path.basename(os.path.splitext(file)[0]), "filename": file, "inverted": prediction == 1})

        df = pd.DataFrame(d)        
        df.to_csv(output_dir + "/inverted.csv")

        if extract:
            if prompt == "y":
                inverted = df[df["inverted"] == True]
                extraction = inverted["filename"].tolist()

                destination = output_dir + os.path.join( "/Inverted/", os.path.basename(os.path.normpath(folder)))
                if not os.path.exists(destination):
                    os.makedirs(destination)

                for image in tqdm(extraction, desc="Moving artifical images to ./Output/inverted_imgs/exp/Inverted"):
                    shutil.move(image, destination)

    return pd.read_csv(output_dir + "/inverted.csv").iloc[:,1:]


def background_analysis(data, model=None, run_on_gpu=False, resume=False):
    """
	This function analyzes the image background to identify if it is natural or man-made (Artificial), it also has a label 'Other' capturing images with no background or otherwise non-classifiable images.
	Parameters:
		data: Three modes of operation: 'Single file', 'Directory of files', 'Directory of directories'
		model: (Optional keras model object): model = ImgtoVar.build_model("Background"), the model can be passed as a prebuilt model for increased performance especially in loops.
        run_on_gpu (bool): Determines if tensorflow should use GPU acceleration, default is set to False for stability, if enabled you could run into OOM error depending on available GPU VRAM.
        resume (bool): Determines if the a new experiment will be run or a previous one is resumed. When the data directory contains a folder for each pdf a checkpoint is made after each folder!

	Returns:
		DataFrame
        DataFrame with image paths, 
                       the face number with respect to any given image,
                       the predicted chart type including 'Not_chart' to indicate that no Infographic was detected.
        """

    # Set up what is the code going to be running on
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if run_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # We build the model
    if not model:
        model = build_model("Background")
    
   # Create output folder for checkpoints
    if not os.path.exists("./Output/background_analysis/exp_1"):
        os.makedirs("./Output/background_analysis/exp_1")
        output_dir = "./Output/background_analysis/exp_1"
    else:
        numbers = [int(name.split("_")[-1]) for name in os.listdir("./Output/background_analysis")]
        number = max(numbers)
        if resume:
            output_dir = f"./Output/background_analysis/exp_{number}"
        else:
            os.makedirs(f"./Output/background_analysis/exp_{number + 1}")
            output_dir = f"./Output/background_analysis/exp_{number + 1}"

    # Initialize input
    DIR, intput_folders = functions.initialize_input(data)

    # Initialize DataFrame list
    d = []
    progress = []

    # Loop over either files or folder structure
    for i, folder in enumerate(intput_folders):
        # Check for checkpoints
        if os.path.exists(output_dir + '/checkpoints'):
            with open(output_dir + '/checkpoints', "rb") as fp:   # Unpickling
                checkpoints = pickle.load(fp)

        else:
            checkpoints = []
            
        if folder in checkpoints:
            print(f"Folder {folder}, already analysed!")
            continue

        if isinstance(folder, list):
            filenames = folder
            folder= data
        else:
            filenames = [os.path.join(folder, x) for x in os.listdir(folder)]
        

        for file in tqdm(filenames, desc="Analysing folder {}/{} -- {}".format(i+1, len(intput_folders), folder)):

            img = functions.load_image(img=file, BGR=False)

            background_labels = ["Natural", "Artificial", "Other"]

            prep_img = functions.preprocess_img(img)
            prediction = int(np.argmax(model.predict(prep_img), axis=1))

            d.append({"id": os.path.basename(os.path.splitext(file)[0]), "filename": file, "background": background_labels[prediction]})

        df = pd.DataFrame(d)        
        df.to_csv(output_dir + "/background_analysis.csv")

        progress.append(folder)
        with open(output_dir + "/checkpoints", "wb") as fp:   #Pickling
            pickle.dump(progress, fp)


    return pd.read_csv(output_dir + "/background_analysis.csv").iloc[:,1:]


def face_analysis(
    data,
    align=False,
    actions=("emotion", "age", "gender", "race"),
    models=None,
    detector_backend="retinaface",
    extract=False,
    enforce_detection=False,
    resume = False,
    return_JSON=False,
    custom_model_channel="BGR",
    custom_target_size=(224, 224),
    run_on_gpu=False,
):

    """
	This function analyzes facial attributes including age, gender, emotion and race
	Parameters:
		data: Three modes of operation: 'Single file', 'Directory of files', 'Directory of directories'
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
		extract (boolean): Determines if the faces extracted form the images will be saved in a new dir
        custom_model_channel (string): The channel used to train your custom model default is BGR set to "RGB" if needed
        custom_target_size (tuple): The input size of the custom model, default is (224, 224)
        run_on_gpu (bool): Determines if tensorflow should use GPU acceleration, default is set to False for stability, if enabled you could run into OOM error depending on available GPU VRAM.
        resume (bool): Determines if the a new experiment will be run or a previous one is resumed. When the data directory contains a folder for each pdf a checkpoint is made after each folder!
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

    # Create output folder for checkpoints
    if not os.path.exists("./Output/face_analysis/exp_1"):
        os.makedirs("./Output/face_analysis/exp_1")
        output_dir = "./Output/face_analysis/exp_1"
    else:
        numbers = [int(name.split("_")[-1]) for name in os.listdir("./Output/face_analysis")]
        number = max(numbers)
        if resume:
            output_dir = f"./Output/face_analysis/exp_{number}"
        else:
            os.makedirs(f"./Output/face_analysis/exp_{number + 1}")
            output_dir = f"./Output/face_analysis/exp_{number + 1}"

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if run_on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    DIR, intput_folders = functions.initialize_input(data)

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

    if extract != True:
        print(
            "Extracted faces will not be saved, consider setting extract_faces to True if you prefer saving that output."
        )

    d = []
    progress = []

    for i, folder in enumerate(intput_folders):
        # Check for checkpoints
        if os.path.exists(output_dir + '/checkpoints'):
            with open(output_dir + '/checkpoints', "rb") as fp:   # Unpickling
                checkpoints = pickle.load(fp)
        else:
            checkpoints = []
            
        if folder in checkpoints:
            print(f"Folder {folder}, already analysed!")
            continue

        if isinstance(folder, list):
            filenames = folder
            folder = data
        else:
            filenames = [os.path.join(folder, x) for x in os.listdir(folder)]

        for file in tqdm(filenames, desc="Analysing folder {}/{} -- {}".format(i+1, len(intput_folders), folder)):

            img = functions.load_image(img=file, BGR=True)

            deepface_ms = models.copy()
            deepface_ms.pop("age", None)

            faces = functions.detect_faces(
                img=img,
                detector_backend=detector_backend,
                align=align,
                enforce_detection=enforce_detection,
            )

            for i, (face, region) in enumerate(faces):

                if extract:
                    destination = output_dir + os.path.join( "/Faces/", os.path.basename(os.path.normpath(folder)))
                    if not os.path.exists(destination):
                        os.makedirs(destination)

                    cv2.imwrite(
                        f"{destination}/{os.path.basename(file)}_{i}.jpg",
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
                        "id": os.path.basename(os.path.splitext(file)[0]),
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

        pd.DataFrame(d).to_csv(output_dir + "/facial_analysis.csv")

        progress.append(folder)
        with open(output_dir + "/checkpoints", "wb") as fp:   #Pickling
            pickle.dump(progress, fp)
        

    if return_JSON:
        return DF_resp
    else:
        return pd.read_csv(output_dir + "/facial_analysis.csv").iloc[:,1:]


def detect_objects(
    data,
    model="custom",
    weights="yolov5l.pt",
    conf_thres=0.25,
    imgsz=640,
    save_imgs=False,
    labels=None,
    resume=False
):
    """
	This function is a wrapper for object detection based on the YoloV5 architecture by Ultralytics.
    By default it uses the COCO pre-trained weights provided by the original YoloV5 package.
    Additionally it allows for the specification of two custom models trained specifically for ImgtoVars or finally a custom model trained by the user. 
	Parameters:
		data: Three modes of operation: 'Single file', 'Directory of files', 'Directory of directories'
		model (string): specifies model options. By default it is set to "csutom" which allows the setting of custom weights.
               model can also be set to:
                    * 'c_energy' -- a custom model trained to detected the following objects: ["Crane", "Wind turbine", "farm equipment", "oil pumps", "plant chimney", "solar panels"]
                    * 'sub_open_images' -- a custom model trained on a subset of google Open Images dataset included labels are: [ "Animal", "Tree", "Plant", "Flower", "Fruit", "Suitcase", "Motorcycle", "Helicopter", "Sports Equipment", "Office Building", "Tool", "Medical Equipment", "Mug", "Sunglasses", "Headphones", "Swimwear", "Suit", "Dress", "Shirt", "Desk", "Whiteboard", "Jeans", "Helmet", "Building"]
        weights (path): determines the weights used if model is set to 'custom' by default COCO pre-trained weights are used, specify labels parameter if you are using a custom model
        conf_thres (float): 0 < conf_thresh < 1, the detection confidence cutoff
        imgsz (int): The image size for detection, default set to 640, best results are achieved if the img size is the same as the one used for training for more information chech yolov5 documentation.
        save_imgs (bool): Determines if the images on which detection was ran should be saved with the predictions overlayed. Set to False by default.
        labels (list): A list with the labels used in custom prediction, must be set when using with custom model.
        resume (bool): Determines if the a new experiment will be run or a previous one is resumed. When the data directory contains a folder for each pdf a checkpoint is made after each folder!

	Returns:
		DataFrame
        DataFrame with image paths, 
                       the predicted label,
                       the coordinates of the detected object (xywh format),
                       the confidence level of the prediction,
    """

    # Validate data file format
    if os.path.isfile(data):
        print("Operating in mode 'single file'")
        data = [data]

    elif os.path.isdir(data):
        check = [1 if os.path.isdir(os.path.join(data, x)) else 0 for x in os.listdir(data)]
        if functools.reduce(operator.mul, check, 1) == 0:
            print("Operating in mode 'directory of files'")
            data = [data]
        else:
            print("Operating in mode 'directory of directories'")
            data = [os.path.join(data, x) for x in os.listdir(data)]

    else:
        raise ValueError("Confirm that ", data, " exists and is a file or directory")

    if model == "custom":

        name = str(weights).split(".")[0]

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

    elif model == "sub_open_images":

        functions.download_asset(
            url="https://github.com/dvdimitrov13/thesis_modelzoo/releases/download/v.1.0.0/sub_oid_5l.pt"
        )

        weights = (
            functions.get_imgtovar_home() + "/.imgtovar/weights/" + "sub_oid_5l.pt"
        )
        name = "sub_open_images"

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


    elif model == "c_energy":

        functions.download_asset(
            url="https://github.com/dvdimitrov13/thesis_modelzoo/releases/download/v.1.0.0/c_energy_5l.pt"
        )

        weights = (
            functions.get_imgtovar_home() + "/.imgtovar/weights/" + "c_energy_5l.pt"
        )

        name = "c_energy"

        labels = [
            "Crane",
            "Wind turbine",
            "farm equipment",
            "oil pumps",
            "plant chimney",
            "solar panels",
        ]

    else:
        raise ValueError(
            "Provide a valid model: 'c_energy', 'sub_open_images' or 'custom' with appropriate weights"
        )

    # Create output folder for checkpoints
    if not os.path.exists(f"./Output/object_detection/{name}_exp_1"):
        os.makedirs(f"./Output/object_detection/{name}_exp_1")
        output_dir = f"./Output/object_detection/{name}_exp_1"
    else:
        output_folders = glob.glob(f"./Output/object_detection/{name}*")
        numbers = [int(o_folder.split("_")[-1]) for o_folder in output_folders]
        number = max(numbers)
        if resume:
            output_dir = f"./Output/object_detection/{name}_exp_{number}"
        else:
            os.makedirs(f"./Output/object_detection/{name}_exp_{number + 1}")
            output_dir = f"./Output/object_detection/{name}_exp_{number + 1}"

    # Introduce checkpointing       
    progress = []

    # Check for checkpoints
    if os.path.exists(output_dir + '/checkpoints'):
        with open(output_dir + '/checkpoints', "rb") as fp:   # Unpickling
            checkpoints = pickle.load(fp)
    else:
        checkpoints = []
    
    # Run detection
    for folder in data:
        folder_name = os.path.basename(os.path.normpath(folder))

        if folder_name in checkpoints:
            print(f"Folder {folder}, already analysed!")
            continue

        detect.run(
            source=folder,
            weights=weights,
            conf_thres=conf_thres,
            imgsz=imgsz,
            save_txt=True,
            name=folder_name,
            save_conf=True,
            nosave=not save_imgs,
            project= output_dir
        )

        # Track progress after each folder
        progress.append(folder_name)
        with open(output_dir + "/checkpoints", "wb") as fp:   #Pickling
            pickle.dump(progress, fp)

    # Use utility fucntion to ggregate all label txts into a df
    df = functions.build_yolo_df(labels, output_dir)
    df.to_csv(output_dir + '/object_detection.csv')
    
    return pd.read_csv(output_dir + '/object_detection.csv').iloc[:,1:]

    
    

functions.initialize_folder()

