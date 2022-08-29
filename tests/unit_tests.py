# Adding module to path (required before publishing to pip)
import sys

from charset_normalizer import detect

sys.path.insert(0, "/home/dimitar/Documents/Thesis_research/ImgtoVar")

from imgtovar import ImgtoVar
import warnings
import os
import math

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("-----------------------------------------")

expected_coverage = 51
num_cases = 0
succeed_cases = 0


def evaluate(condition):

    global num_cases, succeed_cases

    if condition is True:
        succeed_cases += 1
    else:
        print('---------------------------------------------')
        print('---------------------------------------------')
        print('---------------------------------------------')
        print('WARNING TEST NOT PASSED')
        print('---------------------------------------------')
        print('---------------------------------------------')
        print('---------------------------------------------')

    num_cases += 1


# ------------------------------------------------

detectors = ["opencv", "mtcnn", "retinaface", "dlib", "mediapipe", "ssd"]


def test_cases():

    print("Testing Data Cleaning pipeline functions.")
    print("-----------------------------------------")
    print("ImgtoVar.color_analysis test.")

    color_analysis = ImgtoVar.color_analysis("./tests/test_data/other_7.jpg")

    print("Color Analysis:")
    print(color_analysis.head())

    evaluate(color_analysis.filename[0] is not None)
    evaluate(color_analysis["H/L_pairs"][0] is not None)
    evaluate(color_analysis["200_dominant_pairs_%"][0] is not None)

    print("-----------------------------------------")
    print("ImgtoVar.detect_infographics test.")

    infographics = ImgtoVar.detect_infographics("./tests/test_data/infographic_6.jpg")

    print("Infographics_detection:")
    print(infographics.head())

    evaluate(infographics.filename[0] is not None)
    evaluate(infographics["chart_type"][0] is not None)
    evaluate(infographics["chart_type"][0] == "BarGraph")

    print("-----------------------------------------")
    print("ImgtoVar.detect_invertedImg test.")

    inverted = ImgtoVar.detect_invertedImg("./tests/test_data/inverted_6.jpg")

    print("Inverted images detection:")
    print(inverted.head())

    evaluate(inverted["filename"][0] is not None)
    evaluate(inverted["inverted"][0] is not None)
    evaluate(str(inverted["inverted"][0]) == "True")

    print("-----------------------------------------")
    print("Testing Image background classification")
    print("-----------------------------------------")
    print("ImgtoVar.background_analysis test.")

    background = ImgtoVar.background_analysis("./tests/test_data/nature_6.jpg")

    print("Image background detection:")
    print(background.head())

    evaluate(background["filename"][0] is not None)
    evaluate(background["background"][0] is not None)
    evaluate(background["background"][0] == "Natural")

    print("-----------------------------------------")
    print("Testing Object Detection")
    print("-----------------------------------------")
    print("ImgtoVar.detect_objects with no model specified.")

    objects = ImgtoVar.detect_objects("./tests/test_data/not_nature_16.jpg")

    print("object detection:")
    print(objects.head())

    evaluate(objects["label"][0] is not None)
    evaluate(objects["x"][0] is not None)
    evaluate(objects["y"][0] is not None)
    evaluate(objects["width"][0] is not None)
    evaluate(objects["height"][0] is not None)
    evaluate(objects["confidence"][0] is not None)

    print("-----------------------------------------")
    print("ImgtoVar.detect_objects with custom objects 'c_energy'.")

    objects = ImgtoVar.detect_objects(
        "./tests/test_data/nature_1.jpg", model="c_energy"
    )

    print("object detection:")
    print(objects.head())

    evaluate(objects["label"][0] is not None)
    evaluate(objects["x"][0] is not None)
    evaluate(objects["y"][0] is not None)
    evaluate(objects["width"][0] is not None)
    evaluate(objects["height"][0] is not None)
    evaluate(objects["confidence"][0] is not None)

    print("-----------------------------------------")
    print("ImgtoVar.detect_objects with custom objects 'sub_open_images'.")

    objects = ImgtoVar.detect_objects(
        "./tests/test_data/not_nature_3.jpg", model="sub_open_images"
    )

    print("object detection:")
    print(objects.head())

    evaluate(objects["label"][0] is not None)
    evaluate(objects["x"][0] is not None)
    evaluate(objects["y"][0] is not None)
    evaluate(objects["width"][0] is not None)
    evaluate(objects["height"][0] is not None)
    evaluate(objects["confidence"][0] is not None)

    print("-----------------------------------------")
    print("ImgtoVar.functions.detect_faces test")

    for detector in detectors:
        faces = ImgtoVar.functions.detect_faces(
            "./tests/face_dataset/img1.jpg", detector_backend=detector
        )
        evaluate((len(faces[0][1]) > 0) & (faces[0][0].shape[1] > 0))
        print(detector, " test is done")

    print("-----------------------------------------")

    print("Facial analysis test. Passing nothing as an action")

    demography = ImgtoVar.face_analysis(
        "./tests/face_dataset/img4.jpg", enforce_detection=True
    )

    evaluate(
        (demography.predicted_age[0] == "27-39")
        or (demography.predicted_age[0] == "20-26")
    )
    evaluate(demography.predicted_gender[0] == "Woman")
    evaluate(demography.shape[0] == 1)

    print("-----------------------------------------")

    print("Facial analysis test. Testing JSON output")

    demography = ImgtoVar.face_analysis(
        "./tests/face_dataset/img4.jpg", enforce_detection=True, return_JSON=True
    )

    print("Demography:")
    print(demography)

    # check response is a valid json
    print("Age: ", demography["age"])
    print("Gender: ", demography["gender"])
    print("Race: ", demography["dominant_race"])
    print("Emotion: ", demography["dominant_emotion"])

    evaluate(demography.get("age") is not None)
    evaluate(demography.get("gender") is not None)
    evaluate(demography.get("dominant_race") is not None)
    evaluate(demography.get("dominant_emotion") is not None)

    print("-----------------------------------------")

    print("Facial analysis test. Passing all to the action")

    demography = ImgtoVar.face_analysis(
        "./tests/face_dataset/img4.jpg", actions=["age", "gender", "race", "emotion"]
    )

    print("Demography:")
    print(demography.head())

    evaluate(demography.predicted_age[0] is not None)
    evaluate(demography.predicted_gender[0] is not None)
    evaluate(demography.predicted_race[0] is not None)
    evaluate(demography.predicted_emotion[0] is not None)

    print("-----------------------------------------")

    print("Facial analysis test 2. Remove some actions and check they are not computed")
    demography = ImgtoVar.face_analysis(
        "./tests/face_dataset/img4.jpg", actions=["age", "gender"]
    )

    print(demography)

    evaluate(demography.predicted_age[0] is not None)
    evaluate(demography.predicted_gender[0] is not None)
    evaluate(math.isnan(demography.predicted_race[0]))
    evaluate(math.isnan(demography.predicted_emotion[0]))

    print("-----------------------------------------")


# ---------------------------------------------

test_cases()

print("num of test cases run: " + str(num_cases))
print("succeeded test cases: " + str(succeed_cases))

test_score = (100 * succeed_cases) / num_cases

print("test coverage: " + str(test_score))

if test_score > expected_coverage:
    print("well done! min required test coverage is satisfied")
else:
    print("min required test coverage is NOT satisfied")

assert test_score > expected_coverage
