## This tests are based on DeepFace need to be reworked and expanded for ImgtoVar

import warnings
import os
import tensorflow as tf
import cv2
from deepface import DeepFace

print("-----------------------------------------")

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf_major_version = int(tf.__version__.split(".")[0])

if tf_major_version == 2:
    import logging

    tf.get_logger().setLevel(logging.ERROR)

print("Running unit tests for TF ", tf.__version__)

print("-----------------------------------------")

expected_coverage = 97
num_cases = 0
succeed_cases = 0


def evaluate(condition):

    global num_cases, succeed_cases

    if condition is True:
        succeed_cases += 1

    num_cases += 1


# ------------------------------------------------

detectors = ["opencv", "mtcnn", "retinaface"]

print("-----------------------------------------")


def test_cases():

    print("DeepFace.detectFace test")

    for detector in detectors:
        img = DeepFace.detectFace("dataset/img11.jpg", detector_backend=detector)
        evaluate(img.shape[0] > 0 and img.shape[1] > 0)
        print(detector, " test is done")

    print("-----------------------------------------")

    img_path = "dataset/img1.jpg"
    embedding = DeepFace.represent(img_path)
    print("Function returned ", len(embedding), "dimensional vector")
    evaluate(len(embedding) > 0)

    print("-----------------------------------------")

    print("Face detectors test")

    for detector in detectors:
        print(detector + " detector")
        res = DeepFace.verify(dataset[0][0], dataset[0][1], detector_backend=detector)
        print(res)
        assert res["verified"] == dataset[0][2]

    print("-----------------------------------------")

    print("Facial analysis test. Passing nothing as an action")

    img = "dataset/img4.jpg"
    demography = DeepFace.analyze(img)
    print(demography)

    evaluate(demography["age"] > 20 and demography["age"] < 40)
    evaluate(demography["dominant_gender"] == "Woman")

    print("-----------------------------------------")

    print("Facial analysis test. Passing all to the action")
    demography = DeepFace.analyze(img, ["age", "gender", "race", "emotion"])

    print("Demography:")
    print(demography)

    # check response is a valid json
    print("Age: ", demography["age"])
    print("Gender: ", demography["dominant_gender"])
    print("Race: ", demography["dominant_race"])
    print("Emotion: ", demography["dominant_emotion"])

    evaluate(demography.get("age") is not None)
    evaluate(demography.get("dominant_gender") is not None)
    evaluate(demography.get("dominant_race") is not None)
    evaluate(demography.get("dominant_emotion") is not None)

    print("-----------------------------------------")

    print("Facial analysis test 2. Remove some actions and check they are not computed")
    demography = DeepFace.analyze(img, ["age", "gender"])

    print("Age: ", demography.get("age"))
    print("Gender: ", demography.get("dominant_gender"))
    print("Race: ", demography.get("dominant_race"))
    print("Emotion: ", demography.get("dominant_emotion"))

    evaluate(demography.get("age") is not None)
    evaluate(demography.get("dominant_gender") is not None)
    evaluate(demography.get("dominant_race") is None)
    evaluate(demography.get("dominant_emotion") is None)

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
