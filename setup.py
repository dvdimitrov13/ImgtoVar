from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

DESCRIPTION = "Extracting structured variables from image data"
LONG_DESCRIPTION = "A package that allows researchers to analyse unstructured image data by extracting a range of features."

# Setting up
setup(
    name="imgtovar",
    version="0.5",
    author="Dimitar Dimitrov",
    author_email="dvdimitrov13@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    url="https://github.com/dvdimitrov13/ImgtoVar",
    keywords=[
        "python",
        "research",
        "image data",
        "extract variables",
        "object detection",
        "background detection",
        "face detection",
        "facial attribute analysis",
        "chart detection",
        "graph detection",
        "image color analysis",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7.0",
    install_requires=[
        "tensorflow==2.4.0",
        "keras>=2.9.0",
        "deepface>=0.0.75",
        "yolov5>=6.1.5",
        "PyMuPDF>=1.20.1",
        "dlib>=19.24.0",
        "mediapipe>=0.8.10.1",
    ],
)
