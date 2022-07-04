# Adding module to path (required before publishing to pip)
import sys

sys.path.insert(0, "/home/dimitar/Documents/Thesis_research/ImgtoVar")

from imgtovar import ImgtoVar

# Slow speed comes mainly from face detection, marginal improvements on inference side
df = ImgtoVar.face_analysis("./tests/dataset/", enforce_detection=False)

## I could write a test to see if all the columns are there, or any other thing i would normally check by hand

print(df.head())
