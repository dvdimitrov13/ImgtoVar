# Adding module to path (required before publishing to pip)
import sys
sys.path.append("/home/dimitar/Desktop/ImgtoVar")

from imgtovar import ImgtoVar

# Slow speed comes mainly from face detection, marginal improvements on inference side
df = ImgtoVar.face_analysis("./tests/face_dataset", enforce_detection=False, extract=True, run_on_gpu=False, actions=("age", "gender"))

## I could write a test to see if all the columns are there, or any other thing i would normally check by hand

print(df.head())
