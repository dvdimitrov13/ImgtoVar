# Adding module to path (required before publishing to pip)
import sys
sys.path.append("/home/dimitar/Desktop/ImgtoVar")

from imgtovar import ImgtoVar

print(ImgtoVar.__file__)

background = ImgtoVar.background_analysis("./tests/test_data/nature_6.jpg")

print("Image background detection:")
print(background.head())

assert(background["filename"][0] is not None)
assert(background["background"][0] is not None)
assert(background["background"][0] == "Natural")