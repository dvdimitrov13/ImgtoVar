# Adding module to path (required before publishing to pip)
from imgtovar import ImgtoVar

background = ImgtoVar.background_analysis("./tests/test_data/nature_6.jpg")

print("Image background detection:")
print(background.head())

assert(background["filename"][0] is not None)
assert(background["background"][0] is not None)
assert(background["background"][0] == "Natural")