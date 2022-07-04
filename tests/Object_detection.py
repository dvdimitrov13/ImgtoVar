# Adding module to path (required before publishing to pip)
import sys

from charset_normalizer import detect

sys.path.insert(0, "/home/dimitar/Documents/Thesis_research/ImgtoVar")

from imgtovar import ImgtoVar

# df = ImgtoVar.detect_objects("./tests/test_data")
# df = ImgtoVar.detect_objects("./tests/test_data", model="c_energy", save_imgs=True)
df = ImgtoVar.detect_objects(
    "./tests/test_data", model="sub_open_images", save_imgs=True
)

print(df.head(), df.shape)
