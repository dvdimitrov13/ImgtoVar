# Adding module to path (required before publishing to pip)
from imgtovar import ImgtoVar

# df = ImgtoVar.detect_objects("./tests/test_data")
# df = ImgtoVar.detect_objects("./extract_output", model="sub_open_images", save_imgs=True, resume=True)
df = ImgtoVar.detect_objects(
    "./tests/test_data", model="c_energy", save_imgs=True
)

print(df.head(), df.shape)
