# Adding module to path (required before publishing to pip)
import sys
sys.path.append("/home/dimitar/Desktop/ImgtoVar")

from imgtovar import ImgtoVar

# df = ImgtoVar.color_analysis("./extract_output", extract=True, resume=True)
# df = ImgtoVar.detect_infographics("./tests/test_data/infographic_01.jpg", extract=False, resume=True)
df = ImgtoVar.detect_invertedImg("./extract_output", extract=True, run_on_gpu=True)

print(df.head())