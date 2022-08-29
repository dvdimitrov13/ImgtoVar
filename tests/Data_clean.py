# Adding module to path (required before publishing to pip)
import sys

sys.path.insert(0, "/home/dimitar/Documents/Thesis_research/ImgtoVar")

from imgtovar import ImgtoVar

# df = ImgtoVar.color_analysis("./extract_output", extract=True, resume=True)
# df = ImgtoVar.detect_infographics("./extract_output", extract=True, resume=True)
df = ImgtoVar.detect_invertedImg("./extract_output", extract=True)


print(df.head())
