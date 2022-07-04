# Adding module to path (required before publishing to pip)
import sys

sys.path.insert(0, "/home/dimitar/Documents/Thesis_research/ImgtoVar")

from imgtovar import ImgtoVar

# df = ImgtoVar.color_analysis("./tests/test_data")
# df = ImgtoVar.detect_infographics("./tests/test_data", extract_infographics=True)
df = ImgtoVar.detect_invertedImg("./tests/test_data", filter_inverted=True)


print(df.head())
