# Adding module to path (required before publishing to pip)
import sys

sys.path.insert(0, "/home/dimitar/Documents/Thesis_research/ImgtoVar")

from imgtovar import ImgtoVar

df = ImgtoVar.background_analysis("./extract_output")

print(df.head(30))
