# Adding module to path (required before publishing to pip)
import sys

sys.path.insert(0, "/home/dimitar/Documents/Thesis_research/imgtovars")

from imgtovar import ImgtoVar

df = ImgtoVar.background_analysis("./tests/test_data")

print(df.head())
