# Adding module to path (required before publishing to pip)
import sys
sys.path.append("/home/dimitar/Desktop/ImgtoVar")

from imgtovar import ImgtoVar

ImgtoVar.extract("./tests/test.pdf")
