import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('elmaGRAB.png',0)
size_of_image=img.shape[0]*img.shape[1]
print("size of image : " , size_of_image)
black_pixels_count = size_of_image - cv2.countNonZero(img)
print("black pixels count : ",black_pixels_count)
