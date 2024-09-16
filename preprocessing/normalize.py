import cv2 as cv
import numpy as np
import os


# img = normalize(img)

def white_balance(img):
    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result

def CLAHE(img):
    R, G, B = cv.split(img)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1_R = clahe.apply(R)
    cl1_G = clahe.apply(G)
    cl1_B = clahe.apply(B)
    cl1 = cv.merge((cl1_R, cl1_G, cl1_B))
    return cl1

def normalize(img):
    img_wb = white_balance(img)
    img_wb_CLAHE = CLAHE(img_wb)
    return img_wb_CLAHE

