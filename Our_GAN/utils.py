#import cv2
import tensorflow as tf
from random import shuffle
import sys
import numpy as np
import glob
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

WikiArt_Emotion_DIR = "/Users/wuw/Downloads/WikiArt-Emotions/WikiArt-Emotions-Ag4.tsv"
WikiArt_DIR = "/Users/wuw/Downloads/archive"
BATCH_SIZE = 256

LABELS = {"abstract": 0, "animal-painting": 1, "cityscape": 2, "flower-painting": 3, "landscape": 4, "portrait": 5}

def load_img_list(filename):
    path = os.path.join(WikiArt_DIR, filename)
    X = pd.read_csv(path, sep='\t', header=None)
    return [name for name in X[0]]

def get_image(addr):
    image = Image.open(addr)
    image = image.resize((64, 64), resample=Image.LANCZOS)
    #image = tf.image.resize(image, [64, 64])
    image = np.asarray(image).astype("float32")
    image = (image - 127.5) / 127.5
    return image

def show_image(img):
    plt.imshow(img, interpolation='nearest')

def load_data_by_category(my_data_dir, IMG_DIR, filename):
    img_list = load_img_list(filename)
    x_data = []
    y_data = []
    for i, img_name in enumerate(img_list):
        addr = os.path.join(my_data_dir, IMG_DIR, img_name)
        img = get_image(addr)
        if img.shape != (64, 64, 3):
            print(f"Cannot read image {i}: {img_name}")
        else:
            x_data.append(img)
            y_data.append(LABELS[IMG_DIR])
    y_data = np.reshape(y_data, (len(y_data), 1))
    return np.stack(x_data), y_data

"""
def resize_image(img, dim):
	# resize image
	image = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
	return image
"""

img_flower, y_flower = load_data_by_category(WikiArt_DIR, "flower-painting", "flower.txt")
tf_flower = tf.data.Dataset.from_tensor_slices((img_flower, y_flower))

img_cityscape, y_cityscape = load_data_by_category(WikiArt_DIR, "cityscape", "cityscape.txt")
tf_cityscape = tf.data.Dataset.from_tensor_slices((img_cityscape, y_cityscape))

tf_data = tf_flower.concatenate(tf_cityscape)