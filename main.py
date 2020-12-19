import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

def last_4chars(x):
    return(x[-7:])

image_width=128
image_height=128
image_ch=3

file_list_test= os.listdir("test/images/")
image_path_test = 'test/images/'
test_ids = next(os.walk(image_path_test))[2]
X_test = np.zeros((len(test_ids), image_height, image_width, image_ch),dtype=np.uint8)

file_list_mask= os.listdir("test/masks/")
mask_path = 'test/masks/'
mask_ids = next(os.walk(mask_path))[2]
Y_test = np.zeros((len(mask_ids), image_height, image_width, 1), dtype=np.bool)


for n, ids in tqdm(enumerate(sorted(file_list_mask, key = last_4chars)), total=len(mask_ids)):
    path = mask_path + ids
    mask = imread(path)
    mask = (resize(mask, (image_height, image_width,1), mode='constant', preserve_range=True))
    Y_test[n] = mask

for n, ids in tqdm(enumerate(sorted(file_list_test, key = last_4chars)), total=len(test_ids)):

    path = image_path_test + ids
    img = imread(path)
    img = resize(img, (image_height, image_width,1), mode='constant', preserve_range=True)
    X_test[n] = img
from utils import plot_imgs

model = tf.keras.models.load_model("segm_model_v3.h5")
y_pred = model.predict(X_test)

plot_imgs(org_imgs=X_test, mask_imgs=Y_test, pred_imgs=y_pred, nm_img_to_plot=4)

plt.show()