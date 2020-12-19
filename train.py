import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

import random

def last_4chars(x):
    return(x[-7:])

image_width=128
image_height=128
image_ch=3

file_list_mask= os.listdir("train/masks")
file_list_img= os.listdir("train/images")

image_path = 'train/images/'
mask_path = 'train/masks/'
train_ids = next(os.walk(image_path))[2]
label_ids = next(os.walk(mask_path))[2]

X_train = np.zeros((len(train_ids), image_height, image_width, image_ch),dtype=np.uint8)
Y_train = np.zeros((len(train_ids), image_height, image_width, 1), dtype=np.bool)

for n, ids in tqdm(enumerate(sorted(file_list_img, key = last_4chars)), total=len(train_ids)):
    path = image_path + ids
    img = imread(path)
    img = resize(img, (image_height, image_width,1), mode='constant', preserve_range=True)
    X_train[n] = img

for n, ids in tqdm(enumerate(sorted(file_list_mask, key = last_4chars)), total=len(label_ids)):
    path = mask_path + ids
    mask = imread(path)
    mask = (resize(mask, (image_height, image_width,1), mode='constant', preserve_range=True))
    Y_train[n] = mask

image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()

from utils import get_augmented

train_gen = get_augmented(
    X_train, Y_train, batch_size=2,
    data_gen_args = dict(
        rotation_range=5.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='constant'
    ))

from keras.callbacks import ModelCheckpoint


model_filename = 'segm_model_v2.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename,
    verbose=1,
    monitor='val_loss',
    save_best_only=True,
)
from model import unet
model=unet()

from keras.optimizers import  SGD
from metrics import iou, iou_thresholded


model.compile(
    optimizer=SGD(lr=0.01, momentum=0.99),
    loss='binary_crossentropy',
    metrics=[iou, iou_thresholded]
)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=200,
    epochs=15,

    validation_data=(X_train, Y_train),
    callbacks=[callback_checkpoint]
)

from utils import plot_segm_history

plot_segm_history(history)

