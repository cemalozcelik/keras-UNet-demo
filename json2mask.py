import numpy as np
import json
import cv2
import os

def last_4chars(x):
    return(x[-7:])

file_list = os.listdir("test/jsons")
path_to_masks_folder='test/masks'

for j, filename in enumerate(sorted(file_list, key = last_4chars)):
    classes = {}
    f = open('test/jsons/'+filename,)
    data = json.load(f)
    labels=[]
    h = data['imageHeight']
    w = data['imageWidth']
    print(filename)
    c=0
    mask_full = np.zeros((h, w))
    mask2 = np.zeros((h, w))
    print(h,w)
    for i in data['shapes']:
        blobs = []
        label = i['label']
        mask = np.zeros((w, h))

        points = i['points']
        points=np.around(points)
        points=points.astype(int)
        cv2.fillPoly(mask, [points], color=(255,0,255))
        cv2.fillPoly(mask_full, [points], color=(255,0,255))
    cv2.imwrite(path_to_masks_folder + "/00" + str(j)  + ".jpg", mask_full)

