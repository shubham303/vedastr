import random

import cv2
import lmdb
env = lmdb.open("/usr/datasets/synthetic_text_dataset/lmdb_dataset_Hindi/hindi/validation" , lock=False,
                readonly=True)
from PIL import Image
import six
import numpy as np
with env.begin(write=False) as txn:
    n_samples = int(txn.get('num-samples'.encode()))
    for index in range(10,n_samples):
        #index =index+random.randint(500, 1500)
        idx = index+1
        label_key = 'label-%09d'.encode() % idx
        label = txn.get(label_key).decode('utf-8')
        img_key = 'image-%09d'.encode() % idx
        imgbuf = txn.get(img_key)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')  # for color image
        img = np.array(img)
        
        cv2.imwrite("temp.jpg", img)
        print(label)
        label=None
import cv2

lines = open("/home/shubham/Documents/IIIT-ILST/Devanagari/cropped/Devanagari/WordImagesList.txt").readlines()

for i, line in enumerate(lines):
    i+=2000
    img_name , label= line.split(" ")
    img = cv2.imread("/home/shubham/Documents/IIIT-ILST/Devanagari/cropped/Devanagari/{}".format(
        img_name))
    cv2.imwrite("temp/{}_{}.jpg".format(label, i),img)
    
    