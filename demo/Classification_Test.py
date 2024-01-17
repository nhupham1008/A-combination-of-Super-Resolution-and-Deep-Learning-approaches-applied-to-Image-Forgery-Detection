import tensorflow as tf
import tensorflow.keras as keras
#%env SM_FRAMEWORK=tf.keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm_notebook as tqdm
import joblib
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from imageio import imread
import imageio
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, UpSampling2D, Lambda, AveragePooling2D, Flatten, Dense, TimeDistributed
from tensorflow.keras.applications import ResNet50

import datetime
from sklearn import metrics
from keras import backend as K

data_df = test_data = pd.read_csv('Test_Data/Classification/class_test_data.csv')
model_path='ResNet50/ResNet50_model'
show_image=0

# load ResNet50
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

b1 = resnet50.get_layer('conv5_block3_out').output  # 16x16x2048

c1 = AveragePooling2D(pool_size=(16, 16), input_shape=(16, 16, 2048))(b1)
c2 = Flatten(input_shape=(1, 1, 2048))(c1)
output1 = Dense(2, activation='softmax')(c2)

ResNet50_model = Model(resnet50.input, output1)
for layer in ResNet50_model.layers[0:171]:
    layer.trainable = False
ResNet50_model.load_weights('ResNet50/ressnet50_model_adam_weights_33.h5')

samples = data_df.iloc[0:0+len(data_df)]
#model = tf.keras.models.load_model(model_path, compile=False)
input_img = []
input_img_array = []
mask_array = []
GT_mask = []
Predicted_mask = []
for index,row in samples.iterrows():
    #print(row['X'])
    img = load_img(row['X'])
    image = load_img(row['X'])

    #tf_image = tf.data.Dataset.from_tensor_slices(image_arr)

    mask = imread(row['y'])
    mask = np.expand_dims(mask, axis =-1)
    mask = mask/3
    mask1 = mask.astype(np.int32)
    #mask = np.concatenate([mask1, mask1, mask1], axis =2)

    #image = cv2.resize(image, (512, 512))
    #image_arr=image/255

    #image_arr = image_arr.astype(np.float32)
    image_arr = img_to_array(image)
    image_arr=np.array(image_arr).reshape(-1, 512, 512, 3)/255.0

    #output = model.predict(np.expand_dims(image_arr, axis=0))[0]
    output = ResNet50_model.predict(image_arr).reshape(2)
    output = tf.round(output, tf.int32)
    if output[0] == 1:
        s = 'Fake image'
    else:
        s = 'Pristine image'


    #image = image*255
    #image = image.astype(np.int32)

    image1 = cv2.imread(row['X'], cv2.IMREAD_GRAYSCALE)
    image1 = np.expand_dims(image1, axis =-1)
    image1 = image1/3
    image1 = image1.astype(np.int32)

    gt = np.concatenate([image1, image1, mask1], axis=2)
    input_img.append(image)
    GT_mask.append(gt)
    Predicted_mask.append(s)
    #cv2.imwrite(row['X'].split('/')[-1], final_image)
for i in range(len(data_df)):
    plt.figure(figsize=(10, 10))
    ax1=plt.subplot(1, 3, 1)
    ax1.imshow(input_img[i])
    ax1.set_title("Input Image")
    ax2=plt.subplot(1, 3, 2)
    ax2.imshow(GT_mask[i])
    ax2.set_title("Groud Truth Mask")
    ax3=plt.subplot(1, 3, 3)
    ax3.imshow(input_img[i])
    ax3.set_title(Predicted_mask[i])
plt.show()


