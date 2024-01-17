import tensorflow as tf
from tensorflow import keras
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, UpSampling2D, Lambda, AveragePooling2D, Flatten, Dense, TimeDistributed
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
#import tensorflow_addons as tfa
import datetime
from sklearn import metrics
from keras import backend as K

def Conv_Block(inputs, num_filters):
  x = Conv2D(num_filters, 3, padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2D(num_filters, 3, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  return x

def Decoder_Block(inputs, skip_features, num_filters):
  x = Conv2D(int(num_filters/2), 1, padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2DTranspose(int(num_filters/2), (4, 4), strides=2, padding='same')(x)

  x = Conv2D(num_filters, 1, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Concatenate()([x, skip_features])
  x = Conv_Block(x, num_filters)
  return x

def Build_ResNet50_UNet():
  """ Input """
  inputs = Input(shape=(512,512,3))

  """ Pretrained ResNet50 Model """
  resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

  """ Encoder """
  s1 = resnet50.get_layer('conv1_relu').output        #256x256x64
  s2 = resnet50.get_layer('conv2_block3_out').output  #128x128x256
  s3 = resnet50.get_layer('conv3_block4_out').output  #64x64x512
  s4 = resnet50.get_layer('conv4_block6_out').output  #32x32x1024

  """ Brigde """
  b1 = resnet50.get_layer('conv5_block3_out').output  #16x16x2048

  """ Decoder """
  d1 = Decoder_Block(b1, s4, 1024)                    #32x32x1024
  d2 = Decoder_Block(d1, s3, 512)                     #64x64x512
  d3 = Decoder_Block(d2, s2, 256)                     #128x128x256
  d4 = Decoder_Block(d3, s1, 64)                      #256x256x64

  """ Upsampling """
  u = UpSampling2D((2, 2))(d4)                        #512x512x64

  """ Output """
  o = Conv2D(2, (1,1), padding='same', activation='softmax')(u)  #512x512x2

  model = Model(inputs, o)
  for layer in model.layers[0:171]:
    layer.trainable=False

  return model

if __name__ == '__main__':
    data_df = test_data = pd.read_csv('Test_Data/Segmentation/mUNet/mUNet_test_data.csv')
    mUNet = Build_ResNet50_UNet()
    mUNet.load_weights('mUNet/mUNet_weights.h5')
    samples = data_df.iloc[0:0+len(data_df)]
    #model = keras.models.load_model(model_path, compile=False)
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
        output = mUNet.predict(image_arr).reshape(512, 512, 2)
        output = np.argmax(output, axis=-1)
        output = np.expand_dims(output, axis=-1)
        output = output * (255 / 3)
        output1 = output.astype(np.int32)

        #image = image*255
        #image = image.astype(np.int32)

        image1 = cv2.imread(row['X'], cv2.IMREAD_GRAYSCALE)
        image1 = np.expand_dims(image1, axis =-1)
        image1 = image1/3
        image1 = image1.astype(np.int32)

        gt = np.concatenate([image1, image1, mask1], axis=2)
        pred = np.concatenate([image1, image1, output1], axis=2)
        input_img.append(image)
        GT_mask.append(gt)
        Predicted_mask.append(pred)
        #cv2.imwrite(row['X'].split('/')[-1], final_image)
    plt.figure(figsize=(20, 20))
    ax1 = plt.subplot(3, len(data_df),  1)
    ax1.set_ylabel("Input Image")
    ax2 = plt.subplot(3, len(data_df),  1 + 6)
    ax2.set_ylabel("Groud Truth Mask")
    ax3 = plt.subplot(3, len(data_df), 1 + 2 * 6)
    ax3.set_ylabel("Predicted Mask")
    for i in range(len(data_df)):

        ax1=plt.subplot(3, len(data_df), i+1)
        ax1.imshow(input_img[i])
        #ax1.set_ylabel("Input Image")
        ax2=plt.subplot(3, len(data_df), i+1+6)
        ax2.imshow(GT_mask[i])
        #ax2.set_ylabel("Groud Truth Mask")
        ax3=plt.subplot(3, len(data_df), 1+i+2*6)
        ax3.imshow(Predicted_mask[i])
        #ax3.set_ylabel("Predicted Mask")
    plt.show()