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


def Build_ResNet50_and_mUNet():
    """ Input """
    inputs = Input(shape=(512, 512, 3))

    """ Pretrained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    b1 = resnet50.get_layer('conv5_block3_out').output  # 16x16x2048

    c1 = AveragePooling2D(pool_size=(16, 16), input_shape=(16, 16, 2048))(b1)
    c2 = Flatten(input_shape=(1, 1, 2048))(c1)
    output1 = Dense(2, activation='softmax', name="output1")(c2)
    ResNet50_model = Model(inputs, output1)
    ResNet50_model.load_weights('ResNet50/ressnet50_model_adam_weights_33.h5')
    output1 = ResNet50_model.output

    """ Encoder """
    s1 = ResNet50_model.get_layer('conv1_relu').output  # 256x256x64
    s2 = ResNet50_model.get_layer('conv2_block3_out').output  # 128x128x256
    s3 = ResNet50_model.get_layer('conv3_block4_out').output  # 64x64x512
    s4 = ResNet50_model.get_layer('conv4_block6_out').output  # 32x32x1024

    """ Brigde """
    b1 = ResNet50_model.get_layer('conv5_block3_out').output  # 16x16x2048

    """ Decoder """
    d1 = Decoder_Block(b1, s4, 1024)  # 32x32x1024
    d2 = Decoder_Block(d1, s3, 512)  # 64x64x512
    d3 = Decoder_Block(d2, s2, 256)  # 128x128x256
    d4 = Decoder_Block(d3, s1, 64)  # 256x256x64

    """ Upsampling """
    u = UpSampling2D((2, 2))(d4)  # 512x512x64

    """ Output """
    output2 = Conv2D(2, (1, 1), padding='same', activation='softmax', name="output2")(u)  # 512x512x2

    mUNet = Model(ResNet50_model.input, output2)
    mUNet.load_weights('mUNet/mUNet_weights.h5')
    output2 = mUNet.output

    model_1 = Model(inputs=mUNet.input, outputs=[output1, output2])
    model_1.load_weights('mUNet/umUNet/umUNet_weights_7.h5')

    x = tf.round(model_1.output[0])
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(x, axis=1)
    z = tf.tile(y, multiples=[1, 512, 512, 1])
    output = tf.add_n([tf.math.multiply(z, model_1.output[1] * [1, -1]), model_1.output[1] * [0, 1], z * [0, 1]],
                      name='output')

    model = Model(inputs=model_1.input, outputs=output)
    return model

def print_results(model_mUNet, model_ResNet50_mUNet,data_df):
    """This function prints the input image,its ground truth mask along with predicted mask
        model_path : Takes the path of trianed model
        split_data_df : Takes the pd.DataFrame object that should contain image path and mask path in X and Y named columns
        show_images : number of random images that should sampled from "split_data_df"
    """
    samples = data_df.iloc[0:0+len(data_df)]
    #model_mUNet = tf.keras.models.load_model(model_mUNet, compile=False)
    #model_umUNet = tf.keras.models.load_model(model_umUNet, compile=False)
    input_img = []
    input_img_array = []
    mask_array = []
    GT_mask = []
    Predicted_mUNet = []
    Predicted_ResNet50_mUNet = []
    for index,row in samples.iterrows():
        img = load_img(row['X'])
        image = load_img(row['X'])

        #tf_image = tf.data.Dataset.from_tensor_slices(image_arr)

        mask = imread(row['y'])
        mask = np.expand_dims(mask, axis =-1)
        mask = mask/3
        mask1 = mask.astype(np.int32)

        #image = cv2.resize(image, (512, 512))
        #image_arr=image/255

        #image_arr = image_arr.astype(np.float32)
        image_arr = img_to_array(image)
        image_arr=np.array(image_arr).reshape(-1, 512, 512, 3)/255.0

        #output = model.predict(np.expand_dims(image_arr, axis=0))[0]
        output_mUNet = model_mUNet.predict(image_arr).reshape(512,512,2)
        output_mUNet = np.argmax(output_mUNet, axis=-1)
        output_mUNet = np.expand_dims(output_mUNet, axis=-1)
        output_mUNet = output_mUNet*(255/3)
        output_mUNet = output_mUNet.astype(np.int32)

        output_ResNet50_mUNet = model_ResNet50_mUNet.predict(image_arr).reshape(512,512,2)
        output_ResNet50_mUNet = np.argmax(output_ResNet50_mUNet, axis=-1)
        output_ResNet50_mUNet = np.expand_dims(output_ResNet50_mUNet, axis=-1)
        output_ResNet50_mUNet = output_ResNet50_mUNet*(255/3)
        output_ResNet50_mUNet = output_ResNet50_mUNet.astype(np.int32)
        #image = image*255
        #image = image.astype(np.int32)

        image1 = cv2.imread(row['X'], cv2.IMREAD_GRAYSCALE)
        h, w = image1.shape
        line = np.ones((30, w, 3))*255
        image1 = np.expand_dims(image1, axis =-1)
        image1 = image1/3
        image1 = image1.astype(np.int32)

        gt = np.concatenate([image1, image1, mask1], axis=2)
        pred_mUNet = np.concatenate([ image1, image1, output_mUNet], axis=2)
        pred_ResNet50_mUNet = np.concatenate([ image1, image1, output_ResNet50_mUNet], axis=2)

        #final_image = np.concatenate([image, line, pred], axis=-3)
        input_img.append(image)
        GT_mask.append(gt)
        Predicted_mUNet.append(pred_mUNet)
        Predicted_ResNet50_mUNet.append(pred_ResNet50_mUNet)
        #cv2.imwrite(row['X'].split('/')[-1], final_image)

    plt.figure(figsize=(20, 20))
    ax1 = plt.subplot(int(len(data_df)/2), 4,  1)
    ax1.set_title("Input Image")
    ax2 = plt.subplot(int(len(data_df)/2), 4,  2)
    ax2.set_title("Groud Truth Mask")
    ax3 = plt.subplot(int(len(data_df)/2), 4,  3)
    ax3.set_title("Predicted mUNet")
    ax4 = plt.subplot(int(len(data_df)/2), 4,  4)
    ax4.set_title("Predicted ResNet50 + mUNet")
    for i in range(len(data_df)):
        if i >= int(len(data_df)/2):
            k=i-3
        else:
            k=i
        if i == int(len(data_df)/2):
            plt.figure(figsize=(20, 20))
            ax1 = plt.subplot(int(len(data_df) / 2), 4, 1)
            ax1.set_title("Input Image")
            ax2 = plt.subplot(int(len(data_df) / 2), 4, 2)
            ax2.set_title("Groud Truth Mask")
            ax3 = plt.subplot(int(len(data_df) / 2), 4, 3)
            ax3.set_title("Predicted mUNet")
            ax4 = plt.subplot(int(len(data_df) / 2), 4, 4)
            ax4.set_title("Predicted ResNet50 + mUNet")

        ax1 = plt.subplot(int(len(data_df) / 2), 4, 4*k+1)
        ax1.imshow(input_img[i])
        #ax1.set_ylabel("Input Image")
        ax2=plt.subplot(int(len(data_df) / 2), 4, 4*k+2)
        ax2.imshow(GT_mask[i])
        #ax2.set_ylabel("Groud Truth Mask")
        ax3=plt.subplot(int(len(data_df) / 2), 4, 4*k+3)
        ax3.imshow(Predicted_mUNet[i])
        #ax3.set_ylabel("Predicted Mask")
        ax4 = plt.subplot(int(len(data_df) / 2), 4, 4 * k + 4)
        ax4.imshow(Predicted_ResNet50_mUNet[i])
    plt.show()

if __name__ == '__main__':
    data_df = test_data = pd.read_csv('Test_Data/Segmentation/mUNet_vs_ResNet50_mUNet/mUNet_vs_ResNet50_mUNet_test_data.csv')
    mUNet = Build_ResNet50_UNet()
    mUNet.load_weights('mUNet/mUNet_weights.h5')
    ResNet50_mUNet = Build_ResNet50_and_mUNet()
    ResNet50_mUNet.load_weights('ResNet_mUNet/ResNet50_mUNet_weights.h5')
    print_results(mUNet, ResNet50_mUNet, data_df)