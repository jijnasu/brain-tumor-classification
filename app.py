# %%blocks

import gradio as gr
import numpy as np

# all pkages
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, optimizers, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers.experimental import preprocessing


def my_model():
    # data_augmentation = tf.keras.Sequential([
    #     layers.RandomCrop(height=224, width=224),      
    #     layers.RandomFlip("horizontal_and_vertical"),  
    #     layers.RandomTranslation(height_factor=0.2, width_factor=0.2),  
    #     layers.RandomRotation(factor=0.2),              
    #     layers.RandomZoom(height_factor=0.2, width_factor=0.2),  
    #     layers.RandomContrast(factor=0.2),              
    #     layers.RandomBrightness(factor=0.2)             
    # ])

    base_model = keras.applications.EfficientNetB2(
        weights="imagenet",
        input_shape=(ims, ims, 3),
        include_top=False,
    )

    # Freeze the base_model
    # base_model.trainable = False

    # scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    # scale_layer = tf.keras.applications.resnet.preprocess_input


    # Create new model on top
    x = inputs = keras.Input(shape=(ims, ims, 3))
    # x = data_augmentation(inputs)
    # x = scale_layer(x)
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    # x = keras.layers.Dense(256, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)


#     model = Sequential()
#     model.add(base_model)
#     model.add(GlobalAveragePooling2D())
# #     model.add(Dense(900, activation='relu', kernel_regularizer=l2(0.001)))
#     model.add(Dropout(0.5))
# #     model.add(Dense(200, activation='relu'))#, kernel_regularizer=l2(0.001)))
#     model.add(Dense(num_classes, activation='softmax'))

    return model

ims = 260
num_classes = 4
weight_dir = './models/new_way/EfficientNetB2_2613_top_trained_fc_model.h5'
labels = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}
model = my_model()
model.load_weights(weight_dir)
model.build(keras.layers.InputLayer(input_shape=(ims, ims, 3)))


# import requests

# Download human-readable labels for ImageNet.
# response = requests.get("https://git.io/JJkYN")
# labels = response.text.split("\n")

def classify_image(image):
    image = image.reshape((-1, ims, ims, 3))
    # image = tf.keras.applications.EfficientNetB2.preprocess_input(image)
    prediction = model.predict(image).flatten()
    confidences = {labels[i]: float(prediction[i]) for i in range(num_classes)}
    return confidences

img = image.load_img('image.jpg', target_size=(ims,ims))
img = image.img_to_array(img)

classify_image(img)

gr.Interface(fn=classify_image,
             inputs=gr.Image(shape=(ims, ims)),
             outputs=gr.Label(num_top_classes=num_classes),
            #  examples=["banana.jpg", "car.jpg"]
             ).launch(share=True)

