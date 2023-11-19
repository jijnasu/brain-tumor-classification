# %%blocks



import gradio as gr
import numpy as np

# all pkages
# import os
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
# import matplotlib.pyplot as plt
# import seaborn as sns

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def load_and_preprocess_image(img_path, target_size=(260, 260)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

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

    # Create new model on top
    x = inputs = keras.Input(shape=(ims, ims, 3))
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    return model

ims = 260
num_classes = 4
weight_dir = './models/new_way/EfficientNetB2_2613_top_trained_fc_model.h5'
labels = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}
model = my_model()
model.load_weights(weight_dir)
model.build(keras.layers.InputLayer(input_shape=(ims, ims, 3)))


def classify_image(image):
    image = image.reshape((-1, ims, ims, 3))
    prediction = model.predict(image).flatten()
    confidences = {labels[i]: float(prediction[i]) for i in range(num_classes)}
    return confidences

img = image.load_img('image.jpg', target_size=(ims,ims))
img = image.img_to_array(img)

classify_image(img)


if __name__=="__main__":

    st.header("Brain Tumor Classification")
    st.link_button('github', 'https://github.com/jijnasu/brain-tumor-classification')
    st.divider()


    # st.write("Upload an image to classify the brain tumor.")

    uploaded_file = st.file_uploader("Upload an image to classify the brain tumor", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=False)
        st.write("")

        # Load and preprocess the uploaded image
        img_array = load_and_preprocess_image(uploaded_file)

        # Classify the image
        result = classify_image(img_array)
        st.write("Classification Result:")

        # Display individual status bars for each class
        for class_label, confidence in sorted(result.items(), key = lambda x:-x[1]):
            st.write(f"{' '.join(class_label.split('_')).title()} : {round(confidence*100, 2)} %")
            st.progress(confidence)

    st.divider()
    st.subheader('About:')
    st.write('by Kumar Jijnasu')
    st.link_button('github.com/jijnasu','https://github.com/jijnasu')
