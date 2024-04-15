# %%blocks



# import gradio as gr
import numpy as np
import os

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
# from tensorflow.keras.layers.experimental import preprocessing

# import matplotlib.pyplot as plt
# import seaborn as sns


import streamlit as st
import streamlit.components.v1 as components
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print('-'*50)
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    # print(gpu)
print('-'*50)

def load_and_preprocess_image(img_path, target_size=(260, 260)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

def classify_image(image, model):
    image = image.reshape((-1, ims, ims, 3))
    prediction = model.predict(image).flatten()
    confidences = {labels[i]: float(prediction[i]) for i in range(num_classes)}
    return confidences


@st.cache_resource(max_entries=1, hash_funcs={"MyUnhashableClass": lambda _: None})
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

    # running for the first time
    img = image.load_img('image.jpg', target_size=(ims,ims))
    img = image.img_to_array(img)
    classify_image(img, model)

    return model

ims = 260
num_classes = 4
weight_dir = '/workspaces/brain-tumor-classification/models/new_way/EfficientNetB2_2613_top_trained_fc_model.h5'
# weight_dir = './models/new_way/EfficientNetB2_2613_top_trained_fc_model.h5'
labels = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}
model = my_model()
model.load_weights(weight_dir)
model.build(keras.layers.InputLayer(input_shape=(ims, ims, 3)))




if __name__=="__main__":

    st.title("Brain Tumor Classification")
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
        result = classify_image(img_array, model)
        st.write("Classification Result:")

        # Display individual status bars for each class
        for class_label, confidence in sorted(result.items(), key = lambda x:-x[1]):
            st.write(f"{' '.join(class_label.split('_')).title()} : {round(confidence*100, 2)} %")
            st.progress(confidence)


    st.divider()
    st.subheader('About:')
    st.write('by Kumar Jijnasu')
    st.link_button('github.com/jijnasu','https://github.com/jijnasu')
    st.write('Other projects')
    st.link_button('Weapon Detection', 'https://weapon-detection-kj.streamlit.app')




# ==================================================================

    # st.divider()
    
    # test_img_dir = "Testing/glioma_tumor/"
    # sample_images = ["image.jpg", "image(1).jpg", "image(2).jpg", "image(3).jpg"]
    # sample_images = [os.path.join(test_img_dir, image) for image in sample_images]
# ==================================================================
    # with st.container():
    #     # components
    #     components.html(
    #         """







    #             <style>
    #                 body {
    #                     overflow-x: hidden; /* Hide horizontal scrollbar on the body */
    #                 }

    #                 .image-container {
    #                     display: flex;
    #                     overflow-x: auto; /* Enable horizontal scrollbar for the container */
    #                     white-space: nowrap; /* Prevent images from wrapping to the next line */
    #                 }

    #                 .image-container img {
    #                     width: 300px; /* Adjust the width of the images as needed */
    #                     height: auto;
    #                     margin-right: 10px; /* Adjust the margin between images as needed */
    #                 }
    #             </style>



    #             <div>
    #                 <img src="image.jpg" alt="Image 1">
    #             </div>



    #         """,
    #     )

    
    # # with st.container():
    # #     st.markdown('<div class="image-container">', unsafe_allow_html=True)
    # #     for image_path in image_paths:
    # #         st.image(image_path, width=100, caption=os.path.basename(image_path), use_column_width=False, output_format="auto")
    # #     st.markdown('</div>', unsafe_allow_html=True)
    # # with st.container():
    #     # st.components.v1.streamlit.image
    #     components.html('<div class="image-container">')
    #     for image_path in sample_images:
    #         st.image(image_path, width=100, caption=os.path.basename(image_path), use_column_width=False, output_format="auto")
    #     components.html('</div>')



    # # Custom HTML and CSS for horizontally scrolling image gallery
    # image_gallery_html = f"""
    #     <div class="image-container">
    #         {" ".join([f'<img src="{image}" alt="Image"></img>' for image in sample_images])}
    #     </div>
    # """

    # # Bootstrap 4 collapse example with added image gallery
    # components.html(
    #     f"""
    #     <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    #     <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    #     <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
        
    #     <div id="accordion">
    #     <div class="card">
    #         <div class="card-header" id="headingOne">
    #         <h5 class="mb-0">
    #             <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
    #             Collapsible Group Item #1
    #             </button>
    #         </h5>
    #         </div>
    #         <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
    #         <div class="card-body">
    #             {image_gallery_html}
    #         </div>
    #         </div>
    #     </div>
    #     <div class="card">
    #         <div class="card-header" id="headingTwo">
    #         <h5 class="mb-0">
    #             <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
    #             Collapsible Group Item #2
    #             </button>
    #         </h5>
    #         </div>
    #         <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
    #         <div class="card-body">
    #             {image_gallery_html}
    #         </div>
    #         </div>
    #     </div>
    #     </div>
    #     """
    # )

    # def horizontal_image_gallery(image_paths):
    #     # Display images horizontally using st.image
    #     st.markdown(
    #         """
    #         <style>
    #             .collapsible {
    #                 cursor: pointer;
    #                 padding: 18px;
    #                 width: 100%;
    #                 border: none;
    #                 text-align: left;
    #                 outline: none;
    #                 font-size: 15px;
    #             }

    #             .content {
    #                 display: none;
    #                 overflow: hidden;
    #             }

    #             .image-container {
    #                 display: flex;
    #                 overflow-x: auto;
    #                 white-space: nowrap;
    #             }

    #             .image-container img {
    #                 width: 300px;
    #                 height: auto;
    #                 margin-right: 10px;
    #             }
    #         </style>
    #         """,
    #         unsafe_allow_html=True
    #     )

    #     for i, image_path in enumerate(image_paths):
    #         st.markdown(
    #             f"""
    #             <button class="collapsible" onclick="toggleContent({i})">Collapsible Group Item #{i + 1}</button>
    #             <div class="content" id="content{i}">
    #                 <div class="image-container">
    #                     <img src="{image_path}" alt="Image">
    #                 </div>
    #             </div>
    #             """,
    #             unsafe_allow_html=True
    #         )
    # horizontal_image_gallery(sample_images)
# ========================================================================
    # Function to create the custom HTML with embedded CSS and JavaScript
    # def horizontal_image_gallery(image_paths):
    #     # Display images horizontally using st.image
    #     st.markdown(
    #         """
    #         <style>
    #             .image-container {
    #                 display: flex;
    #                 overflow-x: auto;
    #                 white-space: nowrap;
    #             }

    #             .image-container img {
    #                 width: 300px;
    #                 height: auto;
    #                 margin-right: 10px;
    #             }
    #         </style>
    #         Example
    #         """,
    #         unsafe_allow_html=True
    #     )

    #     with st.container():
    #         st.markdown('<div class="image-container">', unsafe_allow_html=True)
    #         for image_path in image_paths:
    #             st.image(image_path, width=100, caption=os.path.basename(image_path), use_column_width=False, output_format="auto")
    #         st.markdown('</div>', unsafe_allow_html=True)

    # # Sample image paths
    # test_img_dir = "Testing/glioma_tumor/"
    # sample_images = ["image.jpg", "image(1).jpg", "image(2).jpg", "image(3).jpg"]

    # # Construct absolute paths
    # sample_images = [os.path.join(test_img_dir, image) for image in sample_images]

    # # Display the horizontally scrollable image gallery
    # horizontal_image_gallery(sample_images)

    # components.html()