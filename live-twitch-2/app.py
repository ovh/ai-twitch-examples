# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# import dependences
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model


# load image
def load_image(image_file):
    img = Image.open(image_file)
    return img


# convert image to array
def img_to_array(img):
    img_base = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_base)
    img_array = np.array([img_array])
    return img_array


# classify the uploded image
def classification(img_array, model):
    # classes name
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    # load the pretrained model
    model = load_model('saved_model/my_model')
    predictions = model.predict(img_array)
    class_id = np.argmax(predictions, axis = 1)
    result = class_names[class_id.item()]
    return result


# main
if __name__ == '__main__':
    background = Image.open("flower-class-logo.png")
    col1, col2, col3 = st.columns([0.2, 2, 0.2])
    col2.image(background, use_column_width=True)
    st.write('## Welcome on the flower classification app!')
    st.write('___')
    st.write("##### This AI is able to classify 5 flower types:")
    st.write("rose, tulip, sunflower, dandelion, daisy")

    # upload img
    image_file = st.file_uploader("Choose a flower image", type=["png", "jpg", "jpeg"])
    classification_model = load_model('saved_model/my_model')

    if image_file is not None:

        # load image
        st.image(load_image(image_file), width=250)

        # convert image to array
        img_array = img_to_array(image_file)

        st.write('### Classification result')

        # if you select the predict button
        if st.button('Classify'):
            # write the prediction: the prediction of the last sound sent corresponds to the first column
            st.write("The flower image correspond to: ", str(classification(img_array, classification_model)))
    else:
        st.write("No image has been uploaded yet.")