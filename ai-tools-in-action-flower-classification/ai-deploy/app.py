"""
Streamlit web app for flower image classification
"""

# import dependences
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model


# load the model
@st.cache(allow_output_mutation=True)
def modelLoading():
    model = load_model('saved_model/my_model')
    return model


# classify the uploaded image
def modelPrediction(img_array, image_size, model):

    # resize image
    img = cv2.resize(img_array, (image_size, image_size))
    img = img.reshape(1, image_size, image_size, 3)

    # classes name
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    # get prediction
    predictions = model.predict(img)
    class_id = np.argmax(predictions, axis = 1)
    result = class_names[class_id.item()]

    return result


# main
if __name__ == '__main__':

    # load model
    classification_model = modelLoading()

    # upload image
    image_file = st.file_uploader("Choose a flower image", type=["png", "jpg", "jpeg"])

    if image_file is not None:

        # load, display and convert image
        display_img = Image.open(image_file)
        st.image(display_img)
        img_array = np.array(Image.open(image_file))
        
        # display classification results
        st.write('### Classification result')
        # if you select the predict button
        if st.button('Classify'):
            # write the prediction: the prediction of the last sound sent corresponds to the first column
            st.write("The flower image correspond to: ", str(modelPrediction(img_array, 224, classification_model)))
    else:
        st.write("No image has been uploaded yet.")
