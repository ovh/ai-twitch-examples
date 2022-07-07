import streamlit as st
import requests
import base64
import io
import os
from PIL import Image


def generate_image(session, url, text, nb_image):
    result = session.post(f'{url}/dalle', json={"text": text, "num_images": nb_image})
    json = result.json()
    all_img_b64 = json['generatedImgs']
    all_img = []
    for img in all_img_b64:
        image_data = base64.decodebytes(bytes(img, 'utf8'))
        image = Image.open(io.BytesIO(image_data))
        all_img.append(image)

    return all_img


def main():
    backend_url = os.environ.get("BACKEND_URL")
    if not backend_url:
        backend_url = ''
    st.set_page_config(page_title="Image generation model", page_icon="🤖")
    st.title("Image generation model")
    session = requests.Session()
    with st.form("my_form"):
        backend_url = st.text_input("API URL", key="text", value=backend_url)
        nb_image = st.number_input("Number of images to generate", min_value=4, max_value=100, key="index")
        text = st.text_input("Describe the image that you want", key="text")
        submitted = st.form_submit_button("Submit")

        nb_column = 4

        if submitted:
            st.write("Result")
            all_img = generate_image(session, backend_url, text, nb_image)
            if all_img:
                for index in range(len(all_img)):
                    img = all_img[index]
                    col_index = index % nb_column
                    if col_index == 0:
                        cols = st.columns(nb_column)
                    cols[col_index].image(img)
            else:
                st.error("Error")


if __name__ == '__main__':
    main()
