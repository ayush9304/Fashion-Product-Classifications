import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from utils.custom_metrics import f1_score
from utils.class_labels import COLOURS, PRODUCT_TYPES, GENDERS, SEASONS
import cv2

# Load the model
model = tf.keras.models.load_model('models/model.h5', custom_objects={'f1_score':f1_score})

# Define the Streamlit app
def app():

    st.title('Fashion Product Image Classification')

    # Allow the user to upload a file
    file = st.file_uploader('Please upload an image file', type=['jpg', 'jpeg', 'png'])

    # If a file was uploaded
    if file is not None:
        img = Image.open(file).convert('RGB')
        img = np.asarray(img)
        img = cv2.resize(img, (299, 299))
        img = img/255
        img = np.expand_dims(img, axis=0)

        # Make a prediction using the model
        p = model.predict(img)
        bc, at, g, s = p
        bc_i, at_i, g_i, s_i = [np.argmax(p) for p in [bc, at, g, s]]
        baseColor = COLOURS[bc_i]
        articleType = PRODUCT_TYPES[at_i]
        gender = GENDERS[g_i]
        season = SEASONS[s_i]

        st.image(file, caption='Uploaded Image', use_column_width=True)

        # Display the prediction to the user
        st.write(f'The uploaded image is classifies as:')
        st.write(f'Colour: {baseColor}')
        st.write(f'Product Type: {articleType}')
        st.write(f'Gender: {gender}')
        st.write(f'Season: {season}')

# Run the app
if __name__ == '__main__':
    app()
