import streamlit as st
import numpy as np
from PIL import Image
import pickle

# Load the pre-trained VGG16 model from .pkl file
with open('vgg.pkl', 'rb') as f:
    model = pickle.load(f)

# Set up the Streamlit app title
st.title('Cat vs Dog Classifier')

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    image = image.resize((150, 150))  # Resize to the VGG16 input size
    image = np.array(image)  # Convert image to NumPy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 224, 224, 3)
    image = image / 255.0  # Normalize the image if necessary

    # Make predictions using the loaded model
    predictions = model.predict(image)
    predicted_class = 'Dog' if predictions[0][0] > 0.5 else 'Cat'

    # Display the prediction result
    st.write(f"Prediction: {predicted_class}")
