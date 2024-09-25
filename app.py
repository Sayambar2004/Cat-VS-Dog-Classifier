import streamlit as st
import numpy as np
from PIL import Image
import pickle

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 48px;
        color: white;
        background-color: #4CAF50;
        padding: 20px;
        border-radius: 10px;
    }
    .footer {
        font-size: 16px;
        color: #888888;
        text-align: center;
    }
    .developer-section {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 10px;
    }
    .model-info {
        background-color: black;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-bottom: 30px;
        font-size: 18px;
        line-height: 1.6;
        color: white;
    }
    .model-info h2 {
        text-align: center;
        font-size: 30px;
        margin-bottom: 20px;
    }
    .important {
        color: #e74c3c;
        font-weight: bold;
        font-size: 22px;
    }
    .key-info {
        color: #2980b9;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Page Title with Custom Styling
st.markdown('<div class="title">Cat vs Dog Classifier 🐱🐶</div>', unsafe_allow_html=True)

# Developer Info in Sidebar
st.sidebar.markdown("""
    ## Developer Info 👨‍💻
    **Name:** Sayambar Roy Chowdhury  
    **LinkedIn:** [Sayambar's LinkedIn](https://www.linkedin.com/in/sayambar-roy-chowdhury-731b0a282/)  
    **GitHub:** [Sayambar's GitHub](https://github.com/Sayambar2004)  
    **Developer's Note:** This is my first CNN Project. The DL Model still is not sufficient to judge any class other than Cat or Dog.
    """)

# Load the pre-trained VGG16 model from .pkl file
with open('vgg.pkl', 'rb') as f:
    model = pickle.load(f)

# Image uploader
st.markdown("### Upload an image of a **Cat** or a **Dog**:")
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

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
    predicted_class = 'Dog 🐕' if predictions[0][0] > 0.5 else 'Cat 🐈'

    # Display the prediction result
    st.markdown(f"## Prediction: **{predicted_class}**")

# Add a separator between the prediction and model information sections
st.markdown("---")  # Horizontal line for separation

# Improved Model Information Section
st.markdown("""
    <div class="model-info">
        <h2>Model Architecture and Performance Overview</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("""
- **Base Model:** VGG16 (pre-trained on ImageNet, excluding top layers)

### Transfer Learning Approach:
I utilized **Transfer Learning** by adopting the well-established **VGG16** architecture. Pre-trained on the extensive **ImageNet** dataset, the model has been fine-tuned specifically for the binary task of classifying **cats** and **dogs**. This approach drastically reduces training time while maintaining a high level of accuracy.
""")

st.write("#### Model Modifications:")
st.write("""
- **Layer Adjustments:** The 5th convolutional block of VGG16 was dropped to allow for task-specific fine-tuning.
- **Fully Connected Layers:** Added a **Flatten** layer followed by custom **Dense** layers with **ReLU** activation functions.
- **Binary Output:** A final **Sigmoid** layer was added to output probabilities for binary classification (Cat or Dog).
- **Loss Function:** Used **Binary Cross Entropy** to handle the binary classification problem, ensuring that the model learns to minimize log loss effectively.
""")

st.write("### Training Results:")
st.write("""
- **Training Accuracy:** 99.93%
- **Validation Accuracy:** 100%
- **Training Loss:** 0.0023 (measured using **Binary Cross Entropy**)
- **Validation Loss:** 0.000242
""")

st.write("#### Training Hyperparameters:")
st.write("""
- **Optimizer:** RMSprop with a low learning rate of 2e-5, ensuring smooth convergence without overfitting.
- **Batch Size:** 32
- **Training Epochs:** 10
""")

st.write("#### Image Preprocessing:")
st.write("""
Input images were resized to **150x150** pixels and normalized by scaling pixel values between **0 and 1** to ensure compatibility with the VGG16 architecture.
""")

st.write("### Model Performance:")
st.write("""
The model has achieved **near-perfect performance** in classifying cats and dogs. By fine-tuning a pre-trained model with an extremely low learning rate, the model excels at feature extraction, leading to almost flawless predictions. The utilization of Binary Cross Entropy as the loss function also ensures that the model learns effectively in this binary classification task.
""")

# Footer with a cool message
st.markdown('<div class="footer">This model is for experimental purposes only.</div>', unsafe_allow_html=True)
