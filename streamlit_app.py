import streamlit as st
import cv2
import numpy as np
from PIL import Image

def image_to_sketch(image, blur_value):
    # Convert to grayscale 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert the grayscale image
    inverted_image = 255 - gray_image
    # Blur the inverted image
    blurred = cv2.GaussianBlur(inverted_image, (blur_value, blur_value), 0)
    # Invert the blurred image
    inverted_blurred = 255 - blurred
    # Create the pencil sketch
    sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
    return sketch

# Streamlit app UI configuration
st.set_page_config(
    page_title="Image to Sketch AI",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("\U0001F3A8 Image to Sketch")
st.write("### Convert your photos into professional-grade pencil sketches!")

# Sidebar for customization options
st.sidebar.header("Customization Options")
blur_value = st.sidebar.slider("Blur Intensity", 1, 51, 21, step=2)

# Upload image section
uploaded_file = st.file_uploader("Upload your image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert the image to a sketch
    sketch = image_to_sketch(image_cv, blur_value)

    # Display the sketch
    st.write("### Pencil Sketch")
    st.image(sketch, caption="Pencil Sketch", use_container_width=True, channels="GRAY")

    # Option to download the sketch
    result = Image.fromarray(sketch)
    st.download_button(
        label="Download Sketch",
        data=result.tobytes(),
        file_name="sketch.png",
        mime="image/png",
    )
else:
    st.write("Upload an image to get started!")

# Footer
st.markdown("""
---
**Designed by [theaimart](https://theaimart.com)**
""")
