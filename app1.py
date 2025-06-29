import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from PIL import Image, ImageOps  # Pillow library # type: ignore 

# Add custom CSS for background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1695728213930-93ced4114eb0?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjB8fGRhcmslMjB0aWxlcyUyMGJhY2tncm91bmR8ZW58MHx8MHx8fDA%3D");
        # background-image: url("https://images.unsplash.com/photo-1478760329108-5c3ed9d495a0?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        # opacity: 0.8;
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: -1;
    }
    # header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# sidebar_bg = ""  # Web URL
sidebar_bg = "https://images.unsplash.com/photo-1642944267662-dacacac1132c?q=80&w=642&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"  # Web URL

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] > div:first-child {{
        background-image: url("{sidebar_bg}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color:white; 
        text-shadow: 0.3px 0.3px 0.3px white,0.6px 0.6px 1px black;
        border-right : 0.2px solid white; 

        
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Load labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Function to predict with TFLite model
def predict_image_tflite(image_file):
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load image
    image = Image.open(image_file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    
    # Normalize image to [-1, 1]
    normalized_image_array = (image_array / 127.5) - 1.0

    # Add batch dimension
    input_data = np.expand_dims(normalized_image_array, axis=0)

    # Set tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.squeeze(output_data)

    # Get highest confidence index
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[index]

    return class_name, confidence_score

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Data", "Leather Anomaly Detection"])

# Main Page
if app_mode == "Home":
    st.header("")
    st.markdown("""
# Leather Anomaly Detection in Manufacturing""")
    st.image("https://plus.unsplash.com/premium_photo-1669930762980-563aef80a9a6?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",caption="Tiles Anomaly Detection")
    st.markdown("""


### Project Overview:
In leather manufacturing industries, ensuring the quality of leather products is critical. Defects like color inconsistencies, physical damages, or improper processing can affect product quality and market value. This project automates the **visual inspection process** by using **computer vision and deep learning** techniques to detect different types of surface anomalies in leather products.

---

### Objective:
To build an automated system that can classify leather images into various categories based on surface conditions, improving inspection efficiency and reducing manual errors.

---


### Importance:
- Reduces dependency on manual inspection.
- Minimizes faulty deliveries.
- Ensures consistent product quality.
- Helps in early detection of process errors.
- Speeds up the quality control workflow.

        """)
    
elif app_mode == "About Data":
    st.header("About Data")
    st.markdown("""

## Dataset Description: Leather Anomaly Detection

### Context:
In leather manufacturing industries, visual inspection is a critical quality control step. Manual inspection, however, can be **inconsistent, time-consuming, and prone to human error**. To address this, we curated a dataset of high-resolution leather surface images, categorized into multiple defect classes typically encountered during the manufacturing process.

---

### Dataset Structure:
The dataset is organized into **6 distinct classes** representing different types of surface conditions:

| Class   | Description                                     | Example Issues                        |
|:--------|:------------------------------------------------|:--------------------------------------|
| **Color**  | Irregular color patches, discoloration         | Faded spots, uneven tanning            |
| **Cut**    | Visible cuts, tears, or slits on the surface    | Knife cuts, accidental slashes        |
| **Poke**   | Small holes or puncture defects                | Needle holes, sharp object pokes      |
| **Good**   | Defect-free, high-quality leather surfaces     | Smooth, consistent texture and color  |
| **Fold**   | Wrinkles or folds in the leather               | Creases from handling or storage      |
| **Glue**   | Visible glue stains or adhesive marks          | Processing adhesive residues          |

---

### Image Details:
- **Image Dimensions**: Variable (e.g., 128x128, 224x224) depending on preprocessing.
- **Image Type**: RGB images captured using high-resolution cameras in industrial setups.
- **Format**: JPEG / PNG.


---

### Dataset Purpose:
- Train, validate, and test deep learning models for image classification.
- Develop automated anomaly detection systems for leather manufacturing.
- Benchmark different CNN architectures and preprocessing techniques.

---

### Possible Future Extensions:
- **Add more defect classes**: (e.g., stain, scratch, burn mark)
- **Pixel-wise defect segmentation** for localization.
- **Anomaly detection using autoencoders** for unseen defect types.

        """)

elif app_mode == "Leather Anomaly Detection":
    st.header("Leather Anomaly Detection")
    test_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_container_width=True)

        if st.button("Predict"):

            class_name, confidence_score = predict_image_tflite(test_image)

            st.subheader(f"**Prediction:** {class_name}")
            st.info(f"**Confidence Score:** {confidence_score:.2f}")

