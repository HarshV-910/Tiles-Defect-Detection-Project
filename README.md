# Tiles-Defect-Detection-Project
---

# Surface Detection in Ceramic Tiles
This project focuses on identifying surface defects in ceramic tiles using a TensorFlow Lite model deployed via a Streamlit web application. The goal is to automate defect detection and ensure quality control in ceramic tile manufacturing.

## Defect Classes
The model is trained to classify ceramic tile images into the following categories:
- Crack  
- Glue Strip  
- Good (No Defect)  
- Gray Stroke  
- Oil  
- Rough 

## Model Information
Model Type: TensorFlow Lite  
Model Created Using: Google Teachable Machine  

## Deployment
Deployment Platform: Streamlit  

App Pages:
- Home: Overview and objective of the project.  
- About Data: Information regarding the dataset and its source.  
- Tile Defect Detection:  
  - Upload image (JPG or JPEG)  
  - (Optional) Preview the image  
  - Predict defect class with confidence score  

## Dataset
Source: MVTec Anomaly Detection Dataset  
Dataset Used: Tile dataset from MVTec AD  

## Version Control
Tools Used: Git, GitHub  

## How to Run

1. Clone the repository:
git clone https://github.com/HarshV-910/Tiles-Defect-Detection-Project.git
cd Tiles-Defect-Detection-Project

2. Install dependencies:
pip install -r requirements.txt

3. Run the Streamlit app:
streamlit run app.py

## Acknowledgements
Google Teachable Machine  
MVTec AD Dataset  
Streamlit  
TensorFlow Lite
