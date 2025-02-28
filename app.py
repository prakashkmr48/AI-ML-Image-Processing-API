import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import requests
import io

# Load ImageNet Class Labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.split("\n")

# Load Pre-trained Model (ResNet for image classification)
model = models.resnet18(pretrained=True)
model.eval()

# Define Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("AI/ML Image Processing App")
st.write("Upload an image and get predictions using a pre-trained ResNet model.")



uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    image_tensor = transform(image).unsqueeze(0)

    # Model Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)

    st.write(f"**Prediction:** {labels[int(predicted.item())]}")
