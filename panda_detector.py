import streamlit as st
from PIL import Image
from torchvision import models, transforms
import torch

# Load a pre-trained EfficientNet model
model = models.efficientnet_b0(pretrained=True)
model.eval()

# Class labels (from ImageNet)
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = None

@st.cache_resource
def load_labels():
    import requests
    response = requests.get(LABELS_URL)
    return response.json()

labels = load_labels()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match EfficientNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        label = labels[predicted.item()]
        return label

# Streamlit app
st.title("Panda Detector App (EfficientNet)")
st.write("Upload an image to check if it contains a panda!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    label = predict(image)
    if "panda" in label.lower():
        st.success(f"Yes! This is a panda! (Detected: {label})")
    else:
        st.warning(f"No panda detected. Detected: {label}")
