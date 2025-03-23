import streamlit as st # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
import torchvision.transforms as transforms # type: ignore
import cv2 # type: ignore
from PIL import Image # type: ignore
from torchvision.models import resnet50 # type: ignore

# Load pre-trained model
model = resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("Fashion Image Feature Extractor")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image for model
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        feature = model(image)

    feature_vector = feature.squeeze().numpy()

    st.subheader("Extracted Features:")
    st.write(feature_vector)  # Display feature vector

    # Save extracted features
    np.save("uploaded_fashion_features.npy", feature_vector)
    st.success("Feature extraction complete! Features saved.")
