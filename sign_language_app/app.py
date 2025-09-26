import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import os

# ----------------------
# 1. Constants & Model Definition
# ----------------------
IMG_SIZE = 224
NUM_CLASSES = 32

CLASS_LABELS = {
    0: ('ain', 'ع'), 1: ('al', 'ال'), 2: ('aleff', 'ا'), 3: ('bb', 'ب'),
    4: ('dal', 'د'), 5: ('dha', 'ظ'), 6: ('dhad', 'ض'), 7: ('fa', 'ف'),
    8: ('gaaf', 'ق'), 9: ('ghain', 'غ'), 10: ('ha', 'ه'), 11: ('haa', 'ها'),
    12: ('jeem', 'ج'), 13: ('kaaf', 'ك'), 14: ('khaa', 'خ'), 15: ('la', 'لا'),
    16: ('laam', 'ل'), 17: ('meem', 'م'), 18: ('nun', 'ن'), 19: ('ra', 'ر'),
    20: ('saad', 'ص'), 21: ('seen', 'س'), 22: ('sheen', 'ش'), 23: ('ta', 'ط'),
    24: ('taa', 'ت'), 25: ('thaa', 'ث'), 26: ('thal', 'ذ'), 27: ('toot', 'ة'),
    28: ('waw', 'و'), 29: ('ya', 'ي'), 30: ('yaa', 'يا'), 31: ('zay', 'ز')
}

MODEL_PATH = 'model/final_efficientnetb0.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_pytorch_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: The model file '{MODEL_PATH}' was not found.")
        return None
    try:
        model = models.efficientnet_b0(weights=None)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES)
        )
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        st.success(f"PyTorch model loaded on {DEVICE}.")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_pytorch_model()

# ----------------------
# 2. Hand Detection & Prediction (OpenCV-based)
# ----------------------
def detect_hand_and_crop(image_bgr):
    """Detect hand using color/contour segmentation and return cropped hand."""
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Skin color range for segmentation (adjust if needed)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Morphological operations to remove noise
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    # Largest contour assumed to be the hand
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    padding = 20
    x_min = max(0, x - padding)
    y_min = max(0, y - padding)
    x_max = min(image_bgr.shape[1], x + w + padding)
    y_max = min(image_bgr.shape[0], y + h + padding)
    
    cropped = image_bgr[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return None
    return cropped

def preprocess_and_predict(image_array, model):
    if model is None:
        return "Model not loaded.", None
    
    cropped_hand = detect_hand_and_crop(image_array)
    if cropped_hand is None:
        return "Hand not detected", None
    
    # Prepare input for model
    input_tensor = TRANSFORM(cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2RGB))
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted_index = torch.argmax(probabilities, 1).item()
    predicted_label_ar = CLASS_LABELS.get(predicted_index, ('Unknown', 'غير معروف'))[1]
    
    return predicted_label_ar, cropped_hand

# ----------------------
# 3. Streamlit App Interface
# ----------------------
st.set_page_config(page_title="Arabic Sign Language Alphabet Recognition (ArASL) App", layout="wide")
st.title("👋 Arabic Sign Language Alphabet Recognition (ArASL) App")
st.markdown("Upload an image or take a photo to predict the corresponding Arabic letter.")

uploaded_file = st.file_uploader("Upload a hand sign image:", type=["jpg","jpeg","png"])
camera_input = st.camera_input("Take a photo (Webcam):")

image_to_process = camera_input if camera_input else uploaded_file
source_label = "Camera Image" if camera_input else "Uploaded Image"

if image_to_process:
    image = Image.open(image_to_process).convert('RGB')
    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    predicted_char, cropped_hand = preprocess_and_predict(image_array, model)
    
    col1, col2 = st.columns([1,1])
    
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        if cropped_hand is not None:
            st.image(cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2RGB), caption="Detected Hand Area", use_column_width=True)
        else:
            st.warning("Hand could not be detected. Please try again.")
    
    st.markdown("---")
    st.markdown("## **Predicted Letter:**")
    st.success(f"## {predicted_char}")













