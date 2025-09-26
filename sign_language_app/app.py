import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
import os
import io

# ----------------------
# 1. Constants & Model Definition
# ----------------------

# Image size used during model training
IMG_SIZE = 224
NUM_CLASSES = 32

# Class names mapping
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

# --- Model File Name  ---
MODEL_PATH = 'C:/Users/hp/Downloads/python/python/sign_language_app/model/final_efficientnetb0.pth' # Make sure this is your file name

# Define the PyTorch Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformations for PyTorch model input
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # Use ImageNet normalization as you used IMAGENET1K_V1 weights
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@st.cache_resource
def load_pytorch_model():
    """
    Loads the PyTorch model architecture and weights, matching the custom structure
    
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: The model file '{MODEL_PATH}' was not found. Please ensure the file is in the same folder.")
        return None
    
    try:
        # 1. Define the base model architecture (EfficientNetB0)
        model = models.efficientnet_b0(weights=None) 
        
        # 2. Freeze the backbone layers (matching the training script)
        for param in model.parameters():
            param.requires_grad = False
        
        # 3. Get input feature count for the classifier
        # EfficientNetB0.classifier[1] is a Linear layer, from which we get the number of inputs
        num_ftrs = model.classifier[1].in_features
        
        # 4. CRITICAL: Redefine the classifier block to match your custom structure
        custom_classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),       # Index 0: Custom Linear Layer
            nn.ReLU(),                      # Index 1: ReLU
            nn.Dropout(0.4),                # Index 2: Dropout
            nn.Linear(256, NUM_CLASSES)     # Index 3: Final Linear Layer (32 classes)
        )
        
        model.classifier = custom_classifier

        # 5. Load the custom trained weights from the .pth file
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        
        # 6. Set the model to evaluation mode
        model.to(DEVICE)
        model.eval()
        
        st.success(f"PyTorch model (EfficientNetB0) loaded successfully on {DEVICE}.")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.error("Please check the file name and the custom classifier structure (256/32).")
        return None

# Load the model
model = load_pytorch_model()

# Initialize MediaPipe Hands (Same as before)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------
# 2. Hand Detection and Prediction Logic (PyTorch version)
# ----------------------


def preprocess_and_predict(image_array, model):
    """
    Detects hand, preprocesses ROI, and makes a prediction using PyTorch.
    """
    if model is None:
        return "Model not loaded.", None

    # Convert BGR image to RGB
    img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Process the image to find hand landmarks
    results = hands.process(img_rgb)
    
    predicted_label_ar = "Hand not detected"
    cropped_img = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = image_array.shape
            
            # Calculate bounding box coordinates
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Add padding to the bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Crop the image to the hand ROI
            cropped_img_bgr = image_array[y_min:y_max, x_min:x_max]

            if cropped_img_bgr.size == 0:
                predicted_label_ar = "Detection area is empty."
                break

            # Convert to RGB array for PyTorch Transform
            cropped_img_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)

            # Apply PyTorch Transformations (Resize, ToTensor, Normalize)
            input_tensor = TRANSFORM(cropped_img_rgb)
            
            # Add batch dimension and move to correct device
            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

            # Make Prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class_index = torch.argmax(probabilities, 1).item()
            
            # Get the Arabic label
            predicted_label_ar = CLASS_LABELS.get(predicted_class_index, ('Unknown', 'غير معروف'))[1]
            break # Process only the first detected hand

    return predicted_label_ar, cropped_img_bgr

# ----------------------
# 3. Streamlit App Interface
# ----------------------
# The part related to st.set_page_config, st.title, and the rest of the Streamlit interface remains unchanged.
st.set_page_config(page_title="Arabic Sign Language Alphabet Recognition (ArASL) App", layout="wide")

st.title("👋 Arabic Sign Language Alphabet Recognition (ArASL) App")
st.markdown("Please upload an image of your hand sign or use the camera to predict the corresponding Arabic letter.")

# File Uploader or Camera Input
uploaded_file = st.file_uploader("Upload a hand sign image:", type=["jpg", "jpeg", "png"])
st.markdown("---")
camera_input = st.camera_input("Take a photo (Webcam):")

# Processing logic
image_to_process = None
source_label = ""

if camera_input is not None:
    # Use camera input
    image_to_process = camera_input
    source_label = "Camera Image"
elif uploaded_file is not None:
    # Use uploaded file
    image_to_process = uploaded_file
    source_label = "Uploaded Image"

if image_to_process is not None:
    # Convert Streamlit UploadedFile to PIL Image and then to OpenCV array
    image = Image.open(image_to_process).convert('RGB')
    image_array = np.array(image)
    # Convert RGB to BGR for OpenCV processing
    image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    st.subheader(f"Result ({source_label}):")

    # Run detection and prediction
    predicted_char, cropped_hand = preprocess_and_predict(image_array_bgr, model)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        if cropped_hand is not None and cropped_hand.size > 0:
            # Convert BGR cropped image to RGB for Streamlit display
            cropped_hand_rgb = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2RGB)
            st.image(cropped_hand_rgb, caption="Detected Hand Area", use_column_width=True)
        else:
            st.warning("The system could not crop a clear image of a hand. Please try again.")

    st.markdown("---")
    st.markdown(f"## **Predicted Letter:**")
    st.success(f"## {predicted_char}")

    if predicted_char in ["Model not loaded.", "Hand not detected", "Detection area is empty."]:
        st.info("Please make sure your hand is clear and well-lit within the frame.")











