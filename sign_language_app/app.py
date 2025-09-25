# app.py
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp

# -----------------------
# Configuration / Model
# -----------------------
st.set_page_config(page_title="Arabic Sign Language Recognition", layout="wide")

NUM_CLASSES = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture
model = models.efficientnet_b0(pretrained=False)
model.classifier = nn.Sequential(
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)

# Load weights (adjust path if needed)
state_dict = torch.load(
    "C:/Users/hp/Downloads/python/python/sign_language_app/model/final_efficientnetb0.pth",
    map_location=DEVICE
)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# Class names (same as your dataset order)
class_names = [
    'ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad',
    'fa', 'gaaf', 'ghain', 'ha', 'haa', 'jeem', 'kaaf', 'khaa',
    'la', 'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 'ta',
    'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay'
]

# Choose normalization that matches your training.
# You can toggle 'use_imagenet_norm' in sidebar if needed.
def make_transform(use_imagenet_norm=True):
    t = [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
    if use_imagenet_norm:
        t.append(transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225]))
    return transforms.Compose(t)

# -----------------------
# Sidebar controls
# -----------------------
st.title("🤟 Arabic Sign Language Recognition — MediaPipe + EfficientNet")
st.markdown("Live hand detection (skeleton) + model prediction. Use Debug to inspect crop/landmarks/top-3.")

use_imagenet_norm = st.sidebar.checkbox("Use ImageNet normalization (0.485/0.229)", value=True)
mirror_input = st.sidebar.checkbox("Mirror input (horizontal flip)", value=False)
debug_mode = st.sidebar.checkbox("Enable Debug Mode (show crop, skeleton, top-3)", value=True)
min_detection_conf = st.sidebar.slider("MediaPipe detection confidence", 0.3, 0.95, 0.6)

transform = make_transform(use_imagenet_norm)

# -----------------------
# Hand detector + transformer
# -----------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # MediaPipe Hands (keep instance to reuse)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=0.5,
        )
        # last results stored here (main thread will read them)
        self.last_pred = None
        self.last_top3 = None
        self.last_crop_bgr = None
        self.last_landmarks_img_bgr = None

    def transform(self, frame):
        # Convert incoming frame to BGR numpy array
        img_bgr = frame.to_ndarray(format="bgr24")
        if mirror_input:
            img_bgr = cv2.flip(img_bgr, 1)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        # Reset defaults
        self.last_pred = None
        self.last_top3 = None
        self.last_crop_bgr = None
        self.last_landmarks_img_bgr = None

        if not results.multi_hand_landmarks:
            # no hand detected -> return original
            return img_bgr

        # We have at least one hand; use the first one
        hand_lms = results.multi_hand_landmarks[0]
        h, w, _ = img_bgr.shape

        # Convert normalized landmark coords to pixel coords
        xs = [int(lm.x * w) for lm in hand_lms.landmark]
        ys = [int(lm.y * h) for lm in hand_lms.landmark]

        # bounding box around landmarks with padding
        padding = int(min(w, h) * 0.12)  # 12% padding
        x_min = max(min(xs) - padding, 0)
        x_max = min(max(xs) + padding, w)
        y_min = max(min(ys) - padding, 0)
        y_max = min(max(ys) + padding, h)

        # crop the hand region
        crop = img_bgr[y_min:y_max, x_min:x_max].copy()
        if crop.size == 0:
            return img_bgr

        # draw landmarks / skeleton on a copy for debugging overlay
        lmk_img = img_bgr.copy()
        mp_drawing.draw_landmarks(lmk_img, hand_lms, mp_hands.HAND_CONNECTIONS)

        # MODEL INFERENCE
        try:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            img_t = transform(pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(img_t)
                probs = torch.softmax(outputs, dim=1)[0]
                topk = torch.topk(probs, 3)

                top3 = [(class_names[int(topk.indices[i])], float(topk.values[i])) for i in range(3)]
                pred = top3[0][0]

                # Save debug info to transformer instance (readable from main thread)
                self.last_pred = pred
                self.last_top3 = top3
                # store BGR images for UI
                self.last_crop_bgr = crop
                self.last_landmarks_img_bgr = lmk_img

                # annotate the live frame so user sees box + label
                cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(img_bgr, f"{pred}", (x_min, max(y_min - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            # If inference fails, leave last_pred None (we'll show debug info)
            # Avoid calling Streamlit functions here.
            print("Inference error:", e)

        return img_bgr

# -----------------------
# Run WebRTC streamer
# -----------------------
webrtc_ctx = webrtc_streamer(
    key="sign-lang-mediapipe",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,  # allow async transforms (recommended)
)

# -----------------------
# Main UI / Debug panels (runs in main Streamlit thread)
# -----------------------
col_vid, col_debug = st.columns([2, 1])

with col_debug:
    pred_box = st.empty()
    top3_box = st.empty()
    crop_box = st.empty()
    lmk_box = st.empty()
    st.markdown("---")
    st.markdown("Tips:")
    st.write("- Place your hand clearly in front of the camera.")
    st.write("- Keep background uncluttered and stable lighting.")
    st.write("- Toggle Mirror if your dataset used mirrored images.")

# Read last predictions from transformer (if active)
if webrtc_ctx and webrtc_ctx.video_transformer:
    vt = webrtc_ctx.video_transformer

    # Show prediction (if present)
    if vt.last_pred:
        pred_box.subheader(f"📌 Prediction: {vt.last_pred}")
    else:
        pred_box.subheader("📌 Prediction: — (no confident detection)")

    if debug_mode:
        # show top-3 probabilities
        if vt.last_top3:
            top3_box.markdown("### 🔎 Top-3 Predictions")
            for cls, p in vt.last_top3:
                top3_box.write(f"{cls}: {p:.3f}")
        else:
            top3_box.markdown("### 🔎 Top-3 Predictions")
            top3_box.write("No predictions yet")

        # show crop
        if vt.last_crop_bgr is not None:
            crop_box.image(cv2.cvtColor(vt.last_crop_bgr, cv2.COLOR_BGR2RGB), caption="Hand Crop")
        else:
            crop_box.write("Hand crop not available")

        # show landmarks overlay on full frame
        if vt.last_landmarks_img_bgr is not None:
            lmk_box.image(cv2.cvtColor(vt.last_landmarks_img_bgr, cv2.COLOR_BGR2RGB),
                          caption="Landmarks / Skeleton Overlay")
        else:
            lmk_box.write("No landmarks to show")
else:
    pred_box = st.empty()
    pred_box.write("Start the camera to get predictions.")










