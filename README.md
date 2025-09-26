# GTC-Hand-Gesture-Recognition
# Arabic Sign Language Hand Gesture Recognition

## Problem Statement
Gesture-based interfaces are becoming increasingly important in applications such as human-computer interaction, gaming, sign language interpretation, and touchless control systems. This project aims to build a system that can accurately recognize Arabic hand gestures from images or video streams.

## Project Idea & Scope
Develop a hand gesture recognition model that can classify static images or live video frames into predefined Arabic gesture categories. The system should be lightweight, accurate, and adaptable for real-time use cases.

## Dataset
We are using a publicly available **Arabic Sign Language dataset**.  
The dataset contains labeled images of hand gestures corresponding to Arabic letters or words.

### Dataset Source
- [[Arabic Sign Language Dataset](https://example-link-to-dataset) *(replace with actual dataset link)*](https://www.kaggle.com/datasets/muhammadalbrham/rgb-arabic-alphabets-sign-language-dataset)

### Dataset Preparation
- Resize images to a consistent shape (e.g., 224x224 pixels).
- Normalize images to standardize pixel values.
- Ensure consistent labeling of gestures.

## Features & Data Augmentation
- Apply rotations, horizontal flips, and brightness adjustments to improve model robustness.
- Explore gesture distribution to ensure balanced classes.

## Model Training & Validation
- Train a CNN or use transfer learning (e.g., VGG16, ResNet, MobileNet) for gesture classification.
- Evaluate performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score

## Deployment
- Deploy the model in a lightweight frontend using Streamlit or JavaScript.
- Allow users to upload images or use a webcam to test real-time gesture recognition.

## Requirements
- Python 3.10+
- PyTorch or tensorflow
- torchvision
- OpenCV
- Streamlit (for web interface)
- numpy, pandas, matplotlib, etc.

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
## Links

- [Watch the project demo video]([https://drive.google.com/your-video-link](https://drive.google.com/file/d/12pJy_jfXQhRQtlCpAEuFRKGNpA4Q_Ilw/view))  
- [Try the deployed app]([https://your-app-link.com](https://4lnfwgwp6penpz4mbqz9qk.streamlit.app/))
