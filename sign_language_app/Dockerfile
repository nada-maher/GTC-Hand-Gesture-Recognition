# Use an official Python image
FROM python:3.10-slim

# Install system dependencies (needed for opencv, mediapipe, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install mediapipe separately to avoid build conflicts
RUN pip install mediapipe==0.10.9

# Copy the rest of the code
COPY . .

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Command to run Streamlit on Spaces
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
