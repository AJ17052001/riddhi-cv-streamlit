# ---------------------- MODEL LOADING ----------------------
import os
import pickle
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request

# Check for OpenCV system dependencies on Streamlit Cloud
try:
    import cv2
except ImportError:
    st.error("OpenCV dependencies are missing. Please ensure your `packages.txt` contains:\n\n`libgl1-mesa-glx` \n`libglib2.0-0`")

# ==========================================
# 1. LIVE REPAIR FOR CORRUPTED/LFS MODELS
# ==========================================
def secure_model_file(filename, expected_min_size_mb=1):
    """Checks if a file exists and deletes it if it is a text/LFS pointer."""
    if os.path.exists(filename):
        file_size_bytes = os.path.getsize(filename)
        # Convert MB to bytes
        min_bytes = expected_min_size_mb * 1024 * 1024
        
        if file_size_bytes < min_bytes:
            st.warning(f"Detected corrupted placeholder file for {filename} (Size: {file_size_bytes} bytes). Repairing...")
            os.remove(filename)

# Scan and clear out any fake text files holding up your models
secure_model_file('yolov8m.pt', expected_min_size_mb=10)
secure_model_file('caption_model.keras', expected_min_size_mb=5)
secure_model_file('tokenizer.pkl', expected_min_size_mb=0.01)

# ==========================================
# 2. MODEL LOADING INFRASTRUCTURE
# ==========================================
@st.cache_resource
def load_models():
    # Load YOLOv8 (Will auto-download safely now that corrupt files are gone)
    try:
        from ultralytics import YOLO
        yolo_model = YOLO('yolov8m.pt')
    except Exception as e:
        st.error(f"Failed to initialize YOLOv8: {e}")
        yolo_model = None

    # Load Tokenizer
    try:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading tokenizer.pkl. Your file might be a Git LFS text pointer. Details: {e}")
        tokenizer = None

    # Load Keras Captioning Model
    try:
        import tensorflow as tf
        caption_model = tf.keras.models.load_model('caption_model.keras')
    except Exception as e:
        st.error(f"Error loading caption_model.keras. Your file might be a Git LFS text pointer. Details: {e}")
        caption_model = None

    return yolo_model, tokenizer, caption_model

# Initialize models
yolo, tokenizer, caption_model = load_models()

# ==========================================
# 3. STREAMLIT INTERFACE UI
# ==========================================
st.title("📸 AI Image Caption Generator")
st.write("Upload an image to detect objects using YOLOv8 and generate a detailed description.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Check if models loaded successfully before running predictions
    if yolo is None or tokenizer is None or caption_model is None:
        st.error("Cannot process image because one or more models failed to load correctly. Please check the error messages above.")
    else:
        with st.spinner("Processing image and generating caption..."):
            # Convert PIL image to NumPy array for YOLO processing
            img_array = np.array(image)
            
            # 1. Run Object Detection
            yolo_results = yolo(img_array)
            
            # Extract detected object names
            detected_objects = []
            for result in yolo_results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    label = yolo.names[class_id]
                    if label not in detected_objects:
                        detected_objects.append(label)
            
            # 2. Display Detection Results
            if detected_objects:
                st.subheader("Objects Detected:")
                st.write(", ".join(detected_objects))
            else:
                st.write("No distinct objects detected by YOLO.")
            
            # 3. Generate Caption (Placeholder logic - adapt to your model's predict function)
            st.subheader("Generated Caption:")
            try:
                # Add your exact feature extraction & text sequence generation logic here
                # Example snippet: 
                # features = feature_extractor.predict(img_array)
                # generated_text = predict_caption(caption_model, tokenizer, features)
                
                st.info("✨ [Your model's caption will output here. Link your feature processing step here!]")
            except Exception as eval_err:
                st.error(f"Error during caption generation evaluation: {eval_err}")
