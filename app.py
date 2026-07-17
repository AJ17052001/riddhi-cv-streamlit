import os
import sys

# ====================================================================
# 0. STRICT ENVIRONMENT ISOLATION (MUST RUN FIRST)
# ====================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_MANAGED_MEMORY'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import pickle
import numpy as np
import streamlit as st
from PIL import Image
import urllib.request

# Check for OpenCV dependencies
try:
    import cv2
except ImportError:
    st.error("OpenCV system files are missing. Ensure your `packages.txt` file contains:\n\n`libgl1-mesa-glx` \n`libglib2.0-0`")

# ====================================================================
# 1. CONFIGURATION
# ====================================================================
CAPTION_MODEL_URL = "https://your-hosting-site.com/caption_model.keras"
TOKENIZER_URL = "https://your-hosting-site.com/tokenizer.pkl"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"

# ====================================================================
# 2. THREAD-SAFE REPAIR & DOWNLOAD ENGINE
# ====================================================================
def secure_and_download(filename, download_url, expected_min_size_mb=1.0):
    """Safely handles Git LFS pointer text files and triggers authentic binary downloads."""
    if os.path.exists(filename):
        try:
            file_size_bytes = os.path.getsize(filename)
            min_bytes = expected_min_size_mb * 1024 * 1024
            
            if file_size_bytes < min_bytes:
                st.warning(f"Removing placeholder file: '{filename}'...")
                # Thread-safe deletion check
                if os.path.exists(filename):
                    os.remove(filename)
        except FileNotFoundError:
            # Another concurrent execution thread already deleted it
            pass
            
    if not os.path.exists(filename):
        if "your-hosting-site" in download_url and filename != 'yolov8m.pt':
            create_mock_fallback_file(filename)
            return

        with st.spinner(f"📥 Downloading authentic {filename}..."):
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                
                urllib.request.urlretrieve(download_url, filename)
                st.success(f"Successfully deployed: {filename}")
            except Exception as e:
                st.error(f"Failed to download {filename}: {e}")
                create_mock_fallback_file(filename)

def create_mock_fallback_file(filename):
    """Creates fallback mocks purely using basic pickle or isolated lazy imports."""
    st.info(f"⚙️ Generating a local fallback mock for '{filename}'...")
    
    if filename == 'tokenizer.pkl':
        mock_tokenizer = {"<start>": 1, "<end>": 2, "a": 3, "photo": 4, "of": 5}
        with open(filename, 'wb') as f:
            pickle.dump(mock_tokenizer, f)
            
    elif filename == 'caption_model.keras':
        # LAZY IMPORT ONLY WHEN NEEDED: Prevents TensorFlow from starting prematurely
        import tensorflow as tf
        inputs = tf.keras.Input(shape=(2048,))
        outputs = tf.keras.layers.Dense(10, activation='softmax')(inputs)
        mock_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        mock_model.save(filename)

# Run verification metrics safely before cache assignments
secure_and_download('yolov8m.pt', YOLO_URL, expected_min_size_mb=10.0)
secure_and_download('caption_model.keras', CAPTION_MODEL_URL, expected_min_size_mb=2.0)
secure_and_download('tokenizer.pkl', TOKENIZER_URL, expected_min_size_mb=0.005)

# ====================================================================
# 3. RESOURCE INITIALIZATION (CACHED INTERFACES)
# ====================================================================
@st.cache_resource
def load_yolo():
    try:
        import torch
        torch.backends.cudnn.enabled = False
        from ultralytics import YOLO
        return YOLO('yolov8m.pt')
    except Exception as e:
        st.error(f"YOLOv8 structural error: {e}")
        return None

@st.cache_resource
def load_tokenizer():
    try:
        with open('tokenizer.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Tokenizer parsing error: {e}")
        return None

@st.cache_resource
def load_caption_model():
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.keras.backend.clear_session()
        return tf.keras.models.load_model('caption_model.keras')
    except Exception as e:
        st.error(f"Keras weight compilation error: {e}")
        return None

# Load each cleanly separated to isolate memory allocations
yolo = load_yolo()
tokenizer = load_tokenizer()
caption_model = load_caption_model()

# ====================================================================
# 4. STREAMLIT FRONTEND USER INTERFACE
# ====================================================================
st.title("📸 AI Image Caption Generator")
st.write("Upload any image file to segment objects via YOLOv8 and generate predictive captions.")

uploaded_file = st.file_uploader("Upload Image Asset...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Target Image Matrix", width='stretch')
    
    if yolo is None or tokenizer is None or caption_model is None:
        st.error("Cannot evaluate. One or more architectural components failed configuration checks.")
    else:
        with st.spinner("Analyzing frames and computing tags..."):
            img_array = np.array(image)
            
            # Phase 1: Object Extraction Inference
            yolo_results = yolo(img_array)
            detected_objects = []
            
            for result in yolo_results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    label = yolo.names[class_id]
                    if label not in detected_objects:
                        detected_objects.append(label)
            
            # Phase 2: Render Object Context UI
            st.subheader("🎯 Objects Extracted:")
            if detected_objects:
                st.success(", ".join(detected_objects))
            else:
                st.info("No distinct features cataloged by YOLO.")
            
            # Phase 3: Text Sequence Generation Interface
            st.subheader("✨ Generated Caption Output:")
            try:
                if "your-hosting-site" in CAPTION_MODEL_URL:
                    st.warning("Running mockup pipeline. Replace placeholders at the top of code to hook up your unique models!")
                    st.info(f"**Generated Text prediction:** A photo showing context containing: {', '.join(detected_objects) if detected_objects else 'an item'}.")
                else:
                    st.info("System fully connected! Plug your feature embedding function down here to stream structural text.")
            except Exception as generation_error:
                st.error(f"Error mapping features to text sequence: {generation_error}")
