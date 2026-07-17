# ---------------------- MODEL LOADING ----------------------
import os
import pickle
import numpy as np
import streamlit as st
from PIL import Image
import urllib.request

# Force TensorFlow to run light on memory and prevent CPU-based segmentation faults
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# Check for OpenCV system dependencies
try:
    import cv2
except ImportError:
    st.error("OpenCV system files are missing. Ensure your `packages.txt` file contains:\n\n`libgl1-mesa-glx` \n`libglib2.0-0`")

# ====================================================================
# 1. CONFIGURATION: PASTE YOUR DIRECT DOWNLOAD LINKS HERE (OPTIONAL)
# ====================================================================
CAPTION_MODEL_URL = "https://your-hosting-site.com/caption_model.keras"
TOKENIZER_URL = "https://your-hosting-site.com/tokenizer.pkl"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"

# ====================================================================
# 2. DYNAMIC REPAIR & DOWNLOAD ENGINE
# ====================================================================
def secure_and_download(filename, download_url, expected_min_size_mb=1.0):
    """Detects fake Git LFS text pointer files, removes them, and downloads the real binary."""
    if os.path.exists(filename):
        file_size_bytes = os.path.getsize(filename)
        min_bytes = expected_min_size_mb * 1024 * 1024
        
        if file_size_bytes < min_bytes:
            st.warning(f"Removing corrupted placeholder file: '{filename}' ({file_size_bytes} bytes)...")
            os.remove(filename)
            
    if not os.path.exists(filename):
        if "your-hosting-site" in download_url and filename != 'yolov8m.pt':
            create_mock_fallback_file(filename)
            return

        with st.spinner(f"📥 Downloading authentic {filename} from remote server..."):
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
    """Creates temporary working components if actual files aren't hosted yet."""
    import tensorflow as tf
    st.info(f"⚙️ Generating a local fallback mock for '{filename}' to keep the UI active...")
    
    if filename == 'tokenizer.pkl':
        mock_tokenizer = {"<start>": 1, "<end>": 2, "a": 3, "photo": 4, "of": 5}
        with open(filename, 'wb') as f:
            pickle.dump(mock_tokenizer, f)
            
    elif filename == 'caption_model.keras':
        inputs = tf.keras.Input(shape=(2048,))
        outputs = tf.keras.layers.Dense(10, activation='softmax')(inputs)
        mock_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        mock_model.save(filename)

# Run verification matrix before loading
secure_and_download('yolov8m.pt', YOLO_URL, expected_min_size_mb=10.0)
secure_and_download('caption_model.keras', CAPTION_MODEL_URL, expected_min_size_mb=2.0)
secure_and_download('tokenizer.pkl', TOKENIZER_URL, expected_min_size_mb=0.005)

# ====================================================================
# 3. RESOURCE INITIALIZATION
# ====================================================================
@st.cache_resource
def load_all_models():
    # Load Ultralytics Object Detector
    try:
        from ultralytics import YOLO
        yolo_model = YOLO('yolov8m.pt')
    except Exception as e:
        st.error(f"YOLOv8 structural error: {e}")
        yolo_model = None

    # Load Word Vocabulary Tokenizer
    try:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        st.error(f"Tokenizer parsing error: {e}")
        tokenizer = None

    # Load Keras Recurrent Caption Generator
    try:
        import tensorflow as tf
        # Clear background graph sessions to minimize memory footprint
        tf.keras.backend.clear_session()
        caption_model = tf.keras.models.load_model('caption_model.keras')
    except Exception as e:
        st.error(f"Keras weight compilation error: {e}")
        caption_model = None

    return yolo_model, tokenizer, caption_model

yolo, tokenizer, caption_model = load_all_models()

# ====================================================================
# 4. STREAMLIT FRONTEND USER INTERFACE
# ====================================================================
st.title("📸 AI Image Caption Generator")
st.write("Upload any image file to segment objects via YOLOv8 and generate predictive captions.")

uploaded_file = st.file_uploader("Upload Image Asset...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display media object
    image = Image.open(uploaded_file)
    
    # FIX: Replaced deprecated use_container_width=True with width='stretch'
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
                    st.info("System fully connected! Plug your feature embedding function directly down here to stream structural text.")
            except Exception as generation_error:
                st.error(f"Error mapping features to text sequence: {generation_error}")
