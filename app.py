# ---------------------- MODEL LOADING ----------------------
import os
import pickle
import streamlit as st
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50


@st.cache_resource
def load_all_models():
    try:
        # Absolute path of the current folder
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # File paths
        yolo_path = os.path.join(BASE_DIR, "yolov8m.pt")
        tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")
        caption_model_path = os.path.join(BASE_DIR, "caption_model.keras")

        # Check if files exist
        if not os.path.exists(tokenizer_path):
            st.error(f"Tokenizer not found:\n{tokenizer_path}")
            st.stop()

        if not os.path.exists(caption_model_path):
            st.error(f"Caption model not found:\n{caption_model_path}")
            st.stop()

        # Load YOLO
        yolo_model = YOLO(yolo_path)

        # Load ResNet50
        resnet_model = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )

        # Load tokenizer
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        # Load caption model
        lstm_model = load_model(caption_model_path)

        return yolo_model, resnet_model, tokenizer, lstm_model

    except Exception as e:
        st.error(f"Model loading failed:\n\n{e}")
        st.stop()


# Load all models
yolo_model, resnet_model, tokenizer, lstm_model = load_all_models()
MAX_LENGTH = 31

   
