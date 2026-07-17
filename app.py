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
        # Project folder
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # File paths
        yolo_path = os.path.join(BASE_DIR, "yolov8m.pt")
        tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")
        model_path = os.path.join(BASE_DIR, "caption_model.keras")

        # Debug information
        st.write("Project folder:", BASE_DIR)
        st.write("Files found:", os.listdir(BASE_DIR))

        # Check files
        if not os.path.isfile(yolo_path):
            st.error(f"YOLO model not found:\n{yolo_path}")
            st.stop()

        if not os.path.isfile(tokenizer_path):
            st.error(f"Tokenizer not found:\n{tokenizer_path}")
            st.stop()

        if not os.path.isfile(model_path):
            st.error(f"Caption model not found:\n{model_path}")
            st.stop()

        # Load models
        yolo_model = YOLO(yolo_path)

        resnet_model = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )

        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        lstm_model = load_model(model_path)

        st.success("All models loaded successfully!")

        return yolo_model, resnet_model, tokenizer, lstm_model

    except Exception as e:
        st.error(f"Error loading models:\n\n{e}")
        st.stop()


# Load models
yolo_model, resnet_model, tokenizer, lstm_model = load_all_models()

MAX_LENGTH = 31
