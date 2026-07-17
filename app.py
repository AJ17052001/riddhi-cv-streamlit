
import os
import pickle
import numpy as np
import tempfile
from collections import Counter
import tensorflow as tf
import streamlit as st
from PIL import Image

# Keras Imports
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# YOLO Import
from ultralytics import YOLO

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Hybrid Image Caption Generator", layout="centered")
st.title("📸 Hybrid Image Caption Generator")
st.write("Upload an image to detect objects using **YOLOv8** and generate descriptive captions via an **LSTM Network**.")

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_all_models():
    """Loads and caches all feature extractors and sequence models."""
    try:
        yolo_model = YOLO('yolov8m.pt')
        resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            
        lstm_model = load_model('caption_model.keras')
        return yolo_model, resnet_model, tokenizer, lstm_model
    except Exception as e:
        st.error(f"Error loading models. Please verify that 'tokenizer.pkl' and 'caption_model.keras' are in the same folder. \nDetails: {e}")
        return None, None, None, None

yolo_model, resnet_model, tokenizer, lstm_model = load_all_models()
MAX_LENGTH = 31

# --- FEATURE EXTRACTION LOGIC ---
def extract_image_features(img_path, yolo_model, resnet_model):
    """Extracts and combines YOLOv8m and ResNet50 features for a single image."""
    yolo_results = yolo_model(img_path, verbose=False)[0]
    boxes = yolo_results.boxes
    num_objs = len(boxes)
    yolo_feats = []
    detected_classes = []
    names = yolo_model.names
    
    if num_objs > 0:
        conf_scores = boxes.conf.cpu().numpy()
        sorted_indices = np.argsort(conf_scores)[::-1]
        for i in range(min(15, num_objs)):
            idx = sorted_indices[i]
            b = boxes[idx]
            class_id = float(b.cls[0].cpu().numpy())
            conf = float(b.conf[0].cpu().numpy())
            
            if conf > 0.40:
                detected_classes.append(names[int(class_id)])
                
            cx, cy, w, h = b.xywhn[0].cpu().numpy().tolist()
            area = w * h
            yolo_feats.extend([class_id, conf, cx, cy, w, h, area, cx, cy])
            
    padding_needed = 135 - len(yolo_feats)
    if padding_needed > 0:
        yolo_feats.extend([0.0] * padding_needed)
        
    yolo_feats.append(float(num_objs))
    yolo_vector = np.array(yolo_feats)
    
    # ResNet50
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    resnet_vector = resnet_model.predict(img_array, verbose=0).flatten()
    
    combined_vector = np.concatenate((resnet_vector, yolo_vector))
    return np.array([combined_vector]), yolo_results, detected_classes

# --- HELPER FUNCTIONS FOR INFERENCE ---
def int_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption_beam_search(model, tokenizer, image_feature, max_length, detected_classes, beam_width=5, alpha=0.8):
    start_word = '<start>'
    beam = [([start_word], 0.0)]
    img_tensor = tf.convert_to_tensor(image_feature)
    stop_words = {'a', 'the', 'and', 'is', 'in', 'on', 'of', 'with', 'at', 'to', 'by', 'an', 'are'}
    class_counts = Counter(detected_classes)
    
    for _ in range(max_length):
        candidates = []
        for seq, score in beam:
            if seq[-1] == 'end':
                candidates.append((seq, score))
                continue
                
            seq_str = ' '.join(seq)
            encoded_seq = tokenizer.texts_to_sequences([seq_str])[0]
            padded_seq = pad_sequences([encoded_seq], maxlen=max_length, padding='post')
            seq_tensor = tf.convert_to_tensor(padded_seq)
            
            yhat = model.predict_on_batch([img_tensor, seq_tensor])[0]
            current_max_prob = np.max(yhat)
            yolo_seen_broad = set()
            
            # Count-aware dynamic semantic injection
            for class_name, count in class_counts.items():
                boost_words = [class_name]
                if class_name == 'person':
                    boost_words.extend(['women', 'men', 'people', 'group', 'girls', 'friends'] if count > 1 else ['man', 'woman', 'boy', 'girl', 'runner'])
                elif class_name == 'dog':
                    boost_words.extend(['dogs', 'puppies', 'pack'] if count > 1 else ['puppy', 'hound'])
                elif class_name == 'cat':
                    boost_words.extend(['cats', 'kittens'] if count > 1 else ['kitten', 'feline'])
                elif class_name == 'car':
                    boost_words.extend(['cars', 'vehicles'] if count > 1 else ['vehicle', 'automobile'])
                elif class_name == 'handbag':
                    boost_words.extend(['bag', 'purse'])
                yolo_seen_broad.update(boost_words)
                
                if not any(w in seq for w in boost_words):
                    for w in boost_words:
                        if w in tokenizer.word_index:
                            word_idx = tokenizer.word_index[w]
                            yhat[word_idx] += (current_max_prob * 0.35)
                            
            # Suppression & Penalty Logic
            common_biases = {'dog', 'dogs', 'man', 'woman', 'boy', 'girl', 'person', 'people', 'child', 'children'}
            hallucination_risks = common_biases - yolo_seen_broad
            for risk_word in hallucination_risks:
                if risk_word in tokenizer.word_index:
                    yhat[tokenizer.word_index[risk_word]] *= 0.001
                    
            for word in set(seq):
                if word not in stop_words and word in tokenizer.word_index:
                    yhat[tokenizer.word_index[word]] *= 0.001
                    
            if len(seq) > 0:
                last_word = seq[-1]
                if last_word in tokenizer.word_index:
                    yhat[tokenizer.word_index[last_word]] *= 0.001
                    
            yhat = yhat / (np.sum(yhat) + 1e-10)
            top_indices = np.argsort(yhat)[-beam_width:]
            
            for idx in top_indices:
                word = int_to_word(idx, tokenizer)
                if word is None:
                    continue
                prob = yhat[idx]
                new_score = score + np.log(prob + 1e-10)
                new_seq = seq + [word]
                candidates.append((new_seq, new_score))
                
        def score_with_length_penalty(item):
            s, current_score = item
            L = len(s) - 1
            penalty = ((5 + L) / 6) ** alpha if L > 0 else 1.0
            return current_score / penalty
            
        beam = sorted(candidates, key=score_with_length_penalty)[-beam_width:]
        if all(seq[-1] == 'end' for seq, _ in beam):
            break
            
    best_seq = beam[-1][0]
    if best_seq[0] == '<start>':
        best_seq = best_seq[1:]
    if best_seq[-1] == 'end':
        best_seq = best_seq[:-1]
        
    return ' '.join(best_seq).strip()

# --- STREAMLIT USER INTERFACE ---
if yolo_model and resnet_model and tokenizer and lstm_model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        with st.spinner("Extracting features and generating caption..."):
            # 1. Feature Extraction
            img_feature, yolo_results, detected_classes = extract_image_features(temp_path, yolo_model, resnet_model)
            
            # 2. Text Generation
            caption = generate_caption_beam_search(
                lstm_model, tokenizer, img_feature, MAX_LENGTH, detected_classes, beam_width=5, alpha=0.8
            )
            
            # 3. Process YOLO Annotated Image using Pure Numpy/PIL (Bypassing cv2)
            plotted_img_bgr = yolo_results.plot()
            # Convert BGR to RGB array manually using numpy slicing
            plotted_img_rgb = plotted_img_bgr[:, :, ::-1] 
            
        os.unlink(temp_path)
        
        # --- DISPLAY RESULTS ---
        st.subheader("Generated Caption:")
        st.info(f"**{caption.capitalize()}.**")
        
        if detected_classes:
            st.write(f"**Objects Detected (Confidence > 40%):** {', '.join(set(detected_classes))}")
        
        st.subheader("Object Detection Visualizer:")
        st.image(plotted_img_rgb, use_container_width=True)  
                  
