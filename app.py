import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
from PIL import Image

# -------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection & Localization",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #0E1117;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #555555;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 Brain Tumor Detection & Localization</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an MRI scan to detect tumors, classify the type, and highlight the location via Grad-CAM.</div>', unsafe_allow_html=True)

st.divider()

# -------------------------------------------------------------------------
# 2. Cache Model & Labels Loading
# -------------------------------------------------------------------------
@st.cache_resource
def load_tumor_model():
    # Loading your keras model file
    model = tf.keras.models.load_model('brain_tumor_model.keras')
    return model

@st.cache_data
def load_labels():
    # Loading your class labels
    with open('class_label.json', 'r') as f:
        labels = json.load(f)
    return labels

try:
    model = load_tumor_model()
    labels = load_labels()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.info("Please ensure 'brain_tumor_model.keras' and 'class_label.json' are in the same directory.")
    st.stop()

# -------------------------------------------------------------------------
# 3. Grad-CAM Implementation Placeholder
# -------------------------------------------------------------------------
def generate_gradcam(img_array, model, last_conv_layer_name="top_conv"):
    """
    Generates a Grad-CAM heatmap overlay. 
    Replace 'top_conv' with the final convolutional layer name of your specific model.
    """
    try:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception:
        # Fallback placeholder if layer names mismatch or custom structure prevents standard hook
        return None

def overlay_heatmap(heatmap, original_img):
    # Resize heatmap to match original image dimensions
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap onto original image
    superimposed_img = heatmap * 0.4 + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

# -------------------------------------------------------------------------
# 4. Sidebar & File Upload
# -------------------------------------------------------------------------
st.sidebar.header("Navigation & Info")
st.sidebar.markdown("""
This application uses a Deep Learning model to analyze brain MRI slices. 
* **Step 1:** Upload MRI image (.jpg, .jpeg, .png)
* **Step 2:** Model checks for tumor presence and identifies the variant.
* **Step 3:** Grad-CAM pinpoints the region of interest.
""")

uploaded_file = st.sidebar.file_uploader("Choose an MRI Scan Image...", type=["jpg", "jpeg", "png"])

# -------------------------------------------------------------------------
# 5. Main Application Logic
# -------------------------------------------------------------------------
if uploaded_file is not None:
    # Open and format image
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded MRI Scan")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("Analysis & Detection")
        
        with st.spinner('Analyzing scan...'):
            # Image preprocessing (Adjust target_size to match what your training code used, e.g., 224x224)
            target_size = (224, 224) 
            img_resized = cv2.resize(img_np, target_size)
            img_array = np.expand_dims(img_resized, axis=0) / 255.0
            
            # Predict
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0]) # Use softmax if output layer didn't include it
            class_idx = np.argmax(score)
            
            # Read class label mapping securely
            # Handles list layout or dict layout safely
            if isinstance(labels, list):
                predicted_class = labels[class_idx]
            else:
                predicted_class = labels.get(str(class_idx), f"Class {class_idx}")
                
            confidence = 100 * np.max(score)
            
        # Display Results Metric
        st.metric(label="Diagnosis Result", value=str(predicted_class))
        st.progress(int(confidence))
        st.write(f"**Confidence Level:** {confidence:.2f}%")
        
        st.divider()
        
        # Grad-CAM Display
        st.subheader("Grad-CAM Localization")
        # Try to extract the last conv layer dynamically if possible, or fallback
        conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        
        if conv_layers:
            last_conv = conv_layers[-1]
            heatmap = generate_gradcam(img_array, model, last_conv)
            
            if heatmap is not None:
                gradcam_img = overlay_heatmap(heatmap, img_resized)
                st.image(gradcam_img, caption=f"Tumor Region Highlighted (Layer: {last_conv})", use_column_width=True)
            else:
                st.warning("Could not automatically process Grad-CAM for this specific architecture layout.")
        else:
            st.info("No explicit standard Conv2D layers detected to map Grad-CAM overlay safely.")

else:
    # Landing state info box
    st.info("👈 Please upload an MRI scan image from the sidebar to begin interpretation.")
