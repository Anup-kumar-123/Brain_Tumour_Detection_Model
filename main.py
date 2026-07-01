import io
import json
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

app = FastAPI()

# Load model
model = load_model("brain_tumor_model.keras")

# Load labels
with open("class_label.json") as f:
    class_labels = json.load(f)

IMG_SIZE = 224


# ===================== GRAD CAM =====================
def gradcam(img_array, layer_name="top_conv"):

    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    cam = tf.reduce_sum(weights * conv_out[0], axis=-1)

    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)

    cam = cam.numpy() if hasattr(cam, "numpy") else cam
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_INFERNO)

    return heatmap


# ===================== API ENDPOINT =====================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    # Convert to OpenCV image
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    class_name = class_labels[str(class_idx)]
    confidence = float(np.max(preds))

    # Grad-CAM
    heatmap = gradcam(img_array)

    # Overlay
    overlay = cv2.addWeighted(img_resized, 0.4, heatmap, 0.6, 0)

    # Convert BGR → RGB
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Plot (side-by-side)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    ax[0].imshow(gray, cmap="gray")
    ax[0].set_title("Original MRI")
    ax[0].axis("off")

    ax[1].imshow(overlay)
    ax[1].set_title(f"{class_name} ({confidence*100:.2f}%)")
    ax[1].axis("off")

    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")