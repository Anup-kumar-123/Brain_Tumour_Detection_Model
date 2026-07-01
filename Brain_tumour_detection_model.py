import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.efficientnet import preprocess_input
import warnings as w
w.filterwarnings('ignore')

# Load model
model = load_model(r"brain_tumor_model.keras")

# Load class labels
with open(r"class_label.json") as f:
    class_labels = json.load(f)

IMG_SIZE = 224


def gradcam(image_path, layer_name="top_conv"):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

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

    cam = cv2.resize(cam.numpy() if hasattr(cam, "numpy") else cam, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_INFERNO)

    return heatmap


def predict_and_show(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found")
        return

    # Resize for model
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Preprocess
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    class_name = class_labels[str(class_idx)]
    confidence = np.max(preds)

    print("\n======================")
    print("Prediction:", class_name)
    print("Confidence:", round(confidence * 100, 2), "%")
    print("======================\n")

    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
 
    heatmap = gradcam(image_path)

    # Make tumor more visible by blending strongly
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    overlay = cv2.addWeighted(img_resized, 0.4, heatmap, 0.6, 0)

    # Convert for matplotlib
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Plotting Pictures.
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original MRI Scan")
    plt.imshow(gray_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Tumor Detection: {class_name} ({round(confidence*100,2)}%)")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


predict_and_show(r"D:\Datasets\BT-MRI Dataset\BT-MRI Dataset\Testing\No-tumor\BT-MRI NO Test (16).jpg")
predict_and_show(r"D:\Datasets\BT-MRI Dataset\BT-MRI Dataset\Testing\Glioma\BT-MRI Test GL (41).jpg")
predict_and_show(r"D:\Datasets\BT-MRI Dataset\BT-MRI Dataset\Testing\Meningioma\BT-MRI ME Test (6).jpg")
predict_and_show(r"D:\Datasets\BT-MRI Dataset\BT-MRI Dataset\Testing\Pituitary\BT-MRI PI Test (5).jpg")