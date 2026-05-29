import webview
import tensorflow as tf
import os
import base64
from predict import predict_image
from tkinter import filedialog

# ===== LOAD MODEL =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'model', 'model.keras')

model = tf.keras.models.load_model(model_path)

# ===== API =====
class API:

    def upload_image(self):

        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            return None

        label, confidence = predict_image(file_path)

        with open(file_path, "rb") as image_file:
            encoded = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return {
            "image": encoded,
            "label": label,
            "confidence": confidence
        }

# ===== WEBVIEW =====
api = API()

webview.create_window(
    'AI Art Detector',
    os.path.join(BASE_DIR, 'src', 'web', 'index.html'),
    js_api=api,
    width=1200,
    height=700
)

webview.start()