from flask import Flask
from flask import request
from flask import jsonify
from flask import send_from_directory

import os
import base64

from predict import predict_image

# =========================================
# INIT
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WEB_DIR = os.path.join(BASE_DIR, "web")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

# =========================================
# ROUTE HTML
# =========================================

@app.route("/")
def home():

    return send_from_directory(
        WEB_DIR,
        "index.html"
    )

# =========================================
# STATIC FILES
# =========================================

@app.route("/<path:path>")
def static_files(path):

    return send_from_directory(
        WEB_DIR,
        path
    )

# =========================================
# PREDICT API
# =========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        })

    file = request.files["image"]

    save_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    file.save(save_path)

    label, confidence = predict_image(save_path)

    with open(save_path, "rb") as img_file:

        encoded = base64.b64encode(
            img_file.read()
        ).decode("utf-8")

    return jsonify({

        "image": encoded,

        "label": label,

        "confidence": confidence
    })

# =========================================
# RUN
# =========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )