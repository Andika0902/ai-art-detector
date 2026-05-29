import numpy as np

from tensorflow.keras.models import load_model

from utils import preprocess_image, extract_features

# =========================================
# LOAD MODEL
# =========================================

model = load_model("model/model.keras")

# =========================================
# PREDICT
# =========================================

def predict_image(img_path):

    img = preprocess_image(img_path)

    img = np.expand_dims(img, axis=0)

    feat = extract_features(img_path)

    feat = np.expand_dims(feat, axis=0)

    prediction = model.predict(
        [img, feat],
        verbose=0
    )

    confidence = float(prediction[0][0])

    # Clamp supaya aman
    confidence = max(0.0, min(confidence, 1.0))

    if confidence > 0.5:

        label = "AI Art"

        percent = confidence * 100

    else:

        label = "Human Illustration"

        percent = (1 - confidence) * 100

    return (
        label,
        round(percent, 2)
    )