import cv2
import numpy as np
import pywt

from skimage.feature import local_binary_pattern

from tensorflow.keras.applications.densenet import preprocess_input

IMG_SIZE = 224

# =========================================
# PREPROCESS IMAGE
# =========================================

def preprocess_image(path):

    img = cv2.imread(path)

    if img is None:
        raise ValueError("Gambar gagal dibaca")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = preprocess_input(img.astype(np.float32))

    return img


# =========================================
# FFT FEATURE
# =========================================

def extract_fft(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fft = np.fft.fft2(gray)

    fft_shift = np.fft.fftshift(fft)

    magnitude = np.log(np.abs(fft_shift) + 1)

    magnitude = cv2.resize(magnitude, (32, 32))

    magnitude = magnitude.flatten()

    magnitude /= (np.max(magnitude) + 1e-6)

    return magnitude


# =========================================
# LBP FEATURE
# =========================================

def extract_lbp(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(
        gray,
        P=16,
        R=2,
        method='uniform'
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 19),
        range=(0, 18)
    )

    hist = hist.astype(np.float32)

    hist /= (hist.sum() + 1e-6)

    return hist


# =========================================
# WAVELET FEATURE
# =========================================

def extract_wavelet(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coeffs = pywt.wavedec2(gray, 'haar', level=2)

    features = []

    for coeff in coeffs:

        if isinstance(coeff, tuple):

            for arr in coeff:

                arr = cv2.resize(arr, (16, 16))

                features.extend(arr.flatten())

        else:

            arr = cv2.resize(coeff, (16, 16))

            features.extend(arr.flatten())

    features = np.array(features)

    features /= (np.max(features) + 1e-6)

    return features


# =========================================
# SRM FEATURE
# =========================================

def extract_srm(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0]
    ])

    filtered = cv2.filter2D(gray, -1, kernel)

    filtered = cv2.resize(filtered, (32, 32))

    filtered = filtered.flatten()

    filtered = filtered.astype(np.float32)

    filtered /= (np.max(np.abs(filtered)) + 1e-6)

    return filtered


# =========================================
# COMBINE FEATURES
# =========================================

def extract_features(path):

    img = cv2.imread(path)

    if img is None:
        raise ValueError("Gambar gagal dibaca")

    img = cv2.resize(img, (224, 224))

    fft = extract_fft(img)

    lbp = extract_lbp(img)

    wavelet = extract_wavelet(img)

    srm = extract_srm(img)

    features = np.concatenate([
        fft,
        lbp,
        wavelet,
        srm
    ])

    return features.astype(np.float32)