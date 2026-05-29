import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import (
    DenseNet201,
    ResNet50
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import preprocess_image, extract_features

# =========================================
# CONFIG
# =========================================

DATASET_DIR = "dataset"

MODEL_PATH = "model/model.keras"

IMG_SIZE = 224

# =========================================
# LOAD DATASET
# =========================================

X_image = []
X_feature = []
y = []

classes = ['human', 'ai']

for label, class_name in enumerate(classes):

    folder = os.path.join(DATASET_DIR, class_name)

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        try:

            img = preprocess_image(path)

            feat = extract_features(path)

            X_image.append(img)

            X_feature.append(feat)

            y.append(label)

        except Exception as e:

            print("ERROR:", path)

X_image = np.array(X_image, dtype=np.float32)

X_feature = np.array(X_feature, dtype=np.float32)

y = np.array(y)

# =========================================
# NORMALIZATION
# =========================================

X_feature /= (np.max(X_feature) + 1e-6)

# =========================================
# SPLIT
# =========================================

(
    Ximg_train,
    Ximg_test,
    Xfeat_train,
    Xfeat_test,
    y_train,
    y_test
) = train_test_split(
    X_image,
    X_feature,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================
# DATA AUGMENTATION
# =========================================

datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    shear_range=0.15
)

# =========================================
# INPUT
# =========================================

image_input = layers.Input(
    shape=(224,224,3),
    name="image_input"
)

# =========================================
# DENSENET201 BRANCH
# =========================================

dense_base = DenseNet201(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

for layer in dense_base.layers[:-80]:
    layer.trainable = False

dense = dense_base(image_input)

dense = layers.Dense(
    512,
    activation='relu'
)(dense)

dense = layers.BatchNormalization()(dense)

dense = layers.Dropout(0.5)(dense)

# =========================================
# RESNET50 BRANCH
# =========================================

resnet_base = ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

# Rename layer supaya aman
for layer in resnet_base.layers:
    layer._name = "resnet_" + layer.name

for layer in resnet_base.layers[:-50]:
    layer.trainable = False

res = resnet_base(image_input)

res = layers.Dense(
    512,
    activation='relu'
)(res)

res = layers.BatchNormalization()(res)

res = layers.Dropout(0.5)(res)

# =========================================
# CNN COMBINE
# =========================================

cnn = layers.concatenate([
    dense,
    res
])

# =========================================
# ATTENTION LAYER
# =========================================

attention = layers.Dense(
    cnn.shape[-1],
    activation='sigmoid'
)(cnn)

cnn = layers.multiply([
    cnn,
    attention
])

cnn = layers.Dense(
    512,
    activation='relu'
)(cnn)

cnn = layers.BatchNormalization()(cnn)

cnn = layers.Dropout(0.5)(cnn)

cnn = layers.Dense(
    256,
    activation='relu'
)(cnn)

cnn = layers.Dropout(0.4)(cnn)

# =========================================
# FEATURE INPUT
# =========================================

feature_input = layers.Input(
    shape=(X_feature.shape[1],),
    name="feature_input"
)

f = layers.Dense(
    512,
    activation='relu'
)(feature_input)

f = layers.BatchNormalization()(f)

f = layers.Dropout(0.5)(f)

f = layers.Dense(
    256,
    activation='relu'
)(f)

f = layers.Dropout(0.4)(f)

f = layers.Dense(
    128,
    activation='relu'
)(f)

# =========================================
# FINAL COMBINE
# =========================================

combined = layers.concatenate([
    cnn,
    f
])

z = layers.Dense(
    512,
    activation='relu'
)(combined)

z = layers.BatchNormalization()(z)

z = layers.Dropout(0.5)(z)

z = layers.Dense(
    256,
    activation='relu'
)(z)

z = layers.Dropout(0.4)(z)

z = layers.Dense(
    128,
    activation='relu'
)(z)

z = layers.Dropout(0.3)(z)

output = layers.Dense(
    1,
    activation='sigmoid'
)(z)

# =========================================
# FINAL MODEL
# =========================================

model = models.Model(
    inputs=[
        image_input,
        feature_input
    ],
    outputs=output
)

# =========================================
# COMPILE
# =========================================

model.compile(

    optimizer=Adam(
        learning_rate=0.0001
    ),

    loss=tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0.05
    ),

    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

# =========================================
# CALLBACKS
# =========================================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# =========================================
# TRAIN
# =========================================

history = model.fit(

    [Ximg_train, Xfeat_train],
    y_train,

    validation_data=(
        [Ximg_test, Xfeat_test],
        y_test
    ),

    epochs=40,

    batch_size=16,

    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ]
)

# =========================================
# SAVE MODEL
# =========================================

model.save(MODEL_PATH)

print("\nMODEL BERHASIL DISIMPAN")