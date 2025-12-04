import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# Helper functions
# -------------------------

def load_images(dataset_path, img_size=(128, 128)):
    """
    Load images from dataset, resize, normalize, and apply simple filtering.
    Beginner style: loops over folders manually, lots of prints.
    """
    images = []
    labels = []
    classes = os.listdir(dataset_path)
    print("Classes found:", classes)

    for idx, cls in enumerate(classes):
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.isdir(cls_path):
            continue
        print(f"Loading class '{cls}' with label {idx}")
        for file in os.listdir(cls_path):
            file_path = os.path.join(cls_path, file)
            try:
                # Read image
                img = cv2.imread(file_path)
                if img is None:
                    print("Warning: could not read", file_path)
                    continue

                # Resize image
                img = cv2.resize(img, img_size)

                # Apply simple filtering: Gaussian blur to reduce noise
                img = cv2.GaussianBlur(img, (3, 3), 0)

                # Normalize to 0-1
                img = img / 255.0

                images.append(img)
                labels.append(idx)
            except Exception as e:
                print("Error loading image:", file_path, e)
    return np.array(images), np.array(labels)

# -------------------------
# Load dataset
# -------------------------

dataset_dir = "dataset"
print("Loading images from", dataset_dir)
X, y = load_images(dataset_dir)
print("Images loaded:", X.shape)
print("Labels loaded:", y.shape)

# Convert labels to categorical (binary)
y_cat = to_categorical(y, num_classes=2)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -------------------------
# Data augmentation
# -------------------------
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
)
datagen.fit(X_train)
print("Data augmentation initialized.")

# -------------------------
# Build CNN model (beginner style)
# -------------------------
model = Sequential()

# First conv block
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D((2,2)))

# Second conv block
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

# Third conv block
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # binary classification

# -------------------------
# Compile model
# -------------------------
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled successfully.")

# -------------------------
# Train model
# -------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_test, y_test),
    epochs=30,
    callbacks=[early_stop]
)
print("Model training complete.")

# -------------------------
# Evaluate model
# -------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# -------------------------
# Save model
# -------------------------
model.save("melanoma_cnn_model.h5")
print("Model saved as melanoma_cnn_model.h5")

# -------------------------
# Predict on a few samples (beginner style)
# -------------------------
print("Making predictions on first 5 test images:")
for i in range(min(5, X_test.shape[0])):
    img = np.expand_dims(X_test[i], axis=0)
    pred = model.predict(img)
    print(f"Predicted: {pred}, True label: {y_test[i]}")
