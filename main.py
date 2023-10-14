import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator




import json
import os

# Load the JSON metadata
with open('metadata.json', 'r') as json_file:
    metadata = json.load(json_file)

# Specify the directory where your image files are stored
image_directory = 'image_data/'

# Create a dictionary to associate image file paths with metadata
image_data = {}

for entry in metadata['images']:
    image_id = entry['image_id']
    patient_id = entry['patient_id']
    image_file_path = os.path.join(image_directory, f'1{image_id}.png')  # Adjust the filename pattern as needed
    image_data[image_id] = {
        'image_file': image_file_path,
        'patient_id': patient_id
    }

# Example: Accessing image and metadata for a specific image
image_id_to_retrieve = 1
if image_id_to_retrieve in image_data:
    image_info = image_data[image_id_to_retrieve]
    image_file_path = image_info['image_file']
    patient_id = image_info['patient_id']
    # You can now load and process the image file, e.g., using a library like OpenCV or PIL
    # For example, if you're using OpenCV:
    # import cv2
    # image = cv2.imread(image_file_path)
    print(f"Image ID: {image_id_to_retrieve}, Patient ID: {patient_id}")
else:
    print(f"Image ID {image_id_to_retrieve} not found in metadata.")












# Define data directories
train_dir = r'C:\Users\Dipak\Desktop\Spending Time With Code\data\image_data'  
validation_dir = r'D:\MSC_Project\CT\stage_2_test'  # Directory with validation images

# Define data generators with augmentation (adjust parameters as needed)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess images using generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Create a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs as needed
    validation_data=validation_generator
)

# Save the model
model.save('hemorrhage_detection_model.h5')
