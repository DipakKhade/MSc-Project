import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image




try:
        # Step 1: Load the JSON metadata
    metadata_file = 'metadata.json'  # Adjust the path to your metadata file
    image_directory = r'D:\MSC_Project\CT\Hemorrhage'  # Adjust the path to your image directory


    train_dir = r'D:\MSC_Project\CT\Hemorrhage'  
    validation_dir = r'D:\MSC_Project\CT\No Hemorrhage'

    with open(metadata_file, 'r') as json_file:
        metadata = json.load(r'C:\Users\Dipak\Desktop\Spending Time With Code\metadata.json')

    # Create a dictionary to associate image file paths with labels
    image_data = {
        r'D:\MSC_Project\CT\mixed data\1.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\2.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\3.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\4.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\5.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\6.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\7.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\8.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\9.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\10.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\11.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\12.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\13.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\14.png':'Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\15.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\16.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\17.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\18.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\19.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\20.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\21.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\22.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\23.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\24.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\25.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\26.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\27.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\28.png':'No Hemorrhage',
        r'D:\MSC_Project\CT\mixed data\29.png':'No Hemorrhage',
    }

    for entry in metadata['images']:
        image_id = entry['image_id']
        image_file_path = os.path.join(image_directory, f'{image_id}.png')  # Adjust the filename pattern as needed
        
        image_data[image_file_path] = entry  # Store the entire metadata entry for the image

        # In this example, we're associating 'Hemorrhage' with label 1 and 'No Hemorrhage' with label 0
        label = 1 if entry['diagnosis'] == 'Hemorrhage' else 0
        image_data[image_file_path] = label

    # Step 2: Define data directories
    train_dir = image_directory  # Directory with image files

    # Step 3: Define data generators with augmentation (adjust parameters as needed)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,            # Reduced rotation range
        width_shift_range=0.1,        # Reduced width shift range
        height_shift_range=0.1,       # Reduced height shift range
        shear_range=0.1,              # Reduced shear range
        zoom_range=0.1,               # Reduced zoom range
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Load and preprocess images using a generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=12,
        class_mode='binary'
    )


    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    # Step 4: Create a simple CNN model
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

    # Step 5: Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Step 6: Train the model
    history = model.fit(
        train_generator,
        epochs=15,  # Adjust the number of epochs as needed
    )

    # Step 7: Save the model
    model.save('hemorrhage_detection_model.h5')








    #####    testing the fitted model 
    # Load the trained model
    model = tf.keras.models.load_model('hemorrhage_detection_model.h5')

    # Load a new image for prediction (replace 'image_path.jpg' with the actual image file path)
    img_path = 'C:\\Users\Dipak\\Desktop\\PROJECT MSC 2\\11.png'

    # Preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.  # Normalize the image data (if needed)

    # Make a prediction
    prediction = model.predict(img)

    # Interpret the prediction
    if prediction < 0.5:
        result = "No Hemorrhage"
    else:
        result = "Hemorrhage"

    print(f'The given CT Scan contains  : {result}')



except Exception as e:
    print(e)





