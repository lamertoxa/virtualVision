import json

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50
from keras import layers, models
import os
import datetime

# Define the root path to your dataset
root_dataset_path = "dataset/Junctions-train/Junctions-train"

# Set up data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input,
    validation_split=0.2  # using 80/20 train/validation split
)

train_generator = train_datagen.flow_from_directory(
    root_dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    root_dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load the ResNet50 model with pre-trained ImageNet weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=100
)

# Save the model
formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
model_save_path = f"models/{formatted_time}"
os.makedirs(model_save_path, exist_ok=True)
model.save(os.path.join(model_save_path, "model.h5"))

# Save class indices
class_indices = train_generator.class_indices
with open(os.path.join(model_save_path, "class_indices.json"), "w") as f:
    json.dump(class_indices, f)

print("Model and class indices saved successfully.")
