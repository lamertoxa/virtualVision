import tensorflow as tf

from keras.applications.resnet import preprocess_input
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import glob

from keras_preprocessing.image import img_to_array, load_img


# Function to get the most recent model directory
def get_most_recent_model_dir(models_dir):
    list_of_dirs = glob.glob(os.path.join(models_dir, '*/'))
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getmtime)
    return latest_dir

# Load the most recent model
model_directory = "models"
recent_model_dir = get_most_recent_model_dir(model_directory)
if recent_model_dir is None:
    raise FileNotFoundError("No saved model directories found.")

model = tf.keras.models.load_model(os.path.join(recent_model_dir, "model.h5"))

# Load class indices
with open(os.path.join(recent_model_dir, "class_indices.json"), "r") as f:
    class_indices = json.load(f)

# Invert the class indices dictionary to get a mapping from index to class name
class_names = {v: k for k, v in class_indices.items()}

def predict_and_show(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make a prediction
    predictions = model.predict(img_array)

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Get the name of the predicted class
    predicted_class_name = class_names[predicted_class]

    # Display the image and the prediction
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class_name} ({confidence:.2f})")
    plt.show()

# Test the model with an image
predict_and_show("city.jpg")
