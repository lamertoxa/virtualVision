import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Define the path to the test images
test_images_path = "dataset/Junctions-test/Junctions-test"


def get_latest_model_dir(models_dir='models'):
    model_dirs = [os.path.join(models_dir, d) for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    latest_model_dir = max(model_dirs, key=os.path.getmtime)
    return latest_model_dir


# Get the path to the latest model
model_path = os.path.join(get_latest_model_dir(), 'model.h5')
model = load_model(model_path)

# Set up data generator for test images
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

test_generator = test_datagen.flow_from_directory(
    test_images_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Make predictions on the test images
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Generate a classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("Classification Report:\n", report)

# Generate a confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Visualize predictions
for i in range(len(predictions)):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    img_path = os.path.join(test_images_path, test_generator.filenames[i])
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title('Test Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.bar(class_labels, predictions[i])
    plt.title('Prediction Probabilities')
    plt.xlabel('Class')
    plt.ylabel('Probability')

    plt.tight_layout()
    plt.show()

    input("Press Enter to continue...")
