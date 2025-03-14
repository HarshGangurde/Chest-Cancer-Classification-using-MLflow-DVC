import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load the trained model
        model_path = os.path.join("artifacts", "training", "model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = load_model(model_path)

        # Load and preprocess the image
        img = image.load_img(self.filename, target_size=(224, 224))  # Resize
        img = image.img_to_array(img) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Perform prediction
        predictions = model.predict(img)
        result_index = np.argmax(predictions)  # Get index of highest probability

        # Define class labels (Modify based on your dataset)
        class_labels = ["Adenocarcinoma", "Benign", "Normal", "Squamous Cell Carcinoma"]  
        
        # Get the corresponding class name
        predicted_class = class_labels[result_index]

        return [{"image": predicted_class}]
