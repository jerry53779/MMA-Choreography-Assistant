# pose/classifier.py
import numpy as np
import joblib
from typing import List, Any, Optional

class ActionClassifier:
    """
    Classifies a pose based on keypoint data using a pre-trained machine learning model.
    """
    def __init__(self, model_path: str = "data/best_punch_prediction_model.joblib"):
        """
        Initializes the classifier by loading a pre-trained model.

        Args:
            model_path: The path to the pickled model file (e.g., a `.joblib` file).
        """
        self.model = None
        self.labels = []
        try:
            # We use joblib.load as it's specifically designed for scikit-learn models
            self.model = joblib.load(model_path)
            self.labels = self.model.classes_  # Assumes a scikit-learn model with a `.classes_` attribute
            print(f"Successfully loaded classifier model from: {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file not found at '{model_path}'. Classifier is disabled.")
        except Exception as e:
            print(f"Error loading model from '{model_path}': {e}. Classifier is disabled.")

    def classify(self, keypoints: List[List[float]]) -> str:
        """
        Classifies a given set of pose keypoints.

        Args:
            keypoints: A list of lists of normalized coordinates from the PoseDetector.

        Returns:
            The predicted move label as a string, or "Unknown" if no prediction
            can be made.
        """
        if self.model is None or not keypoints:
            return "Unknown"

        try:
            # Flatten the keypoints list into a single feature vector
            feature_vector = np.array(keypoints).flatten().reshape(1, -1)
            
            # Predict the class
            prediction = self.model.predict(feature_vector)[0]
            
            return prediction
        except Exception as e:
            return f"Classification Error: {e}"

