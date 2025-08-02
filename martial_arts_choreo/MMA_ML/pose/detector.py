# pose/detector.py
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional

class PoseDetector:
    """
    A class to detect human poses in video frames using MediaPipe.
    """
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initializes the PoseDetector with configurable parameters.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, frame: np.ndarray) -> Optional[List[List[float]]]:
        """
        Detects pose landmarks in a given video frame.
        
        Args:
            frame: A NumPy array representing the video frame in BGR format.
        
        Returns:
            A list of lists, where each inner list contains the [x, y] normalized
            coordinates of a keypoint, or None if no pose is detected.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.append([lm.x, lm.y])
            return keypoints
            
        return None
        
    def __del__(self):
        """
        Ensures the MediaPipe resources are released when the object is destroyed.
        """
        if self.pose:
            self.pose.close()