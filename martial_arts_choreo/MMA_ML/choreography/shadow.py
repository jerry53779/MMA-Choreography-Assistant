# choreography/shadow.py
import cv2
import numpy as np
from typing import List, Tuple, Dict
from enum import Enum

KEYPOINT_MAPPING = {
    'nose': 0, 'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7,
    'right_elbow': 8, 'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11,
    'right_hip': 12, 'left_knee': 13, 'right_knee': 14, 'left_ankle': 15,
    'right_ankle': 16,
}

CONNECTIONS = [
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
    ('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
]

class ShadowStatus(Enum):
    IDLE = "Idle"
    RECORDING = "Recording"
    SHADOW = "Shadow"

class ShadowFighter:
    """Manages the state and rendering for a "shadowing" feature."""
    def __init__(self):
        self.previous_keypoints: np.ndarray = None
        self.status: ShadowStatus = ShadowStatus.IDLE
        self._keypoint_indices: Dict[str, int] = KEYPOINT_MAPPING
        self._connections: List[Tuple[str, str]] = CONNECTIONS

    def toggle_recording(self):
        self.status = ShadowStatus.RECORDING if self.status != ShadowStatus.RECORDING else ShadowStatus.IDLE

    def toggle_shadow(self):
        self.status = ShadowStatus.SHADOW if self.status != ShadowStatus.SHADOW else ShadowStatus.IDLE

    def update(self, keypoints: np.ndarray):
        """Records the current pose if in recording mode."""
        if self.status == ShadowStatus.RECORDING:
            self.previous_keypoints = keypoints

    def draw(self, frame: np.ndarray, current_keypoints: np.ndarray) -> np.ndarray:
        """Draws the shadow and the user's current pose on the frame."""
        if self.status == ShadowStatus.SHADOW and self.previous_keypoints is not None:
            frame = self._draw_stickman(
                frame, 
                self.previous_keypoints, 
                color_nodes=(0, 0, 0),
                color_lines=(0, 0, 255)
            )

        if current_keypoints is not None:
            frame = self._draw_stickman(
                frame, 
                current_keypoints, 
                color_nodes=(255, 255, 255),
                color_lines=(0, 255, 0)
            )
        return frame

    def _draw_stickman(self, frame: np.ndarray, keypoints: np.ndarray, color_nodes: Tuple[int, int, int], color_lines: Tuple[int, int, int]) -> np.ndarray:
        """Helper to draw the stickman figure on the frame."""
        connection_indices = [(self._keypoint_indices[i], self._keypoint_indices[j]) for i, j in self._connections]

        for idx in range(len(keypoints)):
            x, y = keypoints[idx]
            if not np.isnan(x) and not np.isnan(y):
                cv2.circle(frame, (int(x), int(y)), 4, color_nodes, -1)

        for i, j in connection_indices:
            if i < len(keypoints) and j < len(keypoints):
                x1, y1 = keypoints[i]
                x2, y2 = keypoints[j]
                if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_lines, 2)
        return frame