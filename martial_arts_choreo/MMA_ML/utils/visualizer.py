# utils/visualizer.py
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

KEYPOINT_MAPPING = {
    'nose': 0, 'right_shoulder': 6, 'right_elbow': 8, 'right_wrist': 10,
    'left_shoulder': 5, 'left_elbow': 7, 'left_wrist': 9, 'right_hip': 12,
    'right_knee': 14, 'right_ankle': 16, 'left_hip': 11, 'left_knee': 13,
    'left_ankle': 15,
}

CONNECTIONS = [
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'left_shoulder'), ('right_hip', 'left_hip'),
    ('right_shoulder', 'right_hip'), ('left_shoulder', 'left_hip'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
]

def draw_keypoints(
    image: np.ndarray,
    keypoints: Optional[np.ndarray],
    keypoint_map: Dict[str, int] = KEYPOINT_MAPPING,
    connections: List[Tuple[str, str]] = CONNECTIONS,
    node_color: Tuple[int, int, int] = (255, 255, 255),
    line_color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    radius: int = 4
) -> np.ndarray:
    """Draws keypoints and a skeleton on an image."""
    if keypoints is None or len(keypoints) == 0:
        return image

    for x, y in keypoints:
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(image, (int(x), int(y)), radius, node_color, -1)

    for connection_name1, connection_name2 in connections:
        try:
            i = keypoint_map[connection_name1]
            j = keypoint_map[connection_name2]
            pt1 = keypoints[i]
            pt2 = keypoints[j]
            if not np.isnan(pt1[0]) and not np.isnan(pt1[1]) and \
               not np.isnan(pt2[0]) and not np.isnan(pt2[1]):
                cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), line_color, thickness)
        except (KeyError, IndexError):
            pass

    return image