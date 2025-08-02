import pandas as pd
import numpy as np
import os
import cv2
import mediapipe as mp
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Configuration ---
# Update these lines to your specific paths and folder names
BASE_DATA_DIRECTORY = r'D:\OGDataSet'
SUBFOLDERS = ['BoxingSpeedBag', 'BoxingPunchingBag']
TARGET_COLUMN_NAME = 'move_type'

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_features_from_video(video_path, label):
    """
    Processes a single video file, extracts pose landmarks, and creates a DataFrame
    of features and the given label.
    """
    print(f"Processing video: {video_path}")
    
    # List to hold all landmark data from all frames
    all_frame_data = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR frame to RGB, as MediaPipe requires RGB input
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image to find pose landmarks
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Extract landmarks for this frame
                frame_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    # Append x, y, z coordinates and visibility score
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                # Append the landmark data and the label for the current frame
                all_frame_data.append(frame_landmarks + [label])

    cap.release()
    
    # Get the names for the landmark columns
    landmark_names = []
    for i in range(len(mp_pose.PoseLandmark)):
        landmark_names.extend([f'lm_{i}_x', f'lm_{i}_y', f'lm_{i}_z', f'lm_{i}_visibility'])
    
    # Create the DataFrame
    if all_frame_data:
        df = pd.DataFrame(all_frame_data, columns=landmark_names + [TARGET_COLUMN_NAME])
        return df
    else:
        return None

def load_data_from_folders(base_dir, subfolders, target_column_name):
    """
    Iterates through subfolders, processes all videos, and combines the results.
    """
    print("--- Starting Data Loading from Videos ---")
    
    all_dataframes = []
    
    for folder in subfolders:
        folder_path = os.path.join(base_dir, folder)
        # Use glob to find all video files (mp4, avi, mov, etc.)
        all_videos = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not all_videos:
            print(f"Warning: No video files found in '{folder_path}'. Skipping.")
            continue
            
        print(f"Found {len(all_videos)} video files in '{folder}'.")
        
        for video_file in all_videos:
            video_path = os.path.join(folder_path, video_file)
            df = extract_features_from_video(video_path, folder)
            if df is not None:
                all_dataframes.append(df)
    
    if not all_dataframes:
        print("Error: No data was loaded from any of the specified folders.")
        return None
        
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print("All video data concatenated successfully.")
    print("Combined data shape:", combined_df.shape)
    
    print("--- Data Loading Complete ---")
    return combined_df

def clean_data(df):
    """
    Performs data cleaning steps.
    """
    print("\n--- Starting Data Cleaning ---")
    df.dropna(inplace=True)
    print("Missing values handled. New data shape:", df.shape)
    df.drop_duplicates(inplace=True)
    print("Duplicate rows removed. New data shape:", df.shape)
    # Outlier removal can be tricky with pose data, but we'll use a basic Z-score
    # on the coordinate data.
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    initial_shape = df.shape
    for col in numerical_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        df = df[(z_scores < 3)]
    print(f"Outliers removed. Data shape changed from {initial_shape} to {df.shape}")
    print("--- Data Cleaning Complete ---")
    return df

def preprocess_data(df, target_column):
    """
    Prepares the data for a machine learning model.
    """
    print("\n--- Starting Data Preprocessing ---")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Target classes: {list(label_encoder.classes_)}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled using StandardScaler.")
    print("--- Data Preprocessing Complete ---")
    return X_scaled, y_encoded, label_encoder

def train_model(X, y):
    """
    Splits the data, trains a RandomForestClassifier, and evaluates it.
    """
    print("\n--- Starting Model Training ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("RandomForestClassifier trained successfully.")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return model

if __name__ == "__main__":
    # --- Main Execution Flow ---
    
    # Step 1: Load and process the raw video data
    raw_df = load_data_from_folders(BASE_DATA_DIRECTORY, SUBFOLDERS, TARGET_COLUMN_NAME)
    if raw_df is None:
        exit()
    
    # Step 2: Clean the extracted data
    cleaned_df = clean_data(raw_df)
    
    # Step 3: Preprocess the data for the model
    features, labels, label_encoder = preprocess_data(cleaned_df, TARGET_COLUMN_NAME)
    
    # Step 4: Train and evaluate the model
    trained_model = train_model(features, labels)
    
    print("\n--- Script Finished ---")