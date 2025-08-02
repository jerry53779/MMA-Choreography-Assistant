# app.py
import streamlit as st
import cv2
import numpy as np
import time

from pose.detector import PoseDetector
from pose.classifier import ActionClassifier
from choreography.shadow import ShadowFighter
from choreography.suggester import MoveSuggester
from utils.visualizer import draw_keypoints
from choreography.chatbot import MMAChatbot
from choreography.visualize_summary import generate_summary_image

# --- Streamlit Caching and Initialization ---
@st.cache_resource
def get_pose_detector():
    """Initializes and caches the PoseDetector."""
    return PoseDetector()

@st.cache_resource
def get_action_classifier():
    """
    Initializes and caches the ActionClassifier with the new model.
    """
    # Using the new model file for action classification
    return ActionClassifier(model_path="data/best_punch_prediction_model.joblib")

@st.cache_resource
def get_chatbot():
    """
    Initializes and caches the MMAChatbot.
    NOTE: Hard-coding the API key is NOT recommended for security.
    This approach is used here for simplicity in a local-only project.
    For production, always use `st.secrets` or environment variables.
    """
    api_key = "AIzaSyAohAlxNjFb9cM3E5Mnjbtvu6tByElxTnE"
    return MMAChatbot(api_key=api_key)

@st.cache_resource
def get_camera():
    """Initializes and caches the video capture object."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return None
    return cap

# --- Application Setup ---
st.set_page_config(page_title="🥋 MMA Choreography Studio", layout="wide")
st.title("🥋 Martial Arts Choreography - Real-Time AI Assistant")

pose_detector = get_pose_detector()
classifier = get_action_classifier()
chatbot = get_chatbot()
suggester = MoveSuggester(chatbot_instance=chatbot)
shadow = ShadowFighter()

# --- Session State Management ---
if "session_active" not in st.session_state:
    st.session_state.session_active = False
if "show_summary" not in st.session_state:
    st.session_state.show_summary = False
if "session_data" not in st.session_state:
    st.session_state.session_data = None

# --- Streamlit UI and Control Flow ---
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    start_button = st.button("▶ Start Session", type="primary")
with col2:
    stop_button = st.button("⏹ Stop Session", type="secondary")

if start_button:
    st.session_state.session_active = True
    st.session_state.show_summary = False
    st.session_state.session_data = None
    st.toast("Session started!", icon="✅")

if stop_button:
    st.session_state.session_active = False
    st.session_state.show_summary = True
    suggester.save_session()
    st.session_state.session_data = suggester.get_last_session_data()
    st.toast("Session ended!", icon="⏹")

video_placeholder = st.empty()
summary_placeholder = st.empty()

# --- Main Application Loop ---
if st.session_state.session_active:
    st.sidebar.markdown("## Control Panel")
    if st.sidebar.button("Toggle Shadow Mode"):
        shadow.toggle_shadow()
    if st.sidebar.button("Toggle Recording"):
        shadow.toggle_recording()

    cap = get_camera()
    if cap:
        while st.session_state.session_active:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera stream ended. Stopping session.")
                st.session_state.session_active = False
                break
            
            frame = cv2.flip(frame, 1)
            keypoints = pose_detector.detect(frame)
            
            if keypoints is not None:
                h, w, _ = frame.shape
                pixel_keypoints = np.array([[int(p[0] * w), int(p[1] * h)] for p in keypoints])

                frame_with_drawings = shadow.draw(frame.copy(), pixel_keypoints)
                move = classifier.classify(keypoints)
                suggester.log_action(move)
                
                suggestion = suggester.get_suggestion()
                
                cv2.putText(frame_with_drawings, f"Move: {move}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_with_drawings, f"Suggestion: {suggestion}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(frame_with_drawings, f"Status: {shadow.status.value}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                shadow.update(pixel_keypoints)
            else:
                frame_with_drawings = frame

            video_placeholder.image(cv2.cvtColor(frame_with_drawings, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

# Post-session summary display
if st.session_state.show_summary:
    summary_placeholder.markdown("## Session Summary")
    if st.session_state.session_data:
        summary_image = generate_summary_image(st.session_state.session_data)
        if summary_image:
            summary_placeholder.image(summary_image)
        
        session_moves = " -> ".join(st.session_state.session_data.get("actions", []))
        summary_placeholder.markdown(f"**Moves Performed**: `{session_moves}`")

        with st.spinner("🤖 Getting AI Feedback..."):
            ai_feedback = chatbot.get_feedback(session_moves)
            summary_placeholder.info(f"💡 AI Suggestion: {ai_feedback}")