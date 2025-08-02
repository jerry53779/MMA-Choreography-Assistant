import cv2
from pose.detector import PoseDetector
from pose.classifier import ActionClassifier
from choreography.shadow import ShadowFighter
from choreography.suggester import MoveSuggester
from choreography.visualize_summary import show_summary
from choreography.chatbot import get_feedback  # Gemini integration


# Initialize components
cap = cv2.VideoCapture(0)
detector = PoseDetector()
classifier = ActionClassifier()
shadow = ShadowFighter()
suggester = MoveSuggester()

print("Press 'r' to toggle recording, 's' for shadow mode, 'q' to quit.")

ai_feedback = ""  # Store latest Gemini feedback

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        keypoints = detector.detect(frame)
        action = "No Detection"
        suggestion = ""

        if keypoints is not None:
            action = classifier.classify(keypoints)
            suggester.log_action(action)
            suggestion = suggester.get_suggestion()
            shadow.update(keypoints)
            frame = shadow.draw(frame, keypoints)

            # Overlay move
            cv2.putText(frame, f"Move: {action}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Get AI feedback every few moves (optional optimization)
            if len(suggester.actions) % 5 == 0:
                full_sequence = suggester.get_action_sequence()
                ai_feedback = get_feedback(full_sequence)

        # Overlay status
        cv2.putText(frame, f"Status: {shadow.status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Overlay suggestion
        if suggestion:
            cv2.putText(frame, f"Suggestion: {suggestion}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 100), 2)

        # Overlay Gemini AI feedback
        if ai_feedback:
            y = 150
            for line in ai_feedback.splitlines():
                cv2.putText(frame, line.strip(), (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
                y += 20

        # Show frame
        cv2.imshow("MMA Choreo", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            shadow.toggle_recording()
        elif key == ord('s'):
            shadow.toggle_shadow()
        elif key == ord('q'):
            suggester.save_session()
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Show post-session summary
    session_file = suggester.get_last_session_file()
    if session_file:
        show_summary(session_file)
