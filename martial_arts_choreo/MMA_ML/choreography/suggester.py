# choreography/suggester.py
import os
import json
import uuid
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Union, Optional

from choreography.chatbot import MMAChatbot

class MoveSuggester:
    """
    Manages a user's session of moves, provides suggestions,
    and saves the session data.
    """
    def __init__(self, chatbot_instance: MMAChatbot = None):
        self.actions: List[str] = []
        self.session_id: str = str(uuid.uuid4())
        self.chatbot = chatbot_instance
        os.makedirs("data", exist_ok=True)

    def log_action(self, action: str):
        """Logs a move action, avoiding consecutive duplicates."""
        if not self.actions or self.actions[-1] != action:
            self.actions.append(action)

    def get_suggestion(self) -> str:
        """Provides a suggestion based on the move history."""
        if not self.actions:
            return "Start performing!"
        
        if self.chatbot:
            prompt = f"Given the following sequence of MMA moves: '{self.get_action_sequence()}', what is a good move to try next? Provide a concise, single-sentence suggestion."
            return self.chatbot.get_feedback(prompt)
        
        last = self.actions[-1]
        if last == "punch":
            return "Try a kick next!"
        elif last == "kick":
            return "Go for a block!"
        elif last == "block":
            return "Follow up with a punch!"
        else:
            return "Mix it up with a different move!"

    def get_action_sequence(self) -> str:
        """Returns the full sequence of moves as a readable string."""
        return " -> ".join(self.actions)

    def save_session(self):
        """Saves the current session data to a JSON file."""
        if not self.actions:
            return

        session_data: Dict[str, Union[str, List[str], Dict[str, Any]]] = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "actions": self.actions,
            "summary": dict(Counter(self.actions))
        }

        filename = f"data/session_{self.session_id}.json"
        try:
            with open(filename, "w") as f:
                json.dump(session_data, f, indent=4)
        except IOError as e:
            print(f"[Error saving session to {filename}: {e}]")

    def get_last_session_data(self) -> Optional[Dict[str, Any]]:
        """Returns the data of the current session."""
        if self.actions:
            return {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "actions": self.actions,
                "summary": dict(Counter(self.actions))
            }
        return None