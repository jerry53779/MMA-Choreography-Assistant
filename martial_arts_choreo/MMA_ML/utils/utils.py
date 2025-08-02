# utils/utils.py
import json
import os
from typing import Dict, Any, List, Optional
from collections import Counter

def load_session_data(session_file_path: str) -> Optional[Dict[str, Any]]:
    """Loads and parses session data from a JSON file."""
    try:
        with open(session_file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

def get_session_summary_counts(session_data: Dict[str, Any]) -> Dict[str, int]:
    """Extracts and returns the move counts from a session data dictionary."""
    actions = session_data.get('actions', [])
    return dict(Counter(actions))

def get_all_session_files(directory: str = "data") -> List[str]:
    """Finds and returns a list of all session files in a given directory."""
    try:
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("session_") and f.endswith(".json")]
    except FileNotFoundError:
        return []