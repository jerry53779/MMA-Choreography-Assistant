# choreography/visualize_summary.py
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import io
import json
from typing import Dict, Any, Optional

def generate_summary_image(session_data: Dict[str, Any]) -> Optional[Image.Image]:
    """
    Generates and returns a PIL Image of a summary visualization from session data.

    Args:
        session_data: A dictionary containing the session data.

    Returns:
        A PIL.Image object containing the summary plot, or None if an error occurs.
    """
    try:
        actions = session_data.get('actions', [])
        session_timestamp = session_data.get('timestamp', 'N/A')

        fig, ax = plt.subplots(figsize=(6, 4))
        if not actions:
            ax.text(0.5, 0.5, "No Actions Recorded", ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.axis("off")
        else:
            move_counts = session_data.get('summary', Counter(actions))
            ax.bar(move_counts.keys(), move_counts.values(), color='skyblue')
            ax.set_title(f"Session Summary: {len(actions)} total moves\n", fontsize=12)
            ax.set_ylabel("Count")
            ax.set_xlabel("Move")
            plt.xticks(rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        
        return Image.open(buf)
    
    except Exception as e:
        print(f"An unexpected error occurred in generate_summary_image: {e}")
        return None