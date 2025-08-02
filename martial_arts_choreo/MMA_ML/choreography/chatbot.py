# choreography/chatbot.py
import os
import google.generativeai as genai
from typing import Optional

class MMAChatbot:
    """
    A class to interact with the Google Gemini API for MMA coaching feedback.
    """
    def __init__(self, api_key: str):
        """
        Initializes the MMAChatbot with the provided API key.
        """
        if not api_key:
            raise ValueError("API key must be provided.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

    def get_feedback(self, move_sequence: str) -> str:
        """
        Analyzes an MMA move sequence and provides expert feedback.
        
        Args:
            move_sequence: A string describing the sequence of moves performed.
            
        Returns:
            A string containing the coach's feedback or an error message.
        """
        try:
            prompt = (
                f"You are a professional MMA coach. Analyze the following sequence of moves: '{move_sequence}'. "
                "Provide constructive feedback, point out potential weaknesses, and suggest specific improvements. "
                "Keep your response concise and easy to understand for an amateur fighter."
            )
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[AI Error: {e}]"