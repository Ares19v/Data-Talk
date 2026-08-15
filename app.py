"""
HuggingFace Spaces entry point.
Imports and launches the Gradio app.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from ui.app import build_ui

demo = build_ui()

if __name__ == "__main__":
    demo.launch()
