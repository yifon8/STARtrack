"""Launch the STARtrack Gradio UI."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

runpy.run_module("skills.ui", run_name="__main__")
