"""Launch the STARtrack Gradio UI."""
import runpy
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Gradio uses a renamed Starlette constant; suppress until Gradio fixes it upstream.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")

runpy.run_module("skills.ui", run_name="__main__")
