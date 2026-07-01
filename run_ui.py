"""Launch the STARtrack Gradio UI."""
import runpy
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Gradio accesses a renamed Starlette constant; the warning originates from starlette.status.
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"starlette\..*")

runpy.run_module("skills.ui", run_name="__main__")
