"""
Ensures the project root is on sys.path so test modules can use absolute
imports like `from skills.assessment import assess_answer`, regardless of
how/where pytest is invoked from. Also loads .env so GOOGLE_API_KEY is
available to tests that import Skills directly without going through
agent.py (which is the only other place load_dotenv() is called).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
