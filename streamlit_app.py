"""Streamlit entry point. Streamlit Cloud runs this file."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "app" / "Home.py"), run_name="__main__")
