"""Streamlit Cloud entry point for ftune web UI.

Deploy to Streamlit Community Cloud:
1. Push this repo to GitHub
2. Go to share.streamlit.io
3. Point to this file: streamlit_app.py
4. Set requirements file: requirements-web.txt
"""

import sys
from pathlib import Path

# Add src to path so ftune is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Run the app (Streamlit executes this file top-to-bottom)
exec(open(Path(__file__).parent / "src" / "ftune" / "app.py").read())
