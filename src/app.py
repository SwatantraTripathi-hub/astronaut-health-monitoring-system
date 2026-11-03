"""Canonical Streamlit entry point.

The dashboard implementation remains import-compatible with the historical
``app/app.py`` path while callers migrate to ``python -m streamlit run src/app.py``.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "app" / "app.py"), run_name="__main__")
