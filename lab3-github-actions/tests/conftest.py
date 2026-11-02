import sys
from pathlib import Path

# Make the lab package importable when pytest runs from the repo root
# (CI) or from this folder (local runs).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
