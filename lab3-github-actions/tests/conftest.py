"""Make the lab folder importable: tests import data.py and train.py from it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
