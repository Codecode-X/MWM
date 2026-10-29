"""Inspect saved paper model metadata, and reevaluate its held-out split."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import main

if __name__ == '__main__':
    main()
