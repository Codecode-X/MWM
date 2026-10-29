"""Compatibility entry point for the corrected paper experiment script."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
environment = {**os.environ, 'PYTHON': sys.executable}
raise SystemExit(subprocess.call(['bash', str(root / 'scripts/run_paper_experiments.sh'), *sys.argv[1:]],
                                 cwd=root, env=environment))
