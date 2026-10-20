"""Convenience launcher: python main.py."""
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    raise SystemExit(subprocess.call([sys.executable, '-m', 'streamlit', 'run', str(Path(__file__).with_name('app.py'))]))
