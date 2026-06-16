"""
Entry-point shim for the PyInstaller bundle.
Keeps the repo root clean; the spec file points here.
"""
import sys
from model.app import main

if __name__ == "__main__":
    sys.exit(main())
