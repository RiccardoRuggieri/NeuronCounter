#!/bin/bash
# Double-click this file (macOS Finder) to open the Neuron Counter GUI.
# It uses the project's virtual environment if present.
cd "$(dirname "$0")"
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi
exec python -m neuron_counter.app
