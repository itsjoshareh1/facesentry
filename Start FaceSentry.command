#!/bin/bash
# Double-click this file to start FaceSentry. It sets everything up the first time.
cd "$(dirname "$0")" || exit 1

echo "Starting FaceSentry..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Opening the download page."
  echo "Install it, then double-click this file again."
  open "https://www.python.org/downloads/" 2>/dev/null
  read -n 1 -s -r -p "Press any key to close."
  exit 1
fi

# First run only: make a private Python environment and install what the app needs.
if [ ! -x ".venv/bin/python" ]; then
  echo "First-time setup (about a minute)..."
  python3 -m venv .venv || { echo "Could not create the environment."; read -n 1 -s -r -p "Press any key to close."; exit 1; }
fi
if [ ! -f ".venv/.installed" ]; then
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r requirements.txt || { echo "Install failed. Check your internet and try again."; read -n 1 -s -r -p "Press any key to close."; exit 1; }
  touch .venv/.installed
fi

if [ -n "$FACESENTRY_SETUP_ONLY" ]; then
  echo "Setup finished."
  exit 0
fi

[ -f model.yml ] || echo "First run: it takes 20 photos of your face first, then starts watching."
echo "FaceSentry is starting. A camera window opens. Press q in it to quit."
exec .venv/bin/python watch.py
