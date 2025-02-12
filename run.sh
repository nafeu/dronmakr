#!/bin/bash

# Detect virtual environment paths
VENV_DIRS=(
    "./venv"
    "$HOME/.virtualenvs/dronmakr"
)

# Find a valid virtual environment
for VENV in "${VENV_DIRS[@]}"; do
    if [ -d "$VENV" ]; then
        source "$VENV/bin/activate"
        python dronmakr.py "$@"  # <-- Pass all script arguments to Python
        deactivate
        exit 0
    fi
done

echo "No virtual environment found! Exiting."
exit 1
