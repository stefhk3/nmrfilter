#!/bin/bash

# Get the directory path where the shell script is located
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if the virtual environment already exists in the script's directory
if [ -f "$script_dir/nmrfilter_env/bin/activate" ]; then
    echo "NMRfilter_env already exists. Activating..."
    source "$script_dir/nmrfilter_env/bin/activate"
else
    echo "Creating virtual environment in $script_dir/nmrfilter_env..."
    python3 -m venv "$script_dir/nmrfilter_env"
    source "$script_dir/nmrfilter_env/bin/activate"

    echo "Installing the requirements..."
    python -m pip install -r requirements.txt

    echo "Installing Jupyter Notebook..."
    python -m pip install jupyter
fi

# Start Jupyter Notebook
jupyter notebook

# You may add 'read' command here to pause the script before exiting
# read -p "Press Enter to exit..."
