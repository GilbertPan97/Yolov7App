#!/bin/bash

# Define the project directory relative to the script location and the name of the virtual environment
project_dir="./"
venv_name="venv"

# Path to the requirements file
requirements_path="$project_dir/requirements.txt"

# Navigate to the project directory
cd "$(dirname "$0")"

# Check if the virtual environment already exists
if [ -d "$project_dir/$venv_name" ]; then
    echo "The virtual environment $venv_name already exists in $project_dir."
else
    # Create the virtual environment
    echo "Creating virtual environment $venv_name in $project_dir..."
    python3 -m venv "$project_dir/$venv_name"
    
    if [ $? -eq 0 ]; then
        echo "Virtual environment $venv_name created successfully in $project_dir."
    else
        echo "Failed to create the virtual environment in $project_dir."
        exit 1
    fi
fi

# Activate the virtual environment
echo "Activating virtual environment $venv_name..."
source "$project_dir/$venv_name/bin/activate"

# Function to check if a package is installed
is_package_installed() {
    pip freeze | grep -i "^$1==" >/dev/null
    return $?
}

# Check and install requirements
if [ -f "$requirements_path" ]; then
    echo "Checking installed packages..."
    
    all_installed=true
    
    while IFS= read -r requirement || [[ -n "$requirement" ]]; do
        package=$(echo $requirement | cut -d'=' -f1)
        is_package_installed $package
        if [ $? -ne 0 ]; then
            all_installed=false
            echo "$package is not installed."
            break
        fi
    done < "$requirements_path"
    
    if [ "$all_installed" = false ]; then
        echo "Installing dependencies from $requirements_path..."
        pip install -r "$requirements_path"
        
        if [ $? -eq 0 ]; then
            echo "Dependencies installed successfully."
        else
            echo "Failed to install some dependencies."
            exit 1
        fi
    else
        echo "All dependencies are already installed."
    fi
else
    echo "Requirements file not found at $requirements_path."
    exit 1
fi

echo "Setup complete. Your virtual environment is ready."

