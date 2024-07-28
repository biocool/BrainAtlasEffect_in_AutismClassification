#!/bin/bash

check_python_version() {
    # Get the Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info[:3])' 2>/dev/null)

    # Extract major, minor, and micro versions
    MAJOR_VERSION=$(echo "$PYTHON_VERSION" | cut -d',' -f1 | cut -d'(' -f2)
    MINOR_VERSION=$(echo "$PYTHON_VERSION" | cut -d',' -f2 | tr -d ' ')
    MICRO_VERSION=$(echo "$PYTHON_VERSION" | cut -d',' -f3 | tr -d ' ')

    # Check if Python version is 3.7 or higher
    if [ "$MAJOR_VERSION" -lt 3 ] || { [ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 7 ]; }; then
        echo "Python 3.7 or higher is required. Current version: $MAJOR_VERSION.$MINOR_VERSION.$MICRO_VERSION"
        exit 1
    else
        echo "Python version is $MAJOR_VERSION.$MINOR_VERSION.$MICRO_VERSION. Proceeding with package installation."
    fi
}

install_python_packages() {
    echo "Installing Python packages..."

    # Upgrade pip to the latest version
    python3 -m pip install --upgrade pip

    # Install required packages
    python3 -m pip install mne pandas statsmodels scikit-learn nilearn numpy seaborn matplotlib xlrd plotly pdfkit

    # Install kaleido with a specific version
    python3 -m pip install kaleido==0.1.0.post1

    echo "All packages have been installed successfully."
}


install_system_dependencies() {
    echo "Installing system dependencies for pdfkit..."

    # Update package lists
    sudo apt update

    # Install wkhtmltopdf
    sudo apt install -y wkhtmltopdf

    echo "System dependencies installed successfully."
}


download_processed_datasets_from_ABIDE() {
    echo "Downloading processed datasets from ABIDE..."

    python Download_from_abide.py

    echo "The download of processed datasets has been completed successfully."
}

# Main script execution
check_python_version
install_system_dependencies
install_python_packages
download_processed_datasets_from_ABIDE

