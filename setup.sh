#!/bin/bash

# Brain Hemorrhage Detection System - macOS/Linux Setup Script
# This script will install Python and all required dependencies

echo ""
echo "================================================================================"
echo "Brain Hemorrhage Detection System - Setup"
echo "================================================================================"
echo ""

# Check if Python is installed
echo "Checking for Python installation..."
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "ERROR: Python 3 is not installed"
    echo ""
    echo "Please install Python 3.8 or higher:"
    echo ""
    echo "On macOS (using Homebrew):"
    echo "  brew install python@3.11"
    echo ""
    echo "On Ubuntu/Debian:"
    echo "  sudo apt-get install python3.11 python3.11-venv python3.11-dev"
    echo ""
    echo "On Fedora/RHEL:"
    echo "  sudo dnf install python3.11 python3.11-devel"
    echo ""
    echo "After installing Python, run this script again."
    exit 1
fi

echo "✓ Python found!"
python3 --version
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi
echo "✓ Virtual environment created!"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated!"
echo ""

# Upgrade pip
echo "Installing/Upgrading pip..."
python -m pip install --upgrade pip --quiet
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to upgrade pip"
    exit 1
fi
echo "✓ Pip upgraded successfully!"
echo ""

# Install requirements
echo "Installing dependencies (this may take several minutes)..."
echo "This includes: TensorFlow, OpenCV, Flask, and more..."
echo ""
python -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements"
    exit 1
fi
echo "✓ Dependencies installed successfully!"
echo ""

# Create model
echo "Creating pre-trained model..."
python model/train_model.py
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create model"
    exit 1
fi
echo "✓ Model created successfully!"
echo ""

# Success
echo "================================================================================"
echo "Setup Complete!"
echo "================================================================================"
echo ""
echo "You can now run the application with:"
echo "  source venv/bin/activate  # Activate virtual environment"
echo "  python app.py"
echo ""
echo "Then open http://localhost:5000 in your web browser"
echo ""
