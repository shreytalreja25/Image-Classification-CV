#!/bin/bash

echo "🧪 COMP9517 CV Group Project - Virtual Environment Setup"
echo "--------------------------------------------------------"
echo "Choose your OS:"
echo "1. Windows"
echo "2. macOS/Linux"
read -p "Enter 1 or 2: " os_choice

if [ "$os_choice" == "1" ]; then
    echo "Setting up virtual environment for Windows..."
    python -m venv venv
    echo "✅ Virtual environment created: venv"
    echo "To activate it, run:"
    echo "    .\\venv\\Scripts\\activate"
elif [ "$os_choice" == "2" ]; then
    echo "Setting up virtual environment for macOS/Linux..."
    python3 -m venv venv
    echo "✅ Virtual environment created: venv"
    echo "To activate it, run:"
    echo "    source venv/bin/activate"
else
    echo "❌ Invalid choice. Please enter 1 or 2."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies from requirements.txt..."
if [ "$os_choice" == "1" ]; then
    .\\venv\\Scripts\\activate && pip install -r requirements.txt
else
    source venv/bin/activate && pip install -r requirements.txt
fi

echo "✅ All dependencies installed."
echo "📌 Tip: Run 'deactivate' to exit the virtual environment when you're done."
