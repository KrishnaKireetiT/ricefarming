#!/bin/bash
# Quick setup script for the ricefarming app fixes

echo "==================================="
echo "Rice Farming App - Setup Fixes"
echo "==================================="

cd /mnt/node5_tpu_data_code_1/krishna_home/ricefarming/StreamlitApp

echo ""
echo "Step 1: Installing new dependency..."
pip install extra-streamlit-components>=0.1.60

echo ""
echo "Step 2: Verifying files..."
if [ -f "session_utils.py" ]; then
    echo "✓ session_utils.py created"
else
    echo "✗ session_utils.py missing"
fi

if [ -f "app.py" ]; then
    echo "✓ app.py updated"
else
    echo "✗ app.py missing"
fi

if grep -q "extra-streamlit-components" requirements.txt; then
    echo "✓ requirements.txt updated"
else
    echo "✗ requirements.txt not updated"
fi

echo ""
echo "==================================="
echo "Setup complete! You can now run:"
echo "  streamlit run app.py"
echo "==================================="
