#!/bin/bash
# GeoEvolve Setup Guide for Windows PowerShell

echo "=================================================="
echo "GeoEvolve Setup"
echo "=================================================="
echo ""

# Step 1: Check Python
echo "[1/5] Checking Python installation..."
python --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python not found. Install Python 3.11+ from python.org"
    exit 1
fi
echo "✓ Python OK"
echo ""

# Step 2: Virtual environment
echo "[2/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Step 3: Activate venv
echo "[3/5] Activating virtual environment..."
echo "Run this command in your terminal:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "  venv\Scripts\activate"
else
    echo "  source venv/bin/activate"
fi
echo ""
echo "Then come back and run this script again."
echo ""
exit 0

# Step 4: Install dependencies
echo "[4/5] Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Step 5: Configure API key
echo "[5/5] Configuring OpenAI API key..."
echo "Get your API key from: https://platform.openai.com/api-keys"
echo ""
read -p "Enter your OpenAI API key: " api_key

cat > .env << EOF
OPENAI_API_KEY=$api_key
EOF

echo "✓ API key saved to .env"
echo ""

echo "=================================================="
echo "✓ Setup Complete!"
echo "=================================================="
echo ""
echo "Quick start:"
echo "  1. Test core (no LLM): python test_core.py"
echo "  2. Test MVP (with LLM): python test_mvp.py"
echo "  3. Run full evolution: python main.py"
echo ""
echo "Output will be saved to results/"
