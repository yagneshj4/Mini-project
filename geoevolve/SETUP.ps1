# GeoEvolve Setup for Windows PowerShell

Write-Host "=================================================="
Write-Host "GeoEvolve Setup (Windows)" -ForegroundColor Cyan
Write-Host "=================================================="
Write-Host ""

# Step 1: Check Python
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Install Python 3.11+ from https://python.org" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python OK" -ForegroundColor Green
Write-Host ""

# Step 2: Virtual environment
Write-Host "[2/4] Setting up virtual environment..." -ForegroundColor Yellow
if (!(Test-Path "venv")) {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}
Write-Host ""

# Step 3: Activate venv
Write-Host "[3/4] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "✓ Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Step 4: Install dependencies
Write-Host "[4/4] Installing dependencies..." -ForegroundColor Yellow
pip install -q -r requirements.txt
Write-Host "✓ Dependencies installed" -ForegroundColor Green
Write-Host ""

# Step 5: Configure API key
Write-Host "=================================================="
Write-Host "CONFIGURE OPENAI API KEY" -ForegroundColor Cyan
Write-Host "=================================================="
Write-Host ""
Write-Host "Get your API key from: https://platform.openai.com/api-keys"
Write-Host ""
Write-Host "Option 1: Edit .env file manually (recommended)"
Write-Host "  1. Open: .env"
Write-Host "  2. Replace: your_api_key_here"
Write-Host "  3. Save file"
Write-Host ""

$useManual = Read-Host "Continue with manual setup? (Y/n)"
if ($useManual -eq "n" -or $useManual -eq "N") {
    $apiKey = Read-Host "Enter your OpenAI API key"
    @"
OPENAI_API_KEY=$apiKey
"@ | Out-File -Encoding utf8 .env
    Write-Host "✓ API key saved to .env" -ForegroundColor Green
}

Write-Host ""
Write-Host "=================================================="
Write-Host "✓ Setup Complete!" -ForegroundColor Green
Write-Host "=================================================="
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Test core (no LLM):      python test_core.py"
Write-Host "  2. Test MVP (with LLM):     python test_mvp.py"
Write-Host "  3. Run full evolution:      python main.py"
Write-Host ""
Write-Host "Output: results/" -ForegroundColor Yellow
