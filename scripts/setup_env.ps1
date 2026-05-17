#!/usr/bin/env pwsh
# =============================================================================
#  scripts/setup_env.ps1
#  PIXEL-2026  —  Complete Environment Setup Script for Windows
#
#  USAGE (from project root, with conda in PATH):
#    .\scripts\setup_env.ps1
#
#  What this script does:
#    1. Creates the pixel-env conda environment (Python 3.13)
#    2. Installs PyTorch 2.x with CUDA 12.8
#    3. Installs all remaining pip dependencies
#    4. Installs this package in editable mode (src/ importable)
#    5. Registers the kernel with Jupyter
#    6. Validates the installation
#
#  Logs:  All output is tee'd to logs/setup_env_<timestamp>.log
#  Date:  May 2026
# =============================================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = Split-Path $PSScriptRoot -Parent
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = "$ROOT\logs\setup_env_$TIMESTAMP.log"

function Log {
    param([string]$msg, [string]$color = "Cyan")
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $LOG -Value $line
}

function Run {
    param([string]$cmd, [string]$desc)
    Log ">>> $desc" "Yellow"
    Log "    CMD: $cmd" "DarkGray"
    $result = Invoke-Expression $cmd 2>&1
    $result | ForEach-Object { Add-Content -Path $LOG -Value $_ }
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne $null) {
        Log "FAILED (exit $LASTEXITCODE): $desc" "Red"
        $result | Select-Object -Last 20 | ForEach-Object { Log "  $_" "Red" }
        throw "Setup step failed: $desc"
    }
    Log "    OK" "Green"
}

New-Item -ItemType Directory -Path "$ROOT\logs" -Force | Out-Null
Log "======================================================" "Magenta"
Log " PIXEL-2026 Environment Setup  |  $TIMESTAMP"         "Magenta"
Log " Log file: $LOG"                                        "Magenta"
Log "======================================================" "Magenta"

# --------------------------------------------------------------------
# 1. Create conda environment (idempotent — skip if already exists)
# --------------------------------------------------------------------
$envExists = (conda env list) -match "pixel-env"
if ($envExists) {
    Log "pixel-env already exists, skipping creation." "DarkGray"
} else {
    Run "conda create -n pixel-env python=3.13 -y" `
        "Creating pixel-env conda environment"
}

# --------------------------------------------------------------------
# 2. PyTorch with CUDA 12.8
# --------------------------------------------------------------------
Run "conda run -n pixel-env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128" `
    "Installing PyTorch with CUDA 12.8"

# --------------------------------------------------------------------
# 3. Remaining pip dependencies
# --------------------------------------------------------------------
Run "conda run -n pixel-env pip install -r `"$ROOT\requirements.txt`"" `
    "Installing all pip requirements"

# --------------------------------------------------------------------
# 4. Editable install of this package
# --------------------------------------------------------------------
Run "conda run -n pixel-env pip install -e `"$ROOT`" --no-deps" `
    "Installing pixel2026 package in editable mode"

# --------------------------------------------------------------------
# 5. Register Jupyter kernel
# --------------------------------------------------------------------
Run "conda run -n pixel-env python -m ipykernel install --user --name pixel-env --display-name 'Python (pixel-env)'" `
    "Registering Jupyter kernel"

# --------------------------------------------------------------------
# 6. Validate
# --------------------------------------------------------------------
Log "Running Phase 0 validation..." "Cyan"
conda run -n pixel-env python "$ROOT\scripts\validate_phase0.py" 2>&1 | Tee-Object -Append -FilePath $LOG

Log "======================================================" "Magenta"
Log " Setup complete!  Activate with:  conda activate pixel-env" "Green"
Log "======================================================" "Magenta"
