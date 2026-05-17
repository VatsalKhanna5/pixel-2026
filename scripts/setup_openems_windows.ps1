#!/usr/bin/env pwsh
# =============================================================================
#  scripts/setup_openems_windows.ps1
#  PIXEL-2026  —  OpenEMS Windows Installation Helper
#
#  OpenEMS is the open-source FDTD EM solver used for dataset generation.
#  On Windows it must be installed from the official release ZIP.
#
#  USAGE (run as Administrator for PATH modification):
#    .\scripts\setup_openems_windows.ps1
#
#  What this does:
#    1. Downloads the latest OpenEMS Windows ZIP from GitHub releases
#    2. Extracts to D:\openEMS (configurable via $OPENEMS_DIR)
#    3. Adds D:\openEMS to the system PATH
#    4. Adds the Python bindings to pixel-env's PYTHONPATH
#    5. Validates a minimal OpenEMS Python import
#
#  Manual fallback if this script fails:
#    a) Go to: https://github.com/thliebig/openEMS-Project/releases
#    b) Download openEMS_v0.0.36_win64.zip (or latest)
#    c) Extract to D:\openEMS
#    d) Add D:\openEMS to PATH in System Properties > Environment Variables
#    e) In pixel-env:  pip install CSXCAD openEMS  (Python binding stubs)
#    f) Validate:  python -c "from openems import openEMS; print('OK')"
#
#  Date:  May 2026
# =============================================================================

Set-StrictMode -Version Latest

$OPENEMS_DIR = "D:\openEMS"
$ROOT = Split-Path $PSScriptRoot -Parent
$LOG = "$ROOT\logs\setup_openems.log"

function Log {
    param([string]$msg, [string]$color = "Cyan")
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $LOG -Value $line
}

Log "======================================================" "Magenta"
Log " PIXEL-2026  OpenEMS Windows Setup"                    "Magenta"
Log "======================================================" "Magenta"

# Check if openEMS is already installed
if (Test-Path "$OPENEMS_DIR\openEMS.exe") {
    Log "openEMS already found at $OPENEMS_DIR — skipping download." "Green"
} else {
    Log "OpenEMS not found. Attempting download from GitHub..." "Yellow"

    # Latest release asset URL — update version number here if needed
    $RELEASE_URL = "https://github.com/thliebig/openEMS-Project/releases/download/v0.0.36/openEMS_v0.0.36_win64.zip"
    $ZIP_PATH = "$env:TEMP\openEMS_win64.zip"

    try {
        Log "Downloading: $RELEASE_URL" "Yellow"
        Invoke-WebRequest -Uri $RELEASE_URL -OutFile $ZIP_PATH -UseBasicParsing
        Log "Download complete. Extracting to $OPENEMS_DIR ..." "Yellow"
        New-Item -ItemType Directory -Path $OPENEMS_DIR -Force | Out-Null
        Expand-Archive -Path $ZIP_PATH -DestinationPath $OPENEMS_DIR -Force
        Log "Extraction complete." "Green"
    } catch {
        Log "Automatic download FAILED: $_" "Red"
        Log "" "White"
        Log "MANUAL INSTALLATION REQUIRED:" "Yellow"
        Log "  1. Open browser: https://github.com/thliebig/openEMS-Project/releases" "White"
        Log "  2. Download the latest win64 ZIP" "White"
        Log "  3. Extract to $OPENEMS_DIR" "White"
        Log "  4. Re-run this script" "White"
        Log "" "White"
        Log "Continuing with Python binding install (may fail without binaries)..." "Yellow"
    }
}

# Add to PATH if not already present
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
if ($currentPath -notlike "*$OPENEMS_DIR*") {
    Log "Adding $OPENEMS_DIR to system PATH..." "Yellow"
    [System.Environment]::SetEnvironmentVariable(
        "PATH",
        "$currentPath;$OPENEMS_DIR",
        "Machine"
    )
    $env:PATH += ";$OPENEMS_DIR"
    Log "PATH updated." "Green"
} else {
    Log "$OPENEMS_DIR already in PATH." "DarkGray"
}

# Install Python binding stubs into pixel-env
Log "Installing Python bindings (CSXCAD, openEMS)..." "Yellow"
conda run -n pixel-env pip install CSXCAD openEMS 2>&1 | Tee-Object -Append -FilePath $LOG

# Validate
Log "Validating Python bridge..." "Cyan"
$test = conda run -n pixel-env python -c "from openems import openEMS; print('openEMS Python bridge: OK')" 2>&1
$test | ForEach-Object { Log $_ "White" }

if ($test -match "OK") {
    Log "openEMS Python bridge: VALIDATED" "Green"
} else {
    Log "openEMS Python bridge: NOT working — will use subprocess fallback in dataset generation." "Yellow"
    Log "See src/dataset/openems_runner.py for subprocess fallback implementation." "Yellow"
}

Log "======================================================" "Magenta"
Log " OpenEMS setup complete."                              "Green"
Log "======================================================" "Magenta"
