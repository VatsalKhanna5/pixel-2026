# scripts/run_phase1.ps1
# PIXEL-2026 Phase 1 Dataset Generation Launcher
#
# Usage:
#   .\scripts\run_phase1.ps1                        # full run
#   .\scripts\run_phase1.ps1 -PilotOnly             # pilot validation only
#   .\scripts\run_phase1.ps1 -Resume                # resume from checkpoint
#   .\scripts\run_phase1.ps1 -Workers 32            # override worker count
#   .\scripts\run_phase1.ps1 -NSamples 100000       # override target count

param(
    [switch]$PilotOnly    = $false,
    [switch]$Resume       = $false,
    [switch]$SkipPilot    = $false,
    [int]   $Workers      = 56,
    [int]   $NSamples     = 200000,
    [string]$Config       = "experiments/configs/base_config.yaml",
    [string]$Output       = "data/raw/pixel_dataset.h5"
)

$ErrorActionPreference = "Stop"

# ── Resolve project root ─────────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  PIXEL-2026  Phase 1 Dataset Generation" -ForegroundColor Cyan
Write-Host "  Project root : $ProjectRoot" -ForegroundColor Cyan
Write-Host "  Workers      : $Workers" -ForegroundColor Cyan
Write-Host "  Target       : $NSamples samples" -ForegroundColor Cyan
Write-Host "  Output       : $Output" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# ── Activate conda environment ────────────────────────────────────────────────
$CondaBase  = "$env:USERPROFILE\anaconda3"
$EnvPath    = "$CondaBase\envs\pixel-env"
$PythonExe  = "$EnvPath\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "pixel-env Python not found at $PythonExe. Run Phase 0 setup first."
    exit 1
}

Write-Host "[run_phase1] Python: $PythonExe" -ForegroundColor Green

# ── Ensure output directory exists ───────────────────────────────────────────
$OutputDir = Split-Path -Parent $Output
if ($OutputDir -and -not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Write-Host "[run_phase1] Created output directory: $OutputDir" -ForegroundColor Green
}

# ── Build argument list ───────────────────────────────────────────────────────
$Args_ = @(
    "-m", "src.dataset.generate",
    "--config",    $Config,
    "--n-samples", $NSamples,
    "--workers",   $Workers,
    "--output",    $Output
)

if ($PilotOnly)  { $Args_ += "--pilot"       }
if ($Resume)     { $Args_ += "--resume"      }
if ($SkipPilot)  { $Args_ += "--skip-pilot"  }

# ── Log file ─────────────────────────────────────────────────────────────────
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir    = "experiments\logs\phase1"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }
$LogFile   = "$LogDir\generate_$Timestamp.log"

Write-Host "[run_phase1] Log: $LogFile" -ForegroundColor Green
Write-Host "[run_phase1] Starting …" -ForegroundColor Green
Write-Host ""

# ── Run ───────────────────────────────────────────────────────────────────────
$StartTime = Get-Date
try {
    & $PythonExe @Args_ 2>&1 | Tee-Object -FilePath $LogFile
    $ExitCode = $LASTEXITCODE
} catch {
    Write-Error "Process failed: $_"
    $ExitCode = 1
}

$Elapsed = (Get-Date) - $StartTime
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
if ($ExitCode -eq 0) {
    Write-Host "  COMPLETED SUCCESSFULLY  (elapsed: $($Elapsed.ToString('hh\:mm\:ss')))" -ForegroundColor Green
} else {
    Write-Host "  FAILED (exit code $ExitCode, elapsed: $($Elapsed.ToString('hh\:mm\:ss')))" -ForegroundColor Red
}
Write-Host "================================================================" -ForegroundColor Cyan

# ── Post-generation audit (skip in pilot-only mode) ──────────────────────────
if ($ExitCode -eq 0 -and -not $PilotOnly) {
    Write-Host ""
    Write-Host "[run_phase1] Running post-generation integrity audit …" -ForegroundColor Yellow
    & $PythonExe "scripts\audit_dataset.py" --h5 $Output 2>&1 | Tee-Object -Append -FilePath $LogFile
    $AuditCode = $LASTEXITCODE
    if ($AuditCode -ne 0) {
        Write-Host "[run_phase1] Audit detected issues. Check $LogFile for details." -ForegroundColor Red
    } else {
        Write-Host "[run_phase1] Audit PASSED." -ForegroundColor Green
    }
}

exit $ExitCode
