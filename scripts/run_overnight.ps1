#Requires -Version 5.1
# Runs the full classical pipeline end-to-end with timestamped logging.
# Each stage logs to logs/ and aborts the chain on failure.

$ErrorActionPreference = "Continue"  # not Stop: we handle exit codes ourselves
$logDir = "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"

function Run-Stage {
    param([string]$name, [string]$script)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  STAGE: $name" -ForegroundColor Cyan
    Write-Host "  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Cyan
    $log = "$logDir/${ts}_${name}.log"
    & uv run python -u $script 2>&1 | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    Write-Host ""
    if ($code -ne 0) {
        Write-Host "[FAIL] $name exit=$code  (log: $log)" -ForegroundColor Red
        return $code
    }
    Write-Host "[OK] $name finished at $(Get-Date -Format 'HH:mm:ss')  (log: $log)" -ForegroundColor Green
    return 0
}

Write-Host "===== OVERNIGHT classical pipeline =====" -ForegroundColor Magenta
Write-Host "Logs dir: $logDir" -ForegroundColor DarkGray
$startedAt = Get-Date

$code = Run-Stage -name "train" -script "scripts/run_classical_train.py"
if ($code -ne 0) { exit $code }

$code = Run-Stage -name "hardneg" -script "scripts/run_classical_hard_neg.py"
if ($code -ne 0) { exit $code }

$code = Run-Stage -name "infer" -script "scripts/run_classical_infer.py"
if ($code -ne 0) { exit $code }

$elapsed = (Get-Date) - $startedAt
Write-Host ""
Write-Host "===== ALL DONE =====" -ForegroundColor Green
Write-Host "Total elapsed: $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host "Predictions: reports/predictions/classical_test.json" -ForegroundColor Green
