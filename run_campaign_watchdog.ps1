param(
    [string]$RepoRoot = "c:\Users\E1554695\Desktop\projects\ULP-SE-aTENNuate",
    [string]$PythonExe = ".\.venv-audit\Scripts\python.exe",
    [string]$ConfigPath = "experiments\pesq_campaign.toml",
    [string]$Device = "cuda",
    [int]$RestartDelaySeconds = 15
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

$pythonPath = if ([System.IO.Path]::IsPathRooted($PythonExe)) { $PythonExe } else { Join-Path $RepoRoot $PythonExe }
$configPathAbs = if ([System.IO.Path]::IsPathRooted($ConfigPath)) { $ConfigPath } else { Join-Path $RepoRoot $ConfigPath }
$entrypoint = Join-Path $RepoRoot "run_campaign.py"
$stdoutLog = Join-Path $RepoRoot "runs\pesq_campaign.launch.stdout.log"
$stderrLog = Join-Path $RepoRoot "runs\pesq_campaign.launch.stderr.log"
$watchdogLog = Join-Path $RepoRoot "runs\pesq_campaign.watchdog.log"
$summaryPath = Join-Path $RepoRoot "runs\pesq_campaign\campaign_summary.json"

New-Item -ItemType Directory -Force -Path "runs" | Out-Null
Add-Content $watchdogLog "[$(Get-Date -Format o)] watchdog_start"

try {
    while (-not (Test-Path $summaryPath)) {
        $cmdLine = "`"$pythonPath`" `"$entrypoint`" --config `"$configPathAbs`" --device $Device 1>> `"$stdoutLog`" 2>> `"$stderrLog`""
        $proc = Start-Process -FilePath "cmd.exe" -ArgumentList @("/d", "/c", $cmdLine) -WorkingDirectory $RepoRoot -PassThru -Wait
        $exitCode = $proc.ExitCode
        Add-Content $watchdogLog "[$(Get-Date -Format o)] run_exit=$exitCode"
        if ($exitCode -eq 0) {
            break
        }
        Start-Sleep -Seconds $RestartDelaySeconds
    }
} catch {
    Add-Content $watchdogLog "[$(Get-Date -Format o)] watchdog_error=$($_.Exception.Message)"
    throw
} finally {
    Add-Content $watchdogLog "[$(Get-Date -Format o)] watchdog_stop"
}
