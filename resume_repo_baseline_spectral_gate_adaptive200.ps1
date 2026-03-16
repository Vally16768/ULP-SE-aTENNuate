$ErrorActionPreference = "Stop"

$py = ".\.venv-audit\Scripts\python.exe"
$stdout = "runs\repo_baseline_spectral_gate_adaptive200.resume.stdout.log"
$stderr = "runs\repo_baseline_spectral_gate_adaptive200.resume.stderr.log"
$pidFile = "runs\repo_baseline_spectral_gate_adaptive200.pid"
$resume = "runs\repo_baseline_spectral_gate_adaptive200\last_train_state.pt"

if (-not (Test-Path $resume)) {
    throw "Resume checkpoint not found: $resume"
}

if (Test-Path $pidFile) {
    Remove-Item $pidFile -Force
}

$proc = Start-Process -FilePath $py `
    -ArgumentList @(
        "run_repo_baseline_spectral_gate_ft.py",
        "--config",
        "experiments\repo_baseline_spectral_gate_adaptive200.toml",
        "--device",
        "cuda",
        "--threshold-pesq",
        "2.4258807363730033",
        "--resume",
        $resume
    ) `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -WindowStyle Hidden

Set-Content -Path $pidFile -Value $proc.Id
Write-Output "PID=$($proc.Id)"
Write-Output "STDOUT=$stdout"
Write-Output "STDERR=$stderr"
Write-Output "PIDFILE=$pidFile"
