$ErrorActionPreference = "Stop"

$py = ".\.venv-audit\Scripts\python.exe"
$stdout = "runs\repo_baseline_spectral_gate_ft_full_eval.launch.stdout.log"
$stderr = "runs\repo_baseline_spectral_gate_ft_full_eval.launch.stderr.log"
$pidFile = "runs\repo_baseline_spectral_gate_ft_full_eval.pid"

if (Test-Path $pidFile) {
    Remove-Item $pidFile -Force
}

$proc = Start-Process -FilePath $py `
    -ArgumentList @(
        "run_repo_baseline_spectral_gate_ft.py",
        "--config",
        "experiments\repo_baseline_spectral_gate_ft.toml",
        "--device",
        "cuda"
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
