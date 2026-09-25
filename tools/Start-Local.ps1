$ErrorActionPreference = 'Stop'
$projectDir = Split-Path $PSScriptRoot -Parent
Set-Location $projectDir
$pythonExe = Join-Path $projectDir '.venv\Scripts\python.exe'
$ollamaExe = Join-Path $projectDir '.runtime\ollama\ollama.exe'
if (-not (Test-Path $pythonExe)) { throw 'Crie .venv e instale requirements.txt conforme o README.' }
if (-not (Test-Path $ollamaExe)) {
    $ollamaExe = (Get-Command ollama -ErrorAction Stop).Source
}
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_MODELS = Join-Path $projectDir '.runtime\models'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_NO_CLOUD = '1'
New-Item -ItemType Directory -Force (Join-Path $projectDir '.runtime') | Out-Null
try { Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3 | Out-Null }
catch {
    Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $projectDir '.runtime\ollama.out.log') `
        -RedirectStandardError (Join-Path $projectDir '.runtime\ollama.err.log') | Out-Null
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 | Out-Null
            $ready = $true
            break
        } catch { }
    }
    if (-not $ready) { throw 'Ollama não iniciou. Consulte .runtime\ollama.err.log.' }
}
Write-Host 'ReCiclaí: http://127.0.0.1:8000 (Ctrl+C encerra o backend)'
& $pythonExe -m uvicorn app:app --host 127.0.0.1 --port 8000
