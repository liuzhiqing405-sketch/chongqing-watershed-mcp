# Live understanding acceptance. No real MCP services or computations required.
[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$AcceptanceRoot = Split-Path -Parent $PSScriptRoot
$AcceptancePython = Join-Path $AcceptanceRoot ".venv\Scripts\python.exe"
$AcceptanceScript = Join-Path $PSScriptRoot "accept_model_understanding.py"
$AcceptanceOldKey = $env:SLOPE_AGENT_LLM_API_KEY
$AcceptanceSecureKey = $null
$AcceptanceCredential = $null
try {
    if ([string]::IsNullOrWhiteSpace($AcceptanceOldKey)) {
        $AcceptanceSecureKey = Read-Host "DeepSeek API Key (masked; not saved)" -AsSecureString
        $AcceptanceCredential = New-Object System.Net.NetworkCredential("", $AcceptanceSecureKey)
        $env:SLOPE_AGENT_LLM_API_KEY = $AcceptanceCredential.Password
    }
    if ([string]::IsNullOrWhiteSpace($env:SLOPE_AGENT_LLM_API_KEY)) {
        throw "DeepSeek API Key cannot be empty."
    }
    & $AcceptancePython $AcceptanceScript
    $AcceptanceExit = $LASTEXITCODE
    Write-Host "Report: $AcceptanceRoot\runs\model_acceptance\latest\report.md"
    exit $AcceptanceExit
}
finally {
    if ($null -eq $AcceptanceOldKey) {
        Remove-Item Env:SLOPE_AGENT_LLM_API_KEY -ErrorAction SilentlyContinue
    } else { $env:SLOPE_AGENT_LLM_API_KEY = $AcceptanceOldKey }
    $AcceptanceOldKey = $null
    $AcceptanceSecureKey = $null
    $AcceptanceCredential = $null
}
