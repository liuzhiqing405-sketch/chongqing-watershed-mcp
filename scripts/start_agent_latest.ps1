<#
.SYNOPSIS
    Reloads idle, verified project MCP services and starts the current WebUI.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\start_agent_latest.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\start_agent_latest.ps1 -CheckOnly
#>
[CmdletBinding()]
param(
    [string]$Model = $env:SLOPE_AGENT_LLM_MODEL,
    [string]$BaseUrl = $env:SLOPE_AGENT_LLM_BASE_URL,
    [string]$Provider = $env:SLOPE_AGENT_LLM_PROVIDER,
    [string]$ConfigPath = $env:SLOPE_AGENT_CONFIG,
    [switch]$ConfigureInWeb,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $taskRoot '.venv\Scripts\python.exe'
$recordPath = Join-Path $taskRoot 'logs\mcp\services.json'
$webLauncher = Join-Path $PSScriptRoot 'start_agent_webui.ps1'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python is missing' }

function Get-ListenerPid([int]$Port) {
    $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    if (-not ($listeners | Where-Object { $_.Port -eq $Port })) { return $null }
    $pattern = '^\s*TCP\s+\S+:' + $Port + '\s+\S+\s+LISTENING\s+(\d+)\s*$'
    $rows = @(netstat -ano -p TCP | Select-String $pattern |
        ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Select-Object -Unique)
    if ($rows.Count -ne 1) { throw "Cannot identify a unique listener on port $Port" }
    return $rows[0]
}

# The WebUI holds the model credential in its own process. Let its owner stop
# it normally instead of inspecting that process or recovering its environment.
if (Get-ListenerPid 8780) {
    throw 'WebUI port 8780 is occupied. Press Ctrl+C in the previous WebUI terminal, then run this launcher again.'
}

# Complete the whole preflight before stopping either service.
foreach ($tree in @(@('runs\hydro\jobs','status.json'), @('runs\hydro\visualizations','visualization.json'))) {
    $directory = Join-Path $taskRoot $tree[0]
    if (-not (Test-Path -LiteralPath $directory)) { continue }
    foreach ($item in Get-ChildItem -LiteralPath $directory -Directory) {
        $statusPath = Join-Path $item.FullName $tree[1]
        if (-not (Test-Path -LiteralPath $statusPath)) { continue }
        $status = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($status.status -in @('queued','running','cancelling','cancel_requested')) {
            throw "Active work prevents service reload: $($item.Name). Wait for completion or cancel it through the WebUI."
        }
    }
}
$record = if (Test-Path -LiteralPath $recordPath) {
    Get-Content -LiteralPath $recordPath -Raw -Encoding UTF8 | ConvertFrom-Json
} else { $null }
$stopPlan = @()
foreach ($service in @(
    @{Name='spatial'; Port=8765; Module='chongqing_spatial_mcp.server'},
    @{Name='hydro'; Port=8766; Module='watershed_hydro_mcp.server'}
)) {
    $ownerPid = Get-ListenerPid $service.Port
    if (-not $ownerPid) { continue } # Missing services will both be started below.
    $entry = if ($record) { $record.($service.Name) } else { $null }
    if (-not $entry -or [int]$entry.port -ne $service.Port) {
        throw "Port $($service.Port) has no matching project service record; no process stopped."
    }
    $parentId = [int]$entry.pid
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $parentId"
    $modulePattern = '-m\s+' + [regex]::Escape($service.Module) + '(?:\s|$)'
    if (-not $parent -or $parent.ExecutablePath -ne $pythonPath -or $parent.CommandLine -notmatch $modulePattern) {
        throw "Recorded $($service.Name) process no longer matches this project; no process stopped."
    }
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $parentId" |
        Where-Object { $_.CommandLine -match $modulePattern })
    $verifiedIds = @($parentId) + @($children | ForEach-Object { [int]$_.ProcessId })
    if ($ownerPid -notin $verifiedIds) { throw "Port $($service.Port) belongs to an unverified process; no process stopped." }
    # Keep the expected start time to catch PID reuse before stopping.
    $stopPlan += @($children) + @($parent)
}
if ($CheckOnly) {
    Write-Output "Preflight OK. Verified processes to reload: $($stopPlan.Count). No processes changed; no model key requested."
    return
}
foreach ($planned in $stopPlan) {
    $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($planned.ProcessId)"
    if (-not $current) { continue }
    if ($current.CreationDate -ne $planned.CreationDate -or $current.ExecutablePath -ne $planned.ExecutablePath) {
        throw 'Process identity changed during reload; stopped to avoid PID reuse.'
    }
    Stop-Process -Id $planned.ProcessId
}
$deadline = (Get-Date).AddSeconds(5)
while ((Get-ListenerPid 8765) -or (Get-ListenerPid 8766)) {
    if ((Get-Date) -ge $deadline) { throw 'MCP ports have not been released; no duplicate servers started.' }
    Start-Sleep -Milliseconds 200
}
Write-Host 'Starting current project services and WebUI. Existing conversations and scientific results are preserved.'
Push-Location $taskRoot
try {
    & $webLauncher -Model $Model -BaseUrl $BaseUrl -Provider $Provider -ConfigPath $ConfigPath -ConfigureInWeb:$ConfigureInWeb
    if (-not $?) { throw 'WebUI launcher failed; see the preceding diagnostic.' }
} finally { Pop-Location }
