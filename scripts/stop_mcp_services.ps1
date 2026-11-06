<#
.SYNOPSIS
    停止由 start_mcp_services.ps1 启动的两个常驻 MCP 服务。

.DESCRIPTION
    只停止启动脚本记录在 logs\mcp\services.json 里的 PID，并且会先核对进程
    命令行确实是本项目的服务模块，避免 PID 被复用后误杀其他进程。

    停止服务会终止其中正在运行的水文任务；任务记录与已完成阶段仍留在磁盘上，
    下次启动服务时会被标记为 interrupted。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\stop_mcp_services.ps1
#>
[CmdletBinding()]
param(
    [switch]$KeepJobs
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $ProjectRoot "logs\mcp\services.json"

if (-not (Test-Path $PidFile)) {
    Write-Host "没有服务记录（$PidFile）；没有需要停止的服务。"
    exit 0
}

$record = Get-Content -Raw $PidFile | ConvertFrom-Json

function Stop-Recorded {
    param([string]$Name, [string]$Module)
    $entry = $record.$Name
    if (-not $entry) { return }
    $processId = $entry.pid
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if (-not $process) {
        Write-Host "[$Name] pid $processId 已不存在。"
        return
    }
    $commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $processId").CommandLine
    if ($commandLine -notlike "*$Module*") {
        Write-Warning "[$Name] pid $processId 的命令行不像本项目的服务（$Module）；已跳过，不会结束它。"
        return
    }
    Stop-Process -Id $processId
    Write-Host "[$Name] 已停止 pid $processId。"
}

Stop-Recorded -Name "spatial" -Module "chongqing_spatial_mcp.server"
Stop-Recorded -Name "hydro" -Module "watershed_hydro_mcp.server"

if (-not $KeepJobs) {
    Remove-Item -Force $PidFile
    Write-Host "已删除服务记录。"
} else {
    Write-Host "保留了服务记录（-KeepJobs）。"
}

Write-Host ""
Write-Host "注意：停止服务会终止其中正在运行的水文任务。任务记录仍在 runs\hydro\jobs\ 下，"
Write-Host "下次启动服务时会根据心跳判定为 interrupted，可基于原输入重新提交。"
