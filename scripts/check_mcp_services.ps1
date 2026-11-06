<#
.SYNOPSIS
    检查两个常驻 MCP 服务的健康状态。

.DESCRIPTION
    对每个服务报告：PID、地址、进程是否存活，以及一次真实的 MCP 握手结果与
    服务自报版本。只读，不会启动或结束任何进程。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\check_mcp_services.ps1
#>
[CmdletBinding()]
param(
    [string]$SpatialUrl = "http://127.0.0.1:8765/mcp",
    [string]$HydroUrl = "http://127.0.0.1:8766/mcp"
)

$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PidFile = Join-Path $ProjectRoot "logs\mcp\services.json"

if (-not (Test-Path $Python)) {
    Write-Error "找不到项目虚拟环境：$Python"
    exit 2
}

$record = $null
if (Test-Path $PidFile) {
    $record = Get-Content -Raw $PidFile | ConvertFrom-Json
}

function Show-Service {
    param([string]$Name, [string]$Url)
    Write-Host ""
    Write-Host "== $Name" -ForegroundColor Cyan
    $processId_ = $null
    if ($record -and $record.$Name) { $processId_ = $record.$Name.pid }
    if ($processId_) {
        $process = Get-Process -Id $processId_ -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host ("   PID      : {0}（{1}，启动于 {2}）" -f $processId_, $process.ProcessName, $process.StartTime)
        } else {
            Write-Host ("   PID      : {0} 已不存在" -f $processId_) -ForegroundColor Yellow
        }
    } else {
        Write-Host "   PID      : 记录中没有（服务可能由其他方式启动）" -ForegroundColor Yellow
    }
    Write-Host "   地址     : $Url"

    # A real MCP handshake, through the same client the agent uses.
    $probe = & $Python -c @"
import asyncio, json, sys
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

async def main():
    try:
        async with Client(streamable_http_client(r'$Url')) as client:
            tools = await client.list_tools()
            rows = tools.tools if hasattr(tools, 'tools') else tools
            print(json.dumps({'ok': True, 'name': client.server_info.name,
                              'version': str(client.server_info.version),
                              'tools': len(rows)}))
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}))

asyncio.run(main())
"@ 2>&1 | Select-Object -Last 1

    try {
        $health = $probe | ConvertFrom-Json
    } catch {
        $health = $null
    }
    if ($health -and $health.ok) {
        Write-Host ("   健康     : OK（服务名 {0}，版本 {1}，工具 {2} 个）" -f $health.name, $health.version, $health.tools) -ForegroundColor Green
    } else {
        $message = if ($health) { $health.error } else { $probe }
        Write-Host "   健康     : 不可用（$message）" -ForegroundColor Red
    }
}

Show-Service -Name "spatial" -Url $SpatialUrl
Show-Service -Name "hydro" -Url $HydroUrl
Write-Host ""
Write-Host "日志目录：$(Join-Path $ProjectRoot 'logs\mcp')"
Write-Host "停止服务：powershell -ExecutionPolicy Bypass -File scripts\stop_mcp_services.ps1"
