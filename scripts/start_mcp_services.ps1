<#
.SYNOPSIS
    启动两个常驻 MCP 服务（数据与空间处理、水文与稳定性）。

.DESCRIPTION
    智能体以 streamable-http 连接常驻服务，CLI 退出不会终止服务里的水文任务。
    默认只监听 127.0.0.1，日志写入 logs\mcp\。

    端口被占用时脚本会拒绝启动并说明占用者，不会结束任何未知进程。
    停止服务请显式运行 stop_mcp_services.ps1。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start_mcp_services.ps1
#>
[CmdletBinding()]
param(
    [int]$SpatialPort = 8765,
    [int]$HydroPort = 8766,
    [string]$Host_ = "127.0.0.1",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogRoot = Join-Path $ProjectRoot "logs\mcp"
$PidFile = Join-Path $LogRoot "services.json"

if (-not (Test-Path $Python)) {
    Write-Error "找不到项目虚拟环境：$Python；请先创建并安装（pip install -e .）"
    exit 2
}
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Get-PortOwner {
    param([int]$Port)
    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    $owner = $connections | Select-Object -First 1 -ExpandProperty OwningProcess
    if (-not $owner) {
        # Get-NetTCPConnection can silently return no rows without elevation on
        # some Windows installations. The .NET listener table is readable by a
        # normal user; netstat supplies the PID when available.
        $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
        if (-not ($listeners | Where-Object { $_.Port -eq $Port } | Select-Object -First 1)) {
            return $null
        }
        $pattern = "^\s*TCP\s+\S+:$Port\s+\S+\s+LISTENING\s+(\d+)\s*$"
        $match = netstat -ano -p TCP | Select-String -Pattern $pattern | Select-Object -First 1
        if ($match -and $match.Matches.Count) {
            $owner = [int]$match.Matches[0].Groups[1].Value
        }
    }
    $process = if ($owner) { Get-Process -Id $owner -ErrorAction SilentlyContinue } else { $null }
    return [pscustomobject]@{
        Pid     = if ($owner) { $owner } else { "unknown" }
        Name    = if ($process) { $process.ProcessName } else { "unknown" }
        Started = if ($process) { $process.StartTime } else { $null }
    }
}

function Rotate-Log {
    param([string]$Path, [int]$MaxBytes = 5MB, [int]$Keep = 3)
    if (-not (Test-Path $Path)) { return }
    if ((Get-Item $Path).Length -lt $MaxBytes) { return }
    for ($index = $Keep - 1; $index -ge 1; $index--) {
        $older = "$Path.$index"
        if (Test-Path $older) { Move-Item -Force $older "$Path.$($index + 1)" }
    }
    Move-Item -Force $Path "$Path.1"
}

function Start-Service {
    param(
        [string]$Name,
        [string]$Module,
        [int]$Port,
        [string]$HealthUrl
    )
    $existing = Get-PortOwner -Port $Port
    if ($existing) {
        if ($Force) {
            Write-Warning "端口 $Port 已被 PID $($existing.Pid)（$($existing.Name)）占用；-Force 仍然拒绝结束它。"
        }
        $occupied = "端口 $Port 已被 PID $($existing.Pid)（$($existing.Name)）占用。" +
                    "请先确认该进程，或改用 -${Name}Port 指定其他端口。脚本不会结束未知进程。"
        Write-Error $occupied
        return $null
    }
    $log = Join-Path $LogRoot "$Name.log"
    Rotate-Log -Path $log
    $arguments = @(
        "-m", $Module,
        "--transport", "streamable-http",
        "--host", $Host_,
        "--port", "$Port"
    )
    $process = Start-Process -FilePath $Python -ArgumentList $arguments `
        -WorkingDirectory $ProjectRoot -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -WindowStyle Hidden
    Write-Host "[$Name] 已启动 pid=$($process.Id) 地址=http://${Host_}:$Port/mcp 日志=$log"
    return $process
}

$spatial = Start-Service -Name "spatial" -Module "chongqing_spatial_mcp.server" `
    -Port $SpatialPort -HealthUrl "http://${Host_}:$SpatialPort/mcp"
if (-not $spatial) { exit 1 }

$hydro = Start-Service -Name "hydro" -Module "watershed_hydro_mcp.server" `
    -Port $HydroPort -HealthUrl "http://${Host_}:$HydroPort/mcp"
if (-not $hydro) {
    Stop-Process -Id $spatial.Id -ErrorAction SilentlyContinue
    exit 1
}

$record = @{
    spatial = @{ pid = $spatial.Id; port = $SpatialPort; url = "http://${Host_}:$SpatialPort/mcp" }
    hydro   = @{ pid = $hydro.Id; port = $HydroPort; url = "http://${Host_}:$HydroPort/mcp" }
    started_at = (Get-Date).ToString("o")
}
$record | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $PidFile

Write-Host ""
Write-Host "两个服务已启动。请在 Agent 配置中设置："
Write-Host "  mcp:"
Write-Host "    spatial: { transport: streamable-http, url: http://${Host_}:$SpatialPort/mcp }"
Write-Host "    hydro:   { transport: streamable-http, url: http://${Host_}:$HydroPort/mcp }"
Write-Host ""
Write-Host "运行 scripts\check_mcp_services.ps1 查看健康状态。"
