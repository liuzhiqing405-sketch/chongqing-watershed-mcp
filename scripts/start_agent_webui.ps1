<#
.SYNOPSIS
    Starts local MCP services and the WebUI with a configurable model connection.

.DESCRIPTION
    The DeepSeek API key is read as a masked value and exists only in this
    launcher's memory and the WebUI child process environment. It is never
    written to source, YAML, command-line arguments, logs, or session files.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start_agent_webui.ps1
#>
[CmdletBinding()]
param(
    [int]$WebPort = 8780,
    [int]$SpatialPort = 8765,
    [int]$HydroPort = 8766,
    [string]$Model = $env:SLOPE_AGENT_LLM_MODEL,
    [string]$BaseUrl = $env:SLOPE_AGENT_LLM_BASE_URL,
    [string]$Provider = $env:SLOPE_AGENT_LLM_PROVIDER,
    [string]$ConfigPath = $env:SLOPE_AGENT_CONFIG,
    [switch]$ConfigureInWeb
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = if ($ConfigPath) { [System.IO.Path]::GetFullPath($ConfigPath) } else { Join-Path $ProjectRoot "config\agent.deepseek.yaml" }
if ($Provider -and $Provider -notin @('deepseek','openai-compatible','anthropic','grok','gemini','qwen')) { throw 'Unknown model provider.' }
if ($BaseUrl -and -not $Provider) { $Provider = 'openai-compatible' }
$StartServices = Join-Path $PSScriptRoot "start_mcp_services.ps1"
$ContractProbe = Join-Path $PSScriptRoot "check_mcp_contract.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project virtual environment not found. Install with: pip install -e .[web,llm]"
}
if (-not (Test-Path -LiteralPath $Config)) {
    throw "Agent profile not found: $Config"
}

function Test-LocalListener {
    param([int]$Port)
    $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    return $null -ne ($listeners | Where-Object { $_.Port -eq $Port } | Select-Object -First 1)
}

function Get-MCPContract {
    param(
        [string]$Url,
        [string]$ExpectedName,
        [string[]]$RequiredTools
    )
    $raw = & $Python $ContractProbe $Url $ExpectedName ($RequiredTools -join ",") 2>&1
    $line = $raw | Select-Object -Last 1
    try { return $line | ConvertFrom-Json }
    catch { return [pscustomobject]@{ ok = $false; error = [string]$line } }
}

function Assert-MCPContract {
    param(
        [string]$Label,
        [string]$Url,
        [string]$ExpectedName,
        [string[]]$RequiredTools
    )
    $deadline = (Get-Date).AddSeconds(15)
    do {
        $contract = Get-MCPContract -Url $Url -ExpectedName $ExpectedName -RequiredTools $RequiredTools
        if ($contract.ok) {
            Write-Host ("{0}: {1} {2}; {3} tools; contract OK." -f `
                $Label, $contract.name, $contract.version, $contract.tool_count) -ForegroundColor Green
            return
        }
        # A completed handshake with the wrong identity or tool set will not
        # become correct by waiting. Retry only while a newly started service
        # is still unavailable.
        if ($contract.name -or ($contract.missing -and $contract.missing.Count)) { break }
        Start-Sleep -Milliseconds 500
    } until ((Get-Date) -ge $deadline)

    $reason = if ($contract.missing -and $contract.missing.Count) {
        "missing tools: " + ($contract.missing -join ", ")
    } elseif ($contract.name -and $contract.name -ne $ExpectedName) {
        "server name is $($contract.name); expected $ExpectedName"
    } else {
        [string]$contract.error
    }
    throw "$Label MCP contract validation failed ($reason). The port may host an old or unrelated service. " +
          "Run scripts\check_mcp_services.ps1, identify the process, then safely restart the MCP services. " +
          "This launcher will not terminate an unknown process."
}

# MCP workers never need the model credential. Temporarily remove an inherited
# key before starting them so only the WebUI process receives it.
$InheritedKey = $env:SLOPE_AGENT_LLM_API_KEY
Remove-Item Env:SLOPE_AGENT_LLM_API_KEY -ErrorAction SilentlyContinue
if ($ConfigureInWeb) { $InheritedKey = $null }

$spatialReady = Test-LocalListener -Port $SpatialPort
$hydroReady = Test-LocalListener -Port $HydroPort
if ($spatialReady -xor $hydroReady) {
    throw "Only one MCP port is listening. Check services before starting the WebUI."
}
if (-not $spatialReady) {
    & $StartServices -SpatialPort $SpatialPort -HydroPort $HydroPort
    if (-not $?) { throw "MCP service launcher reported a failure." }
    $deadline = (Get-Date).AddSeconds(20)
    do {
        Start-Sleep -Milliseconds 250
        $spatialReady = Test-LocalListener -Port $SpatialPort
        $hydroReady = Test-LocalListener -Port $HydroPort
    } until (($spatialReady -and $hydroReady) -or (Get-Date) -ge $deadline)
    if (-not ($spatialReady -and $hydroReady)) {
        throw "MCP services did not both become ready within 20 seconds. Run scripts\check_mcp_services.ps1."
    }
}

$SpatialRequiredTools = @(
    "list_administrative_areas", "render_administrative_dem", "create_drawn_study_area",
    "list_datasets", "inspect_dataset", "validate_data_bundle",
    "recommend_outlets", "render_outlet_candidates", "delineate_from_outlet", "get_watershed_review",
    "render_process_artifact", "confirm_watershed", "build_virtual_rainfall",
    "prepare_model_inputs", "cleanup_spatial_products"
)
$HydroRequiredTools = @(
    "get_hydro_capabilities", "import_hydro_input_package", "register_hydro_inputs",
    "validate_hydro_input", "list_hydro_inputs", "plan_hydro_run", "start_hydro_run",
    "get_hydro_run_status", "cancel_hydro_run", "get_hydro_run_result",
    "recheck_hydro_qa", "inspect_swmm_model", "list_hydro_artifacts",
    "read_hydro_text_artifact", "list_hydro_visualization_options",
    "create_hydro_visualization", "get_hydro_visualization_status",
    "get_hydro_visualization", "compare_hydro_visualizations",
    "list_visualization_artifacts", "prepare_artifact_export",
    "prepare_swmm_model_bundle", "locate_artifact", "list_swmm_model_artifacts", "cleanup_hydro_products"
)
Assert-MCPContract -Label "Spatial" -Url "http://127.0.0.1:$SpatialPort/mcp" `
    -ExpectedName "chongqing-watershed-spatial" -RequiredTools $SpatialRequiredTools
Assert-MCPContract -Label "Hydro" -Url "http://127.0.0.1:$HydroPort/mcp" `
    -ExpectedName "watershed-hydro-stability" -RequiredTools $HydroRequiredTools

$Prompted = $false
if (-not $ConfigureInWeb -and [string]::IsNullOrWhiteSpace($InheritedKey)) {
    $SecureKey = Read-Host "Model API Key (masked; not saved)" -AsSecureString
    $Credential = New-Object System.Net.NetworkCredential("", $SecureKey)
    $InheritedKey = $Credential.Password
    $Prompted = $true
}
if (-not $ConfigureInWeb -and [string]::IsNullOrWhiteSpace($InheritedKey)) {
    throw "Model API Key cannot be empty."
}

try {
    if ($InheritedKey) { $env:SLOPE_AGENT_LLM_API_KEY = $InheritedKey }
    if ($Provider) { $env:SLOPE_AGENT_LLM_PROVIDER = $Provider }
    if ($BaseUrl) { $env:SLOPE_AGENT_LLM_BASE_URL = $BaseUrl }
    if ($Model) { $env:SLOPE_AGENT_LLM_MODEL = $Model }
    Write-Host "WebUI: http://127.0.0.1:$WebPort"
    Write-Host "API docs: http://127.0.0.1:$WebPort/api/docs"
    Write-Host "Press Ctrl+C to stop the WebUI. MCP services remain available."
    & $Python -m slope_evaluation_agent.web --config $Config --port $WebPort
}
finally {
    Remove-Item Env:SLOPE_AGENT_LLM_API_KEY -ErrorAction SilentlyContinue
    $InheritedKey = $null
    if ($Prompted) {
        $SecureKey = $null
        $Credential = $null
    }
}
