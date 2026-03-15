# OpenBrowser Installer for Windows (PowerShell)
# Usage: irm https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/main/install.ps1 | iex
# Or:    .\install.ps1 [-NoBrowser]
$ErrorActionPreference = 'Stop'

$PACKAGE = "openbrowser-ai"
$MIN_PYTHON_MAJOR = 3
$MIN_PYTHON_MINOR = 12

# --- Colors ---
function Write-Info  { param([string]$Msg) Write-Host $Msg -ForegroundColor Green }
function Write-Warn  { param([string]$Msg) Write-Host $Msg -ForegroundColor Yellow }
function Write-Err   { param([string]$Msg) Write-Host $Msg -ForegroundColor Red }
function Write-Bold  { param([string]$Msg) Write-Host $Msg -ForegroundColor White -BackgroundColor DarkGray }

# --- Parse args ---
$SkipBrowser = $false
foreach ($a in $args) {
    switch ($a) {
        '--no-browser' { $SkipBrowser = $true }
        '-NoBrowser'   { $SkipBrowser = $true }
        { $_ -in '--help', '-h' } {
            Write-Host "Usage: install.ps1 [OPTIONS]"
            Write-Host ""
            Write-Host "Options:"
            Write-Host "  --no-browser   Skip Chromium installation"
            Write-Host "  -h, --help     Show this help message"
            exit 0
        }
    }
}

# --- Find Python 3.12+ ---
$Python = $null

function Find-Python {
    # Try py launcher with specific versions first, then generic commands
    $candidates = @(
        @{ Cmd = 'py'; Args = @('-3', '--version')    ; Run = 'py'; RunArgs = @('-3') },
        @{ Cmd = 'py'; Args = @('-3.13', '--version') ; Run = 'py'; RunArgs = @('-3.13') },
        @{ Cmd = 'py'; Args = @('-3.12', '--version') ; Run = 'py'; RunArgs = @('-3.12') },
        @{ Cmd = 'python3'; Args = @('--version')     ; Run = 'python3'; RunArgs = @() },
        @{ Cmd = 'python';  Args = @('--version')     ; Run = 'python';  RunArgs = @() }
    )

    foreach ($c in $candidates) {
        try {
            $null = Get-Command $c.Cmd -ErrorAction Stop
            $versionOutput = & $c.Cmd @($c.Args) 2>&1 | Out-String
            if ($versionOutput -match '(\d+)\.(\d+)\.(\d+)') {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -gt $MIN_PYTHON_MAJOR -or ($major -eq $MIN_PYTHON_MAJOR -and $minor -ge $MIN_PYTHON_MINOR)) {
                    $script:Python = @{ Cmd = $c.Run; RunArgs = $c.RunArgs; Version = "$major.$minor.$($Matches[3])" }
                    return $true
                }
            }
        }
        catch {
            continue
        }
    }
    return $false
}

function Invoke-Python {
    param([string[]]$Arguments)
    $p = $script:Python
    $allArgs = $p.RunArgs + $Arguments
    & $p.Cmd @allArgs
}

function Get-PythonDisplay {
    $p = $script:Python
    if ($p.RunArgs.Count -gt 0) {
        return "$($p.Cmd) $($p.RunArgs -join ' ')"
    }
    return $p.Cmd
}

# --- Install methods (in preference order) ---
function Install-WithUv {
    try { $null = Get-Command 'uv' -ErrorAction Stop } catch { return $false }
    Write-Info "Installing with uv..."
    & uv tool install $PACKAGE
    if ($LASTEXITCODE -ne 0) { return $false }
    return $true
}

function Install-WithPipx {
    try { $null = Get-Command 'pipx' -ErrorAction Stop } catch { return $false }
    Write-Info "Installing with pipx..."
    $pythonExe = (Invoke-Python -Arguments @('-c', 'import sys; print(sys.executable)')).Trim()
    & pipx install --python $pythonExe $PACKAGE
    if ($LASTEXITCODE -ne 0) { return $false }
    return $true
}

function Install-WithPip {
    if (-not $script:Python) { return $false }
    $display = Get-PythonDisplay
    Write-Info "Installing with $display -m pip..."
    Invoke-Python -Arguments @('-m', 'pip', 'install', $PACKAGE)
    if ($LASTEXITCODE -ne 0) { return $false }
    return $true
}

# --- Main ---
Write-Bold "OpenBrowser Installer"
Write-Host "====================="
Write-Host ""

# OS info
$arch = if ([Environment]::Is64BitOperatingSystem) { "x86_64" } else { "x86" }
$osVersion = [System.Environment]::OSVersion.VersionString
Write-Host "  OS:      Windows ($arch) - $osVersion"

if (-not (Find-Python)) {
    Write-Err "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ is required but not found."
    Write-Host ""
    Write-Host "Install Python:"
    Write-Host "  winget install Python.Python.3.12"
    Write-Host "  https://www.python.org/downloads/"
    exit 1
}

$pyDisplay = Get-PythonDisplay
Write-Host "  Python:  $pyDisplay (Python $($script:Python.Version))"
Write-Host ""

$Installer = $null
if (Install-WithUv) {
    $Installer = "uv"
}
elseif (Install-WithPipx) {
    $Installer = "pipx"
}
elseif (Install-WithPip) {
    $Installer = "pip"
}
else {
    Write-Err "No Python package manager found."
    Write-Host ""
    Write-Host "Install one of: uv, pipx, or pip"
    Write-Host "  powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`""
    Write-Host "  pip install pipx"
    Write-Host "  pipx ensurepath"
    exit 1
}

# --- Install Chromium ---
if (-not $SkipBrowser) {
    Write-Host ""
    Write-Info "Installing Chromium browser..."
    try {
        $null = Get-Command 'openbrowser-ai' -ErrorAction Stop
        & openbrowser-ai install 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Chromium install failed (run 'openbrowser-ai install' manually)"
        }
    }
    catch {
        try {
            $null = Get-Command 'uvx' -ErrorAction Stop
            & uvx playwright install chromium 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "Chromium install failed (run 'openbrowser-ai install' manually)"
            }
        }
        catch {
            Write-Warn "Chromium install skipped. Please run 'openbrowser-ai install' manually after installation completes."
        }
    }
}

# --- Done ---
Write-Host ""
Write-Info "OpenBrowser installed successfully! (via $Installer)"
Write-Host ""
Write-Host "  Get started:"
Write-Host "    openbrowser-ai --help"
Write-Host "    openbrowser-ai -c `"await navigate('https://example.com')`""
Write-Host ""
Write-Host "  Docs: https://docs.openbrowser.me"
Write-Host ""
