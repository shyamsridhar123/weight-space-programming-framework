[CmdletBinding()]
param(
    [switch]$Bootstrap,
    [switch]$SkipCache
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$ProjectDirectory = Join-Path $RepositoryRoot 'formal/weight-space'
$CacheDirectory = Join-Path $RepositoryRoot '.cache/weight-space'
$LeanVersion = '4.24.0'
$MathlibRevision = 'f897ebcf72cd16f89ab4577d0c826cd14afaafc7'
$ArchiveName = "lean-$LeanVersion-windows.zip"
$ArchiveSha256 = 'c2fcf1cba1089e526e184de2a6f6763a6e15596a2991f096dfacd4a6e3ce2cdd'
$ToolchainDirectory = Join-Path $CacheDirectory "lean-$LeanVersion-windows"
$LeanExecutable = Join-Path $ToolchainDirectory 'bin/lean.exe'
$PreviousPath = $env:PATH

function Invoke-CheckedNative {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE."
    }
}

New-Item -ItemType Directory -Path $CacheDirectory -Force | Out-Null

if (!(Test-Path -LiteralPath $LeanExecutable)) {
    if ($Bootstrap) {
        if (![System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
            [System.Runtime.InteropServices.OSPlatform]::Windows)) {
            throw 'This bootstrap installs the Windows toolchain. Install Lean 4.24.0 on PATH on other platforms.'
        }
        $ArchivePath = Join-Path $CacheDirectory $ArchiveName
        if (!(Test-Path -LiteralPath $ArchivePath)) {
            Write-Host "Downloading official Lean $LeanVersion toolchain."
            Invoke-WebRequest -Uri "https://github.com/leanprover/lean4/releases/download/v$LeanVersion/$ArchiveName" -OutFile $ArchivePath
        }
        $ActualSha256 = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualSha256 -ne $ArchiveSha256) {
            throw "Lean archive SHA256 mismatch: expected $ArchiveSha256, got $ActualSha256."
        }
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $CacheDirectory -Force
    } else {
        $AvailableLean = Get-Command lean -ErrorAction SilentlyContinue
        if ($null -eq $AvailableLean) {
            throw 'Lean 4.24.0 is required. Run this script with -Bootstrap to install the pinned Windows toolchain.'
        }
        $LeanExecutable = $AvailableLean.Source
    }
}

$LeanVersionOutput = & $LeanExecutable --version
if ($LASTEXITCODE -ne 0 -or ($LeanVersionOutput -join "`n") -notmatch 'version 4\.24\.0,') {
    throw "Wrong Lean toolchain; expected 4.24.0. Reported: $LeanVersionOutput"
}
Write-Host ($LeanVersionOutput -join "`n")
$ToolchainBin = Split-Path -Parent $LeanExecutable
$LakeExecutable = Join-Path $ToolchainBin $(if ($IsWindows -or $env:OS -eq 'Windows_NT') { 'lake.exe' } else { 'lake' })
if (!(Test-Path -LiteralPath $LakeExecutable)) {
    throw "Lake executable missing next to the selected Lean executable: $LakeExecutable"
}

$Manifest = Get-Content -LiteralPath (Join-Path $ProjectDirectory 'lake-manifest.json') -Raw | ConvertFrom-Json
$LockedMathlib = @($Manifest.packages | Where-Object name -eq 'mathlib')
if ($LockedMathlib.Count -ne 1 -or $LockedMathlib[0].rev -ne $MathlibRevision) {
    throw 'The Mathlib lock does not match the pinned proof project revision.'
}
$PackagesDirectory = Join-Path $ProjectDirectory '.lake/packages'
foreach ($Package in $Manifest.packages) {
    $PackageDirectory = Join-Path $PackagesDirectory $Package.name
    if (!(Test-Path -LiteralPath (Join-Path $PackageDirectory '.git'))) {
        if (!$Bootstrap) {
            throw "Missing dependency $($Package.name). Run with -Bootstrap to install locked dependencies."
        }
        if (Test-Path -LiteralPath $PackageDirectory) {
            throw "Dependency path exists without a Git checkout: $PackageDirectory"
        }
        New-Item -ItemType Directory -Path $PackageDirectory -Force | Out-Null
        Invoke-CheckedNative git @('init', '--quiet', $PackageDirectory)
        Invoke-CheckedNative git @('-C', $PackageDirectory, 'remote', 'add', 'origin', $Package.url)
        Invoke-CheckedNative git @('-C', $PackageDirectory, 'fetch', '--quiet', '--depth', '1', 'origin', $Package.rev)
        Invoke-CheckedNative git @('-C', $PackageDirectory, 'checkout', '--quiet', '--detach', 'FETCH_HEAD')
    }
    $ActualRevision = & git -C $PackageDirectory rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $ActualRevision.Trim() -ne $Package.rev) {
        throw "Dependency $($Package.name) is not at locked revision $($Package.rev)."
    }
    Invoke-CheckedNative git @('-C', $PackageDirectory, 'diff', '--quiet', 'HEAD', '--')
}

Push-Location $ProjectDirectory
try {
    $env:PATH = $ToolchainBin + [System.IO.Path]::PathSeparator + $PreviousPath
    if (!$SkipCache) {
        Write-Host 'Retrieving artifacts for the pinned Mathlib imports.'
        Invoke-CheckedNative $LakeExecutable @(
            'exe', 'cache', 'get',
            'Mathlib.Data.Real.Basic',
            'Mathlib.Algebra.BigOperators.Fin',
            'Mathlib.Algebra.Order.BigOperators.Group.Finset',
            'Mathlib.Tactic.Ring',
            'Mathlib.Tactic.Linarith',
            'Mathlib.Tactic.NormNum'
        )
    }
    Write-Host 'Building the complete WeightSpace proof library.'
    & $LakeExecutable build 2>&1 | Tee-Object -FilePath (Join-Path $CacheDirectory 'build.log')
    if ($LASTEXITCODE -ne 0) {
        throw "Lean proof build failed with exit code $LASTEXITCODE."
    }
    Write-Host 'Checking the transitive axiom allowlist.'
    & $LakeExecutable env lean 'WeightSpace/AxiomAudit.lean' 2>&1 | Tee-Object -FilePath (Join-Path $CacheDirectory 'axiom-audit.log')
    if ($LASTEXITCODE -ne 0) {
        throw "Lean axiom audit failed with exit code $LASTEXITCODE."
    }
    Write-Host 'WeightSpace proof build and axiom audit passed.'
} finally {
    $env:PATH = $PreviousPath
    Pop-Location
}
