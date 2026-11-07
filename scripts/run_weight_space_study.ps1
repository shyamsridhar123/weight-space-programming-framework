#requires -Version 7.0
<#
.SYNOPSIS
Run the frozen weight-space study using already installed local dependencies.
.DESCRIPTION
Relative -Out and -Protocol paths are resolved from the repository root.
The output directory must not exist. Every stage retains separate stdout/stderr
logs, its exact argument array, and its exit status. Failure stops the run and
leaves the evidence in place; reruns need a new output directory.
This script performs no installs, provider calls, or Lean proof generation.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Out,

    [ValidateNotNullOrEmpty()]
    [string]$Protocol = 'experiments/weight-space/protocol.json'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RepositoryRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))

function Resolve-RepositoryPath {
    param([string]$Value)
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Value))
}

function Write-NewJson {
    param([string]$Path, [object]$Value)
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    $writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false))
    try {
        $writer.WriteLine(($Value | ConvertTo-Json -Depth 12))
    }
    finally {
        $writer.Dispose()
    }
}

function Invoke-StudyStage {
    param([string]$Name, [string]$Executable, [string[]]$Arguments)
    $prefix = Join-Path $LogDirectory $Name
    $startedAt = [DateTimeOffset]::UtcNow.ToString('o')
    Write-NewJson "$prefix.command.json" ([ordered]@{
        stage = $Name
        executable = $Executable
        arguments = $Arguments
        working_directory = $RepositoryRoot
        started_at = $startedAt
    })

    # ArgumentList passes paths with spaces literally, without a shell or quoting.
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Executable
    $info.WorkingDirectory = $RepositoryRoot
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($argument in $Arguments) {
        $info.ArgumentList.Add($argument)
    }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    $stdout = $null
    $stderr = $null
    $exitCode = $null
    $failure = $null
    Write-Host "[$Name] Starting; logs: $prefix.stdout.log and $prefix.stderr.log"
    try {
        $stdout = [IO.File]::Open("$prefix.stdout.log", [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write, [IO.FileShare]::Read)
        $stderr = [IO.File]::Open("$prefix.stderr.log", [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write, [IO.FileShare]::Read)
        if (-not $process.Start()) {
            throw "Could not start $Executable."
        }
        # Drain both pipes concurrently so a full stderr pipe cannot block stdout.
        $stdoutCopy = $process.StandardOutput.BaseStream.CopyToAsync($stdout)
        $stderrCopy = $process.StandardError.BaseStream.CopyToAsync($stderr)
        $process.WaitForExit()
        $stdoutCopy.GetAwaiter().GetResult()
        $stderrCopy.GetAwaiter().GetResult()
        $exitCode = $process.ExitCode
    }
    catch {
        $failure = $_.Exception.Message
        throw
    }
    finally {
        if ($null -ne $stdout) { $stdout.Dispose() }
        if ($null -ne $stderr) { $stderr.Dispose() }
        $process.Dispose()
        Write-NewJson "$prefix.exit.json" ([ordered]@{
            stage = $Name
            started_at = $startedAt
            finished_at = [DateTimeOffset]::UtcNow.ToString('o')
            exit_code = $exitCode
            launch_or_capture_error = $failure
        })
    }
    if ($exitCode -ne 0) {
        throw "Stage $Name failed with exit code $exitCode. See $prefix.stderr.log; evidence was retained."
    }
    Write-Host "[$Name] Completed."
}

$OutputRoot = Resolve-RepositoryPath $Out
$ProtocolPath = Resolve-RepositoryPath $Protocol
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Output already exists; choose a new directory: $OutputRoot"
}
if (-not (Test-Path -LiteralPath $ProtocolPath -PathType Leaf)) {
    throw "Protocol file is missing: $ProtocolPath"
}
$PythonExecutable = Join-Path $RepositoryRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    $PythonExecutable = Join-Path $RepositoryRoot '.venv/bin/python'
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw 'The repository .venv Python is missing. Install the locked dependencies before running this script.'
}
$NodeExecutable = (Get-Command node -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$TsxEntry = Join-Path $RepositoryRoot 'node_modules/tsx/dist/cli.mjs'
$RtsEntry = Join-Path $RepositoryRoot 'scripts/weight-space-empires.ts'
foreach ($required in @($TsxEntry, $RtsEntry)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required local file is missing: $required"
    }
}

New-Item -ItemType Directory -Path $OutputRoot -ErrorAction Stop | Out-Null
$LogDirectory = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Path $LogDirectory -ErrorAction Stop | Out-Null
$FreezeDirectory = Join-Path $OutputRoot 'freeze'
$FreezeReceipt = Join-Path $FreezeDirectory 'receipt.json'
$SyntheticDirectory = Join-Path $OutputRoot 'synthetic'
$ExportDirectory = Join-Path $OutputRoot 'rts-export'
$CandidateDirectory = Join-Path $OutputRoot 'rts-candidates'
$CandidateBundle = Join-Path $CandidateDirectory 'programs.json'
$DevelopmentDirectory = Join-Path $OutputRoot 'rts-development'
$ValidationDirectory = Join-Path $OutputRoot 'rts-validation'
$SelectionDirectory = Join-Path $OutputRoot 'rts-selection'
$TestDirectory = Join-Path $OutputRoot 'rts-test'
$SummaryPath = Join-Path $OutputRoot 'rts-summary.json'
$PythonPrefix = @('-m', 'rlcd.weight_space')
$NodePrefix = @($TsxEntry, $RtsEntry)

Invoke-StudyStage '01-freeze' $PythonExecutable ($PythonPrefix + @(
    'freeze', '--protocol', $ProtocolPath, '--out', $FreezeDirectory))
Invoke-StudyStage '02-synthetic' $PythonExecutable ($PythonPrefix + @(
    'synthetic', '--protocol', $ProtocolPath, '--freeze', $FreezeReceipt, '--out', $SyntheticDirectory))
Invoke-StudyStage '03-rts-export' $NodeExecutable ($NodePrefix + @(
    'export', '--protocol', $ProtocolPath, '--out', $ExportDirectory))
Invoke-StudyStage '04-rts-prepare' $PythonExecutable ($PythonPrefix + @(
    'rts-prepare', '--protocol', $ProtocolPath, '--freeze', $FreezeReceipt,
    '--export', $ExportDirectory, '--out', $CandidateDirectory))
Invoke-StudyStage '05-rts-development' $NodeExecutable ($NodePrefix + @(
    'evaluate', '--protocol', $ProtocolPath, '--programs', $CandidateBundle,
    '--split', 'development', '--out', $DevelopmentDirectory))
Invoke-StudyStage '06-rts-validation' $NodeExecutable ($NodePrefix + @(
    'evaluate', '--protocol', $ProtocolPath, '--programs', $CandidateBundle,
    '--split', 'validation', '--out', $ValidationDirectory))
Invoke-StudyStage '07-rts-select' $PythonExecutable ($PythonPrefix + @(
    'rts-select', '--protocol', $ProtocolPath, '--freeze', $FreezeReceipt,
    '--results', (Join-Path $ValidationDirectory 'results.json'), '--programs', $CandidateBundle,
    '--out', $SelectionDirectory))
# Only selection/programs.json can reach held-out worlds. The evaluator adds controls.
Invoke-StudyStage '08-rts-test' $NodeExecutable ($NodePrefix + @(
    'evaluate', '--protocol', $ProtocolPath, '--programs', (Join-Path $SelectionDirectory 'programs.json'),
    '--split', 'test', '--out', $TestDirectory))
Invoke-StudyStage '09-rts-summarize' $PythonExecutable ($PythonPrefix + @(
    'rts-summarize', '--protocol', $ProtocolPath, '--freeze', $FreezeReceipt,
    '--results', (Join-Path $TestDirectory 'results.json'),
    '--selection', (Join-Path $SelectionDirectory 'selection.json'), '--out', $SummaryPath))

Write-Host "Study finished. Synthetic evidence: $SyntheticDirectory"
Write-Host "RTS summary: $SummaryPath"
Write-Host "Stage logs: $LogDirectory"
