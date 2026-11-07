#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$PythonCommand = Get-Command python -CommandType Application -ErrorAction SilentlyContinue
if ($null -eq $PythonCommand) {
    $PythonCommand = Get-Command python3 -CommandType Application -ErrorAction SilentlyContinue
}
if ($null -eq $PythonCommand) {
    throw 'Python 3.10 or later is required on PATH. See PROOFS.md for direct invocation.'
}
& $PythonCommand.Source (Join-Path $PSScriptRoot 'check_proofs.py') --output $OutputDirectory
exit $LASTEXITCODE
