[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent (Split-Path -Parent $ProjectDirectory)
$EvidenceDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$ToolchainDirectory = Join-Path $RepositoryRoot '.cache/weight-space/lean-4.24.0-windows'
$LeanExecutable = Join-Path $ToolchainDirectory 'bin/lean.exe'
$LakeExecutable = Join-Path $ToolchainDirectory 'bin/lake.exe'
$PreviousPath = $env:PATH
$StartedAt = [DateTimeOffset]::UtcNow
$Timer = [System.Diagnostics.Stopwatch]::StartNew()

if (Test-Path -LiteralPath $EvidenceDirectory) {
    throw "Evidence output already exists; choose a fresh directory: $EvidenceDirectory"
}
if (!(Test-Path -LiteralPath $LeanExecutable) -or !(Test-Path -LiteralPath $LakeExecutable)) {
    throw 'The previously verified, pinned local Lean 4.24.0 toolchain is required. This script does not download tools.'
}

$LeanVersionOutput = (& $LeanExecutable --version) -join "`n"
if ($LASTEXITCODE -ne 0 -or $LeanVersionOutput -notmatch 'version 4\.24\.0,') {
    throw "Wrong Lean toolchain: $LeanVersionOutput"
}
$ManifestPath = Join-Path $ProjectDirectory 'lake-manifest.json'
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$LockedMathlib = @($Manifest.packages | Where-Object name -eq 'mathlib')
if ($LockedMathlib.Count -ne 1 -or $LockedMathlib[0].rev -ne 'f897ebcf72cd16f89ab4577d0c826cd14afaafc7') {
    throw 'The dependency lock does not contain the pinned Mathlib revision.'
}
$DependencyRevisions = [ordered]@{}
foreach ($Package in $Manifest.packages) {
    $PackageDirectory = Join-Path $ProjectDirectory ".lake/packages/$($Package.name)"
    $ActualRevision = (& git -C $PackageDirectory rev-parse HEAD) -join ''
    if ($LASTEXITCODE -ne 0 -or $ActualRevision.Trim() -ne $Package.rev) {
        throw "Dependency $($Package.name) is not at its locked revision."
    }
    & git -C $PackageDirectory diff --quiet HEAD --
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency $($Package.name) has modified tracked sources."
    }
    $DependencyRevisions[$Package.name] = $ActualRevision.Trim()
}

function Get-RelativeHashes {
    param([string[]]$RelativePaths)
    $Hashes = [ordered]@{}
    foreach ($RelativePath in $RelativePaths) {
        $AbsolutePath = Join-Path $RepositoryRoot $RelativePath
        $Hashes[$RelativePath] = (Get-FileHash -LiteralPath $AbsolutePath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    return $Hashes
}

$NewSources = @(
    'formal/weight-space/WeightSpace/Generic.lean',
    'formal/weight-space/WeightSpace/Sequential.lean',
    'formal/weight-space/WeightSpace/GenericExamples.lean',
    'formal/weight-space/WeightSpace/GenericAudit.lean'
)
$VerificationSources = $NewSources + @(
    'formal/weight-space/lake-manifest.json',
    'formal/weight-space/lakefile.toml',
    'formal/weight-space/lean-toolchain',
    'formal/weight-space/verification-v2/check.ps1',
    'formal/weight-space/verification-v2/preserved-v1-inputs.json',
    'docs/weight-space/v2/formal.md'
)
$SourceHashes = Get-RelativeHashes $VerificationSources
$PreservationPath = Join-Path $PSScriptRoot 'preserved-v1-inputs.json'
$Preservation = Get-Content -LiteralPath $PreservationPath -Raw | ConvertFrom-Json
function Assert-PreservedInputs {
    foreach ($Property in $Preservation.sha256.PSObject.Properties) {
        $Actual = (Get-FileHash -LiteralPath (Join-Path $RepositoryRoot $Property.Name) -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $Property.Value) {
            throw "Historical v1 source or evidence changed: $($Property.Name)"
        }
    }
}
Assert-PreservedInputs

New-Item -ItemType Directory -Path $EvidenceDirectory | Out-Null
$BuildLog = Join-Path $EvidenceDirectory 'build.log'
$AuditLog = Join-Path $EvidenceDirectory 'axiom-audit.log'
Push-Location $ProjectDirectory
try {
    $env:PATH = (Join-Path $ToolchainDirectory 'bin') + [System.IO.Path]::PathSeparator + $PreviousPath
    Write-Host $LeanVersionOutput
    & $LakeExecutable build WeightSpace.GenericAudit 2>&1 | Tee-Object -FilePath $BuildLog
    if ($LASTEXITCODE -ne 0) { throw "V2 target build failed with exit code $LASTEXITCODE." }
    & $LakeExecutable env lean 'WeightSpace/GenericAudit.lean' 2>&1 | Tee-Object -FilePath $AuditLog
    if ($LASTEXITCODE -ne 0) { throw "V2 standalone transitive audit failed with exit code $LASTEXITCODE." }
} finally {
    $env:PATH = $PreviousPath
    Pop-Location
}

$AfterHashes = Get-RelativeHashes $VerificationSources
foreach ($Key in $SourceHashes.Keys) {
    if ($SourceHashes[$Key] -ne $AfterHashes[$Key]) { throw "Source changed during verification: $Key" }
}
Assert-PreservedInputs
$AuditText = Get-Content -LiteralPath $AuditLog -Raw
$AllMatch = [regex]::Match($AuditText, 'V2 full transitive axiom audit passed: (\d+) declarations; (\d+) theorem declarations\.')
$NewMatch = [regex]::Match($AuditText, 'V2 new namespaces: (\d+) declarations; (\d+) theorem declarations\.')
if (!$AllMatch.Success -or !$NewMatch.Success) { throw 'Required full-namespace audit output is missing.' }
$ObservedMatch = [regex]::Match($AuditText, 'V2 observed axioms:\s*\[([^\]]*)\]\.')
if (!$ObservedMatch.Success) { throw 'Required observed-axiom audit output is missing.' }
$ObservedAxioms = @($ObservedMatch.Groups[1].Value -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$AllowedAxioms = @('propext', 'Classical.choice', 'Quot.sound')
if (@($ObservedAxioms | Where-Object { $_ -notin $AllowedAxioms }).Count -ne 0) {
    throw 'Observed axioms differ from the permitted Lean/Mathlib foundations.'
}
$NamespaceCounts = [ordered]@{}
foreach ($Match in [regex]::Matches($AuditText, 'V2 namespace ([\w.]+): (\d+) declarations; (\d+) theorem declarations\.')) {
    $NamespaceCounts[$Match.Groups[1].Value] = [ordered]@{
        declarations = [int]$Match.Groups[2].Value
        theorem_declarations = [int]$Match.Groups[3].Value
    }
}
$ExplicitCounts = [ordered]@{}
foreach ($RelativePath in $NewSources) {
    $SourceText = Get-Content -LiteralPath (Join-Path $RepositoryRoot $RelativePath) -Raw
    $ExplicitCounts[$RelativePath] = [regex]::Matches($SourceText, '(?m)^theorem ').Count
}
$Timer.Stop()
$Receipt = [ordered]@{
    schema = 'wsp-generic-formal-verification/2'
    started_at = $StartedAt.ToString('o')
    checked_at = [DateTimeOffset]::UtcNow.ToString('o')
    elapsed_seconds = $Timer.Elapsed.TotalSeconds
    exit_code = 0
    commands = @(
        [ordered]@{ command = 'lake build WeightSpace.GenericAudit'; exit_code = 0; log = 'build.log' },
        [ordered]@{ command = 'lake env lean WeightSpace/GenericAudit.lean'; exit_code = 0; log = 'axiom-audit.log' }
    )
    lean_version_output = $LeanVersionOutput
    lean_executable_sha256 = (Get-FileHash -LiteralPath $LeanExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
    mathlib_revision = $LockedMathlib[0].rev
    dependency_revisions = $DependencyRevisions
    dependency_tracked_sources_unmodified = $true
    explicit_new_theorem_statements = [int](($ExplicitCounts.Values | Measure-Object -Sum).Sum)
    explicit_new_theorem_statements_by_source = $ExplicitCounts
    audited_declarations = [int]$AllMatch.Groups[1].Value
    audited_theorem_declarations = [int]$AllMatch.Groups[2].Value
    audited_new_declarations = [int]$NewMatch.Groups[1].Value
    audited_new_theorem_declarations = [int]$NewMatch.Groups[2].Value
    audited_new_namespaces = $NamespaceCounts
    count_conventions = [ordered]@{
        explicit_new_theorem_statements = 'Lines beginning with theorem in the four new sources; excludes definitions and compiler-generated declarations.'
        audited_declarations = 'All WeightSpace namespace declarations loaded by GenericAudit, including the original library and compiler-generated declarations.'
        audited_new_declarations = 'Declarations in Generic, Sequential, or GenericExamples; includes compiler-generated declarations.'
        audited_theorem_declarations = 'Declarations whose Lean ConstantInfo kind is thmInfo, including compiler-generated theorem declarations.'
    }
    allowed_axioms = $AllowedAxioms
    observed_axioms = $ObservedAxioms
    audit_method = 'Lean.collectAxioms follows transitive dependencies of every WeightSpace declaration; elaboration fails for any axiom outside the allowlist.'
    printed_representative_new_theorems = [regex]::Matches((Get-Content -LiteralPath (Join-Path $ProjectDirectory 'WeightSpace/GenericAudit.lean') -Raw), '(?m)^#print axioms ').Count
    proof_placeholders_or_added_axioms_or_native_evaluation_shortcuts_accepted = $false
    source_sha256 = $SourceHashes
    all_verification_sources_unchanged_during_run = $true
    log_sha256 = [ordered]@{
        'build.log' = (Get-FileHash -LiteralPath $BuildLog -Algorithm SHA256).Hash.ToLowerInvariant()
        'axiom-audit.log' = (Get-FileHash -LiteralPath $AuditLog -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    preserved_v1 = [ordered]@{
        snapshot = 'formal/weight-space/verification-v2/preserved-v1-inputs.json'
        snapshot_sha256 = (Get-FileHash -LiteralPath $PreservationPath -Algorithm SHA256).Hash.ToLowerInvariant()
        files_checked = @($Preservation.sha256.PSObject.Properties).Count
        all_unchanged = $true
    }
    scope = 'Real-valued arbitrary frozen feature maps with finite feature coordinates; exact and approximate span protection; factored edit effect and preservation; arbitrary-depth variable-width affine layers with shared componentwise one-Lipschitz activations; explicit non-RTS rational witnesses.'
    trust_boundary = [ordered]@{
        formal_guarantee = 'Lean kernel checking establishes the stated theorems of the explicit definitions under their stated premises.'
        endpoint_witnesses = 'Hand-written rational examples. No claim that these definitions are generated from or formally byte-bound to new Python artifacts.'
        excluded_claims = @(
            'Numerical SVD exactness or solver optimality',
            'IEEE floating-point execution or rounding error bounds',
            'Arbitrary Python or PyTorch forward-program refinement',
            'General transformer or cross-architecture certificate',
            'Unstated feature-span membership or distribution coverage',
            'Task reward, causal repair, absence of forgetting, or source-paper replication',
            'Minimal matrix rank or rank optimality of the factored representation'
        )
    }
}
$Receipt | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $EvidenceDirectory 'verification.json') -Encoding utf8
Write-Host "V2 verification passed. Evidence: $EvidenceDirectory"
