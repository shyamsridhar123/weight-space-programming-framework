# Bounded issue factory

`scripts/dark_factory.py` works on eligible GitHub issues in isolated Git
worktrees. One coding attempt and one independent review must satisfy every
assigned checklist requirement. The host checks evidence paths, repository
changes, dependencies, tests, and publication content. Passing tests is
engineering evidence, not a measure of scientific impact.

With `--publish`, new accepted work is pushed to the deterministic
`factory/<run_id>` branch and submitted as a pull request targeting `main`.
The run stays pending and blocks the queue until that PR is merged separately;
the factory does not merge it or enable automatic merge. Without `--publish`,
accepted work is integrated locally and the issue stays open.

Before commenting on or closing the issue, the factory verifies the PR's
repository identities, branch names, exact title/body and exact candidate head.
It fetches the commit advertised by remote `main` and requires both the original
candidate and the PR's reported merge commit to be ancestors of that fetched
head. Only then does it fast-forward local `main` and record issue completion.
The merge must preserve the candidate's ancestry: squash or rebase merges that
drop the original candidate stay pending. A changed PR head, conflicting PR or
closure without a merge also stops completion.

## Retained state and recovery

Private state defaults to `.runs/dark-factory`. Each run retains its issue
snapshot, assigned provider attempts, raw command output, command receipts, and
`result.json`. Result checkpoints use atomic file replacement before provider
invocation, candidate integration, and publication. Interrupted command output
is retained under its original name; recovery uses new receipt names.

Before each GitHub POST or PATCH, the factory atomically records a pending
`github-pull-request.json`, `github-comment.json` or `github-close.json` journal.
Its binding includes the repository identity, issue, issue snapshot digest, run,
candidate commit, method, path and request digest. The PR binding also fixes the
base commit and base/head refs. PR bodies and completion comments contain
deterministic markers derived from their bindings. The PR body uses fixed factory
text and links to the retained review artifact; model-generated review prose is
not inserted into it.

Recovery reads the current issue, repository identity and dependencies, all PR
listing pages and the selected PR's full state, and every completion-comment
page. A matching PR, comment or completed closure can confirm an effect even
when its original response was lost. Conflicting or duplicate markers stop
recovery. An unresolved journal also stops the queue: absence from a fresh read
cannot prove that a delayed request will never take effect, so the factory does
not automatically resend it. A later matching observation can advance the same
run without repeating the request.

Branch publication is creation-only. An existing `factory/<run_id>` branch must
point to the exact candidate; the factory never overwrites a conflicting branch.
After a lost push response, recovery checks the actual remote ref before
proceeding. Before any PR POST it verifies that ref and writes the pending PR
journal. A lost POST response is reconciled through PR observations rather than
a blind duplicate POST. A verified merged PR can still be reconciled after its
source branch is deleted.

The `run` command recovers retained work before selecting another issue.
Publishing runs require `--publish` during recovery. Candidate branches, PRs and
local integration are reconciled against actual Git and GitHub observations.
Every unfinished run with a checkpointed candidate uses candidate reconciliation,
even when its retained status is `failed`; it cannot fall back to
failure-accounting publication.

Legacy publishing checkpoints without `publication_mode` can complete only when
their exact retained candidate is already an ancestor of fetched remote `main`.
This applies to legacy accepted and failed statuses. An unpublished legacy
candidate stays pending for operator reconciliation; recovery neither pushes it
directly to `main` nor silently creates a replacement PR. New PR-mode runs still
require their bound PR and verified merge ancestry.

The separate failure-accounting path applies only when no candidate commit was
checkpointed. It retains its existing direct-main behavior: it appends a public
accounting receipt to clean local `main` and, with `--publish`, can push that
accounting commit directly to remote `main`. The factory rejects unrelated
unpublished changes on `main`. Failure receipts retain allowlisted attempt and
command accounting and exception classes; failed worktrees, exception text and
raw output stay private. Their commit and push checkpoints support recovery
without duplicate accounting commits. Historical failure receipts retain their
original scope.

Failure publication fetches the exact commit advertised by remote `main` before
checking its ancestry against the retained accounting commit. An advertisement
alone does not provide the local Git object. Divergent history and fetch failure
still stop recovery; the existing failure receipt remains unchanged. The
[focused engineering receipt](../../provenance/verification/factory-recovery-remote-fetch-20261109.json)
records an offline mock of those branches. It does not execute retained recovery
or verify a live missing-object transfer.

## Operator boundaries

The lock is conservative. A killed process can leave `factory.lock`. Inspect
the recorded process and its descendants, retained state, and remote effects
before removing only a confirmed stale lock and rerunning the original command.
Unresolved request journals must remain available for reconciliation. The
factory has no automatic stale-lock reclamation or forced GitHub retry.

Completion is saved before worktree cleanup. A crash at that point can leave an
orphan worktree for later inspection and removal without making the completed
issue eligible for another run.

`tests/test_dark_factory.py` uses temporary Git repositories and a fake GitHub
service to exercise lost responses, interrupted integration/push/API effects,
pagination, conflicting markers, queue blocking, and interrupted failure
accounting. These are offline orchestration checks. They do not establish live
GitHub consistency behavior, safety against concurrent external writers, or
scientific outcomes.

The [PR publication engineering receipt](../../provenance/verification/factory-pr-publication-20261109.json)
records one local attempt on November 9, 2026 that timed out after 300 seconds,
without retry. Progress dots establish no completed test result; JUnit, actual
import inventory and runner after-state artifacts were unavailable. A separate
post-termination read/hash observation found all 68 guarded files unchanged from
launch, before the receipt and documentation edits. Ordinary new-head hosted PR
CI remains the next validation; no live GitHub recovery is established.

## Native Windows schedule

Use Windows Task Scheduler after the verified repository is published and local
`main` matches remote `main`. The task starts at the next clock hour and repeats
hourly while the operator is signed in. It uses the existing interactive token,
stores no account password, runs without elevation, and starts a hidden process.
`IgnoreNew` prevents overlapping task instances; the factory's own lock also
covers manual invocations using the same state directory.

Each invocation selects at most one new eligible issue. The coding worker,
validation command, and independent reviewer each have a 480-second deadline.
The scheduler imposes a separate 55-minute limit on the task. It has no automatic
restart rule, daemon, or additional queue service. A missed start can run when
available; it does not create parallel work or override the lock.

Save this wrapper as `.runs/dark-factory/schedule/run-hourly.ps1`, replacing the
three operator-specific placeholders. Use the existing workspace Python
environment with the repository's dependencies. The `-X utf8 -B -u` flags enable
UTF-8 Python I/O, suppress bytecode writes, and flush output promptly to private
logs.

Before registration, authenticate Git and Codex as the same Windows user who
will run the task. The factory obtains GitHub API credentials from Git's
configured credential helper; that sign-in must allow pushing branches,
creating pull requests, and commenting on and closing issues. The wrapper disables
terminal and credential-helper prompts, so an expired sign-in fails instead of
opening an authentication window. Keep credentials in the existing authentication stores.
Logs and startup errors stay under `.runs`.

```powershell
$ErrorActionPreference = 'Stop'
$repo = '<absolute-public-checkout>'
$python = '<absolute-python.exe>'
$env:PATH = '<installed-codex-bin>;<installed-git-directory>;' + $env:PATH
$env:GIT_TERMINAL_PROMPT = '0'
$env:GCM_INTERACTIVE = '0'
$logDirectory = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$prefix = Join-Path $logDirectory ((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
$arguments = '-X utf8 -B -u "{0}\scripts\dark_factory.py" run --state-directory "{0}\.runs\dark-factory" --worker-seconds 480 --max-issues 1 --publish' -f $repo
try {
    $factory = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput ($prefix + '.stdout.private') -RedirectStandardError ($prefix + '.stderr.private')
    exit $factory.ExitCode
} catch {
    $_.Exception.GetType().FullName | Out-File -LiteralPath ($prefix + '.startup-error.private') -Encoding utf8
    exit 1
}
```

Register with native PowerShell cmdlets. These settings allow battery operation,
avoid an interruption merely because AC power changes, and do not wake the
computer. No password or automatic restart is configured. Registration without
`-Force` preserves an existing task of the same name for inspection.

```powershell
$repo = '<absolute-public-checkout>'
$wrapper = Join-Path $repo '.runs\dark-factory\schedule\run-hourly.ps1'
$taskName = 'WeightSpaceProgramming-DarkFactory'
$now = Get-Date
$action = New-ScheduledTaskAction -Execute (Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe') -Argument ('-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $wrapper + '"') -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Once -At $now.Date.AddHours($now.Hour + 1) -RepetitionInterval (New-TimeSpan -Hours 1)
$trigger.Repetition.StopAtDurationEnd = $false
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 55) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings
Get-ScheduledTask -TaskName $taskName
Get-ScheduledTaskInfo -TaskName $taskName
```

`LastTaskResult` preserves the wrapper's factory exit code: `0` means a completed
invocation or an empty eligible queue, `1` means failed or unresolved work, and
`2` means the factory lock blocked the invocation. Windows can also report its
own scheduler errors. Read the private logs and run's `result.json` to distinguish
an idle queue from completed work; task registration alone proves neither.

A forced stop, sign-out, or computer shutdown can interrupt the wrapper before
cleanup. The scheduler's time limit does not attest that descendants stopped or
that a remote request never took effect. Inspect the recorded process tree,
retained journals, and remote state before removing a confirmed stale lock.
Later triggers never reclaim it. Pending GitHub requests remain subject to the
reconciliation rules above, without automatic resend. An accounted ordinary
failure can lead to a fresh attempt on a later hourly invocation; it does not
replay the previous worker or reviewer call.

Disable future starts with `Disable-ScheduledTask -TaskName $taskName`. This does
not stop an invocation already running; inspect its state before interrupting it.
