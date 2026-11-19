# Security

## Reporting
Please report vulnerabilities privately through GitHub Security Advisories ("Report a vulnerability" on this repository). Do not open public issues for security problems.

## Review guidelines (used by automated security review)
Read `docs/threat-model.md` first. Report a finding only if this change weakens one of its trust boundaries; explain the trigger and the impact, and cite the changed lines.

Treat as **high** severity:
- a secret (API key, ChatGPT token, OpenCode/GitHub token, share key) printed, logged, stored outside its 0600 store, sent to the wrong host, or reachable by model tools;
- a project file or model output able to change user-only settings, enable a feature flag, or bypass the sandbox or network guard (off Linux there is no OS sandbox by owner policy: in-process guards stay, bash gets the same cleared base environment, a refused bash network disables it, and a run that keeps the network says so at start and to the model);
- metered API requests that bypass required reservation or settlement, are missing from the ledger, or bypass a **configured** daily cap. With no configured cap there is no cap check for the owner or authenticated allowlisted share guests. Guest authority, tokens and request quotas still apply;
- OpenCode Go requests that bypass subscription token accounting, or provider-reported paid charges that are missing from the ledger or shared daily spend. Go admission reserves subscription tokens only. The owner's local metered API cap does not apply to Go admission, including provider-side fallback to a paid balance. A configured `oxa.cloud.subscription_limits` is advisory (warn at 80%/100%, recorded, never a local refusal), and a stored owner Go key enabling the Go transport is owner policy: transport selection stays user-only, so a project setting, a delegated model or a restored session model that flips it *is* a finding;
- fork-PR or guest access to secrets, cloud providers or owner credentials;
- new `unsafe`/FFI/process/network code in the engine without bounds checks.

The sandbox can forward the owner's `CARGO_TARGET_DIR` to worker shells. External cache
paths are writable only through an explicit owner `oxa.sandbox.writePaths` mount.

Persisted headless print and JSON runs with a pursuing goal consume plain `oxa send` inbox
messages with `oxa.inbox.headless` set to `true` by default. This user-only setting can opt out;
project settings cannot enable or disable it. Non-persisted sessions, sessions without a pursuing
goal, and the managed-steering path do not mount the plain inbox watcher.

The background-job and per-parent child limits are configurable through `oxa.bash.maxJobs`
(default 16) and `oxa.subagent.maxChildren` (default 8). Read both caps from the user file.
Project settings for remote concurrency remain bounded by the user’s child cap. An explicit
bash `timeout: 0` opts out of the call timeout. Sandbox and permission checks still apply.

Automated Go review reads at most 200 GitHub comment pages (20,000 comments) when it checks the
configured daily quota and prior same-diff receipts. If either listing is incomplete, review is
deferred before quota reservation, provider authentication, or provider I/O. An incomplete history
cannot authorize another token-spending review.

Treat as **medium**: missing bounds on untrusted input (sizes, lengths, recursion), error messages that echo provider text, weakened validation of provider usage/billing fields.

Do not report: style, pre-existing issues outside the diff, test-only code that never runs in production, or hypothetical risks without a concrete path.

For `oxa work`, a task with zero or skipped checks remains `Unverified` and lacks a publishable
acceptance receipt. Project files and worker replies cannot change the owner-only
`oxa.work.localChecks` setting, which defaults to per-package `cargo check`. Rust acceptance
requires hosted test evidence for the exact head, including for changed inputs outside packages
in a tree with Cargo manifests. A worker declaration can add requirements to that evidence.

The repeat guard warns and continues by owner decision (#1448): the third identical tool result gets a notice, a call that still repeats gets one persisted warning, and the run is not stopped. A finding that asks for the old stopping behaviour is not a finding; `oxa.maxTurns` is the configured cap.

A pursuing goal whose request fails transiently (stream cut, engine down or busy, timeout, 429, 5xx) waits without a model request and then sends the same request again, by owner decision (#1448): `oxa.goal.failureBackoffSeconds` (default 60, doubling, capped at 900 unless set higher, `0` opts out) is the only control, and there is no attempt cap. Auth errors, refusals, a repetition stop, an abort and an owner-set budget still stop the goal with the reason shown. A finding that asks for an attempt cap, a default-off switch or a new terminal exception is not a finding; it is answered here.

Board evidence limits degrade only the affected pull request. A connection that cannot be read
completely produces a visible, ineligible row and cannot trigger review, landing, or refill actions;
the other rows remain usable. An over-page merge queue retains its observed entries, while every
unobserved pull request has unknown queue state and cannot trigger a queue action.

Semantic report verification is one consent, by owner decision (#1448): opting into `oxa.reportIssue.semanticDedupe` is the owner's consent to send the scrubbed report, a public source excerpt and candidate summaries to the configured model and to search the repository for matching issues, so the post asks one question — the final preview. New issues are limited to one per task, and the per-worker daily count is the owner setting `oxa.reportIssue.dailyNew`: unset means no check, and at or over a configured cap the preview warns, the ledger records it and the owner's yes still posts. A finding that asks for a second consent question, for a built-in daily cap, or for the cap to block is answered here.

Intentional removals must be stated in the PR body with a `Removes:` line; flag silent deletion of security checks as high.

## Sandbox standard (owner decision)

- **Interactive CLI:** no OS sandbox by default, on Linux and macOS. Dangerous commands ask the owner.
- **Yolo:** no OS sandbox for the session, nor for any worker, subagent, headless or unattended run it starts. Yolo refuses only publishing secrets and an OS-breaking wipe.
- **Unattended runs without yolo:** keep bwrap by default.
- **In-process guards in every mode:** secret masking, private-path denials, user-only settings, offline mode.
- **Re-adding an OS sandbox to interactive or yolo paths** is a regression, not a hardening.
