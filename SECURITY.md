# Security

## Reporting
Please report vulnerabilities privately through GitHub Security Advisories ("Report a vulnerability" on this repository). Do not open public issues for security problems.

## Review guidelines (used by automated security review)
Read `docs/threat-model.md` first. Report a finding only if this change weakens one of its trust boundaries; explain the trigger and the impact, and cite the changed lines. The sandbox standard and the owner decisions below are part of the standard, and so are the review guidelines in `AGENTS.md`: a finding that asks for the opposite is not a finding.

Treat as **high** severity:
- a secret (API key, ChatGPT token, OpenCode/GitHub token, share key) printed, logged, stored outside its 0600 store, sent to the wrong host, or reachable by model tools;
- a project file or model output able to change user-only settings, enable a feature flag, or bypass the sandbox or network guard (off Linux there is no OS sandbox by owner policy: in-process guards stay, bash gets the same cleared base environment, a refused bash network disables it, and a run that keeps the network says so at start and to the model);
- metered API requests that bypass required reservation or settlement, are missing from the ledger, or bypass a **configured** daily cap. With no configured cap there is no cap check for the owner or authenticated allowlisted share guests. Guest authority, tokens and request quotas still apply;
- OpenCode Go requests that bypass subscription token accounting, or provider-reported paid charges that are missing from the ledger or shared daily spend. Go admission reserves subscription tokens only. The owner's local metered API cap does not apply to Go admission, including provider-side fallback to a paid balance. A configured `oxa.cloud.subscription_limits` is advisory (warn at 80%/100%, recorded, never a local refusal), and a stored owner Go key enabling the Go transport is owner policy: the protected user key authorizes an actual Go model argument, including one passed to a delegated worker or nested child. A project setting or restored session model that flips it *is* a finding (the owner's explicit `oxa.cloud.opencode.enabled: true` is the supported way to allow those);
- fork-PR or guest access to secrets, cloud providers or owner credentials;
- new `unsafe`/FFI/process/network code in the engine without bounds checks.

The native cloud gateway's default recommendations use only configured API-key providers
and their standard endpoints. User model entries take precedence, and the owner can opt out.
Recommendations do not initiate network discovery or confer guest access.

Attended bash can run with host rights under the owner’s interactive sandbox policy. It uses
a cleared toolchain environment, and output still redacts every GitHub CLI credential
(all accounts and hosts, with the bash network on or off).
Unattended runs include ownerless RPC and headless runs pursuing a goal. Exec hooks
keep their dedicated sandbox. Network-disabled model shells mask host runtime IPC.
Local model setup refuses agent tool invocations before running engine admission or writing user
settings.

The sandbox can forward the owner's `CARGO_TARGET_DIR` to worker shells. External cache
paths are writable only through an explicit owner `oxa.sandbox.writePaths` mount.
Host runtime sockets (docker, podman) stay masked in sandboxed bash unless the owner lists them in
`oxa.sandbox.sockets` and permits network access. The launch validates and pins each socket
through an FD-based bind (user settings only; the docker socket gives root on the host, so it is noted
at start, not blocked).

`oxa send` keeps owner messages over the 256 KiB inline inbox limit in the receiving session's
private spill directory and queues a short pointer. The sandbox exposes that spill directory only
to the receiving session and sessions inheriting its history; the private agent directory remains
denied to model tools.
Persisted headless print and JSON runs with a pursuing goal consume plain `oxa send` inbox
messages with `oxa.inbox.headless` set to `true` by default. This user-only setting can opt out;
project settings cannot enable or disable it. Non-persisted sessions, sessions without a pursuing
goal, and the managed-steering path do not mount the plain inbox watcher.

An explicit owner CLI `--auto` or `--yolo` is sufficient startup consent for home, root, or a
parent of home, with a warning. Project settings and model output cannot supply that CLI consent.
An interactive time or token limit ends the watched startup session after its bounded wrap-up,
and an idle expiry starts no model request.
The background-job and per-parent child limits are warning thresholds, by owner decision
(2026-11-19): `oxa.bash.maxJobs` (default 16) and `oxa.subagent.maxChildren` (default 8). Above
either, the start goes ahead (extra children queue) and the tool result carries a warning that
names the setting; nothing is refused. Read both from the user file, so a project cannot change them.
Project settings for remote concurrency remain bounded by the user’s child cap. An explicit
bash `timeout: 0` opts out of the call timeout. Sandbox and permission checks still apply.

The agent control lane (steer, pause, resume, cancel for headless sessions) is reachable in release
builds. Control requests are accepted only from the same user and are checked against the lease and
goal revision by the session owner. `oxa.a2a.enabled: false` in the user settings turns the lane
off. A way for another user, a project file or a model to send or authorise a control request is
high severity.
Automated Go review reads at most 200 GitHub comment pages (20,000 comments) when it checks the
configured daily quota and prior same-diff receipts. If either listing is incomplete, review is
deferred before quota reservation, provider authentication, or provider I/O. An incomplete history
cannot authorize another token-spending review.

The memory store (`memory/` under the agent directory) is owner-only: folder `0700`, tightened by a
save when an existing one is looser; entries and `MEMORY.md` `0600` via a private synced temporary;
lock created `0600`. A save is bounded (8 KiB body, 4 KiB frontmatter) and the index scan reads each
file only that far.

Treat as **medium**: missing bounds on untrusted input (sizes, lengths, recursion), error messages that echo provider text, weakened validation of provider usage/billing fields.

Do not report: style, pre-existing issues outside the diff, test-only code that never runs in production, or hypothetical risks without a concrete path.

Image input (B2b): `read` sends an image to a model that takes images only through the same pinned, private-path-checked open as text, when the engine says it takes images, and when `images.blockImages` is off. A project may enable that opt-out but cannot disable the owner's user setting. The user-only `oxa.read.maxImageBytes` (default 20 MiB, `0` = no limit) bounds both an individual image and the aggregate images across retained session history and the current tool-call batch; a result that would cross the limit is not sent and names the limit. An image read that skips the pinned open or the private-path rules is high severity.

The bound, live Lead owner service has one execution surface: the `worker` tool can create a Git
worktree and launch or steer child `oxa` processes. Its supervisor lives for the service lifetime,
but each new launch uses the current user-only `oxa.workers.*` limits and permission ceiling. The
read-only shadow preset and the one-shot `oxa lead chat` command expose no worker tool. Setting
`oxa.workers.enabled` to `false` in user settings disables the surface.

## Sandbox standard (owner decision)

- **Interactive CLI:** no OS sandbox by default, on Linux and macOS. Dangerous commands ask the owner.
- **Yolo:** no OS sandbox for the session, nor for any worker, subagent, headless or unattended run it starts. Yolo refuses only publishing secrets and an OS-breaking wipe. A worker started by the `worker` tool without a `mode` inherits the parent's level (yolo parent, yolo worker); an explicit narrower mode wins, a wider one is clamped.
- **Workspace/auto publishing:** Publishing (owner decision 2026-11-19): in workspace and auto, without asking, the session repo's current non-default branch may be pushed to `origin` with an explicit source refspec (no force, no `+refspec`, no delete, no tag or mirror, no other remote), `gh pr create|edit|comment` without a PR argument, repo or head override, and read-only `gh` (`pr view|checks|diff`, `run view|watch`, `api` GET) run. A push with no refspec, a push to the default branch, a force push, branch deletion, `gh pr merge`, a push to another repo and release or tag pushes ask, or go to the lead's approvals desk for a worker with no human (a headless worker without a desk is refused). Every part of a command line must be in the allowed set: any other command chained with one (`curl x && git push`, `git push | tee f`, a substitution, `bash -c`) asks. The classifier asks whenever it cannot prove the case: unknown remote default branch, push-redirecting repo config (`[push]`, `remote.origin.push|pushurl`), a nested repository, a branch switch earlier in the line, `git -c/--config-env/--exec-path`, any `gh` command outside the read-only and own-PR set. Git metadata reads are bounded and refuse symlinks. The session repo's origin URLs (and URL rewrites) are recorded in memory at the first classified command; a push asks once they differ. The `bash -c` check is a conservative text heuristic, a known limit. Yolo is unchanged.
- **Unattended runs without yolo:** keep bwrap by default.
- **In-process guards in every mode:** secret masking, private-path denials, user-only settings, offline mode.
- **Re-adding an OS sandbox to interactive or yolo paths** is a regression, not a hardening.

## Owner decisions (not findings)
The owner confirm broker answers a typed yes only for a request whose dialog was shown, and a consumer started by the model verifies the broker before it accepts an answer, by owner decision (2026-11-19, post-merge audit of #1572). A finding that asks for the old behaviour (a typed yes answering an unshown request, or a consumer trusting `OXA_CONFIRM_SOCK` as given) is answered here.
A background bash job that finishes on its own queues one follow-up message with the command as written, its exit code and last lines, by owner decision (#1448 W4): `oxa.jobs.wakeOnFinish` (default `true`, `false` opts out, user settings only so a project file cannot turn it on) is the control, and it wakes only the session the job started in. A job the model or the user stopped wakes nothing, and a completion retained behind a failed or interrupted turn stays queued, also across a session reopen, until the next prompt. A finding that asks for a default-off switch, for the output to stay silent, or for a cross-session wake is not a finding; it is answered here.
Limits warn and record instead of blocking, by owner decision (2026-11-19). A goal token budget (`--max-tokens`) counts known usage only: a request with unknown usage (a reply without usage, a failed cloud request) is recorded as `unknown_requests` with one warning, and is not an exhausted budget; the owner's time and turn budgets still stop the goal. Two replies without a tool call give one warning and the goal keeps pursuing; `oxa.goal.idlePause` (default `false`, user settings only) restores the pause. `oxa.toolOutputCap`, `oxa.subagent.resultCap` and `oxa.subagent.softTokens` scale with the window of a cloud model when the owner left them unset (the local profile and owner values are unchanged). A finding that asks for these to refuse, stop or stay fixed is answered here.
A pursuing goal whose request fails transiently (stream cut, engine down or busy, timeout, 429, 5xx) waits without a model request and then sends the same request again, by owner decision (#1448): `oxa.goal.failureBackoffSeconds` (default 60, doubling, capped at 900 unless set higher, `0` opts out) is the only control, and there is no attempt cap. Auth errors, refusals, a repetition stop, an abort and an owner-set budget still stop the goal with the reason shown. A finding that asks for an attempt cap, a default-off switch or a new terminal exception is not a finding; it is answered here.
An extension's `oxa.exec` with a `cwd` outside the session directory asks like any other access outside the project (a write when the command is writable, a read otherwise), by owner decision (2026-11-19, the v0.2.0 gap from the post-merge audit): the directory is mounted into the sandbox, so it needs the same approval as a file write there. Headless without an approver is refused; yolo never asks; a `cwd` inside the session directory is unchanged. A finding that asks for a silent mount, or for a block in yolo, is answered here.

Board evidence limits degrade only the affected pull request. A connection that cannot be read
completely produces a visible, ineligible row and cannot trigger review, landing, or refill actions;
the other rows remain usable. An over-page merge queue retains its observed entries, while every
unobserved pull request has unknown queue state and cannot trigger a queue action.

A provider key opens the native cloud gateway by owner decision (#1823 follow-up, 2026-11-19): an OpenRouter, OpenAI or Anthropic key in the owner's protected key file (or a ChatGPT sign-in) enables the in-process gateway and model discovery without `oxa.engine.inProcessCloud` or `oxa.cloud.discover_models`, and an explicit owner value (`false` opts out) always wins. There is no built-in model table: the model list is fetched live from each keyed provider with the owner's key, and limits and prices the provider does not list come from OpenRouter's public catalog (`GET https://openrouter.ai/api/v1/models`), a keyless request that sends no key and no user data, used only for a provider's default endpoint and never when the owner points discovery at a proxy; the last successful fetch is cached on disk, an unknown limit gets a conservative default and an unknown price is recorded as uncertain spend, never a refusal. An owner entry under `oxa.cloud.models` always wins. This is still user-only: the decision reads the user layer and the key store, a project file or a restored session can never enable a provider, raise a limit or change the transport, and offline mode refuses. The ChatGPT plan's model list follows the same rule (owner-approved 2026-11-19): with a sign-in present, the gateway sends one `GET <chatgpt base>/models?client_version=...` with the existing ChatGPT bearer token and account header (the request the Codex CLI makes), through the same locked refresh path as a chat request, only to the ChatGPT base the route already uses; the token is never logged, only model ids and context windows are read from the reply, a reply that echoes a credential is refused, and the request is pinned to the account the cache key was derived from (a different sign-in is refused unsent), a `401 token_expired` refreshes once through the locked path and retries once (a second failure falls back to the cache), the cache is bound to the account and keeps no token, and an owner's own `oxa.cloud.models` ChatGPT route is kept even when the list omits it. ChatGPT `oxa.cloud.subscription_limits` are advisory like Go's: unset means no check, a configured value warns at 80% and 100% and is recorded, and nothing is refused locally for it (`subscription_limits_not_configured` is gone). A finding that asks for an explicit enable switch, a required cap for a plan route or a refusal on a crossed advisory limit is not a finding; it is answered here.
Limits are owner settings that warn and record. A finding that asks for a default block, a built-in cap or a second consent question is answered here. These decisions bind code review too; `AGENTS.md` points here instead of repeating them.

- **Off Linux** there is no OS sandbox. In-process guards stay, bash gets the same cleared base environment, a refused bash network disables it, and a run that keeps the network says so at start and to the model.
- **No configured cap, no cap check,** for the owner and for authenticated allowlisted share guests. Guest authority, tokens and request quotas still apply.
- **OpenCode Go** admission reserves subscription tokens only, including when the provider falls back to its paid balance. `oxa.cloud.daily_budget_usd` applies to metered API admission, not to Go admission. Provider-reported Go charges are recorded in the ledger and included in the shared daily spend. A configured `oxa.cloud.subscription_limits` is advisory: it warns at 80% and 100% and is recorded, and it is never a local refusal.
- **Keyed Go:** a Go key in the owner's own secrets file enables the Go subscription transport for the owner's own model choice, with `oxa.cloud.opencode.enabled: false` as the opt-out. Transport selection stays user-only: the protected user key authorizes an actual Go model argument, including one passed to a delegated `oxa work` worker or nested child. No project-selected or restored model enables it without an explicit `oxa.cloud.opencode.enabled: true`; a local alias owns a request only on an exact match, and a model scope enables Go only when it names a Go model.
- **Cargo cache:** the sandbox can forward the owner's `CARGO_TARGET_DIR` to worker shells as a toolchain location. External cache paths are writable only through an explicit user-only `oxa.sandbox.writePaths` mount. Review environment forwarding and writable mounts independently.
- **Headless inbox:** persisted headless print and JSON runs with a pursuing goal consume plain `oxa send` inbox messages with `oxa.inbox.headless` set to `true` by default. This user-only setting can opt out; project settings cannot enable or disable it. Non-persisted sessions, sessions without a pursuing goal, and the managed-steering path do not mount the plain inbox watcher.
- **Bash and subagent limits:** `oxa.bash.maxJobs` (default 16 running background jobs) and `oxa.subagent.maxChildren` (default 8 running or queued children per parent, and 8 steps per call) are read from the user file and have a minimum of 1. Project settings for remote concurrency remain bounded by the user's child cap. An explicit bash `timeout: 0` disables the timeout for that call, including a background job. Sandbox and permission checks still apply.
- **Automated Go review** reads at most 200 GitHub comment pages (20,000 comments) when it checks the configured daily quota and prior same-diff receipts. If either listing is incomplete, review is deferred before quota reservation, provider authentication, or provider I/O. An incomplete history cannot authorize another token-spending review.
- **`oxa work` acceptance:** a task with zero or skipped checks remains `Unverified` and lacks a publishable acceptance receipt. Project files and worker replies cannot change the owner-only `oxa.work.localChecks` setting, which defaults to per-package `cargo check`. Rust acceptance requires hosted test evidence for the exact head, including for changed inputs outside packages in a tree with Cargo manifests. A worker declaration can add requirements to that evidence; review their coverage against the task's acceptance criteria.
- **Repeat guard** (#1448): the third identical tool result gets a notice, a call that still repeats gets one persisted warning, and the run is not stopped. `oxa.maxTurns` is the configured cap against loops the guard cannot see.
- **Board evidence limits** degrade only the affected pull request. A connection that cannot be read completely produces a visible, ineligible row and cannot trigger review, landing, or refill actions; the other rows remain usable, and the limit is logged. An over-page merge queue retains its observed entries, while every unobserved pull request has unknown queue state and cannot trigger a queue action.
- **Report dedupe is one consent** (#1448): opting into `oxa.reportIssue.semanticDedupe` is the owner's consent to send the scrubbed report, a public source excerpt and candidate summaries to the configured model and to search the repository for matching issues, so the post asks one question: the final preview. New issues are limited to one per task. The per-worker daily count is the owner setting `oxa.reportIssue.dailyNew`: unset means no check, and at or over a configured cap the preview warns, the ledger records it and the owner's yes still posts.
Intentional removals must be stated in the PR body with a `Removes:` line; flag silent deletion of security checks as high.

The owner confirm broker (`oxa-core/src/owner_confirm.rs`, `OXA_CONFIRM_SOCK`) answers an oxa
confirmation only from the owner: yolo chosen for the session (answer yes, shown and recorded), the
owner's answer to the session's own dialog, or the owner's typed message while a confirmation waits.
A typed yes answers only a confirmation whose dialog oxa has already shown to the owner (the broker
records shown requests; a yes with no shown dialog, or for a different request id, does not count).
A consumer a model-run command starts cannot tell a fake listener from the broker by the socket
path, because the command controls its environment: it trusts only the sandbox's pinned mount point
or a listener verified through the peer's credentials as the session's own oxa process (an ancestor
of the consumer running the same executable as the same user). A consumer that accepts an answer
from an unverified socket is high. In yolo nothing changes: the model already has the owner's rights
and the broker answers yes. The question text is the requesting process's own words and is shown as
data, marked as requested by the process. Off Linux the peer check is unavailable and a consumer
trusts only the sandbox's pinned mount.
A confirmation that a model-run command can satisfy without the broker is high. A broker answer
derived from tool, file, web or agent-inbox text is high. A socket readable by another user is high.
The broker bounds what a request can hold in the oxa process before accepting it for confirmation:
its size, the time to deliver it and the connections served at once (owner-configurable, generous
defaults: 8 MiB, 60 s, 64). An unbounded buffer, read or thread per request is high. Inside the
Linux sandbox the broker socket is mounted only where bwrap denies nested user namespaces
(`--disable-userns`), so a command cannot stack a fake listener over the trusted mount point; a
broker mount without that denial is high. Publishing hands `gh` the approved bytes through a sealed
descriptor (`--body-file /dev/fd/N`), never a pathname a command could swap after the yes.

## Sandbox standard (owner decision)

- **Interactive CLI:** no OS sandbox by default, on Linux and macOS. Dangerous commands ask the owner.
- **Yolo:** no OS sandbox for the session, nor for any worker, subagent, headless or unattended run it starts. Yolo refuses only publishing secrets and an OS-breaking wipe. A worker started by the `worker` tool without a `mode` inherits the parent's level (yolo parent, yolo worker); an explicit narrower mode wins, a wider one is clamped.
- **Unattended runs without yolo:** keep bwrap by default.
- **In-process guards in every mode:** secret masking, private-path denials, user-only settings, offline mode.
- **Settings fail safe:** a user settings file that exists but cannot be read or parsed keeps offline on, bash network off, remote engines off and new projects untrusted, and warns the owner. Only a missing file or a valid file changes this.
- **Re-adding an OS sandbox to interactive or yolo paths** is a regression, not a hardening.
