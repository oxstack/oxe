# Security

## Reporting
Please report vulnerabilities privately through GitHub Security Advisories ("Report a vulnerability" on this repository). Do not open public issues for security problems.

## Review guidelines (used by automated security review)
Read `docs/threat-model.md` first. Report a finding only if this change weakens one of its trust boundaries; explain the trigger and the impact, and cite the changed lines. The sandbox standard and the owner decisions below are part of the standard, and so are the review guidelines in `AGENTS.md`: a finding that asks for the opposite is not a finding.

Treat as **high** severity:
- a secret (API key, ChatGPT token, OpenCode/GitHub token, share key) printed, logged, stored outside its 0600 store, sent to the wrong host, or reachable by model tools;
- a project file or model output able to change user-only settings, enable a feature flag, or bypass the sandbox or network guard (off Linux there is no OS sandbox by owner policy: in-process guards stay, bash gets the same cleared base environment, a refused bash network disables it in unattended runs and offline mode and warns in interactive sessions, an attended interactive Linux session without bwrap can run model bash unsandboxed with a warning only when no sandbox was requested; plan commands and non-yolo project hooks still require bwrap, and a run that keeps the network says so at start and to the model);
- metered API requests that bypass required reservation or settlement, are missing from the ledger, or bypass a **configured** daily cap. With no configured cap there is no cap check for the owner or authenticated allowlisted share guests. Guest authority, tokens and request quotas still apply;
- OpenCode Go requests that bypass subscription token accounting, or provider-reported paid charges that are missing from the ledger or shared daily spend. Go admission reserves subscription tokens only. The owner's local metered API cap does not apply to Go admission, including provider-side fallback to a paid balance. A configured `oxa.cloud.subscription_limits` is advisory (warn at 80%/100%, recorded, never a local refusal), and a stored owner Go key enabling the Go transport is owner policy: the protected user key authorizes an actual Go model argument, including one passed to a delegated worker or nested child. A project setting or restored session model that flips it *is* a finding (the owner's explicit `oxa.cloud.opencode.enabled: true` is the supported way to allow those);
- fork-PR or guest access to secrets, cloud providers or owner credentials;
- new `unsafe`/FFI/process/network code in the engine without bounds checks.

Skill reactivation after compaction is a user-only opt-in. A project setting cannot add
the skill reactivation instructions to later turns.

Dangerous-command rules read one parsed `Effects` value per shell command (launchers unwrapped, the directory
chain of `env -C`, files read, token-printing commands). `credential-read` and `git-hook-write` are approval
prompts in standard mode, not a sandbox; yolo does not reach them. A saved unattended grant never covers a
pipeline fed by anything but a literal `echo`/`printf`, or an input redirect. A `wait` probe that runs `gh`
is network access (plan denies, ask asks once per session).

Saved dangerous-command grants cover a reviewed literal command. Shell parameter or command
substitution markers in any stage make saved grants inapplicable, including markers before a hit.
The native cloud gateway's default recommendations use only configured API-key providers
and their standard endpoints. User model entries take precedence, and the owner can opt out.
Recommendations do not initiate network discovery or confer guest access.

Attended bash can run with host rights under the owner’s interactive sandbox policy. Host tool
shells keep the cleared toolchain environment. Their live and captured output masks exact copies
of known GitHub credentials (all CLI accounts and hosts), the configured engine bearer key,
and nonempty values from oxa's environment or the command spec whose names contain `KEY`,
`SECRET`, `TOKEN`, or `PASSWORD` (case insensitive). This also covers yolo bash and portable host
execution, without changing sandbox or network policy. Output masking is not isolation.
Unattended runs include ownerless RPC and headless runs pursuing a goal. Exec hooks
keep their dedicated sandbox. Network-disabled model shells mask host runtime IPC.
Where a command runs is one decision (`oxa-sandbox/src/decision.rs`, `ExecPolicy`). Exec hooks run in the sandbox: on Linux they are refused without bwrap, never run on the host instead; under yolo and off Linux they run with your rights, and `--offline` still holds. An attended interactive session with no OS sandbox requested and no usable bwrap keeps `oxa.sandbox.net: off` as a start warning only. The output of every command that runs with your rights has the host credentials scrubbed at one sink, on every path, and the GitHub CLI credential discovery for that scrub never gives `gh` the network the owner turned off.
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
Inbox child creation, publication, scanning and privacy wipes stay relative to opened
no-follow directory descriptors on Linux and macOS. A replacement pathname must not
redirect these operations.

Persisted headless print and JSON runs with a pursuing goal consume plain `oxa send` inbox
messages with `oxa.inbox.headless` set to `true` by default. This user-only setting can opt out;
project settings cannot enable or disable it. Non-persisted sessions, sessions without a pursuing
goal, and the managed-steering path do not mount the plain inbox watcher.

An explicit owner CLI `--auto` or `--yolo` is sufficient startup consent for home, root, or a
parent of home, with a warning. Project settings and model output cannot supply that CLI consent.
An interactive time or token limit ends the watched startup session after its bounded wrap-up,
and an idle expiry starts no model request.
The background-job and per-parent child limits warn and record by default, by owner decision
(2026-11-19, refined 2026-11-20): `oxa.bash.maxJobs` (default 16) and `oxa.subagent.maxChildren`
(default 8). Above either, the start goes ahead (extra children queue) and the tool result carries a
warning that names the setting, shown as a notice and recorded in the session. The owner can make either a
refusal with the user-only `oxa.onExceed` (`bashJobs`, `subagentChildren`: `stop`). Both limits, `oxa.onExceed`,
`oxa.tools.wait` and `oxa.toolOutputCap` are read from the user file, so a project cannot change them.
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

Before `oxa work` publishes or pushes a lane candidate, the runner scans the exact commits added
to the destination for credential-like content and refuses the push on a finding. Lane workers do
not receive GitHub credentials, and a lane candidate may not modify GitHub workflows or the lane
runner itself; those controls must land separately through the normal review path.

The repeat guard warns and continues by owner decision (#1448): the third identical tool result gets a notice, a call that still repeats gets one persisted warning, and the run is not stopped. A finding that asks for the old stopping behaviour is not a finding; `oxa.maxTurns` is the configured cap.
The lead web page can instead attach to one running owner session (#2075, `oxa.lead.attachSessionId`, user settings only). The id comes only from owner-private user settings, never from a project file or an HTTP request. The service reads the newest part of that session's file without its writer lock and sends the owner's text only through the session's inbox, the same write `oxa send` makes; it takes no lock, opens no provider and appends nothing. The identity, origin and CSRF checks are unchanged, the file and its directories must stay owner-only and are re-proved before every read and send, and transcript text is shown only to the verified owner, escaped, with credentials in tool arguments and command previews masked. An inbox message is a follow-up for the model, never an approval: approvals stay on the approvals page.
The bound, live Lead owner service has one execution surface: the `worker` tool can create a Git
worktree and launch or steer child `oxa` processes. Its supervisor lives for the service lifetime,
but each new launch uses the current user-only `oxa.workers.*` limits and permission ceiling. The
read-only shadow preset and the one-shot `oxa lead chat` command expose no worker tool. Setting
`oxa.workers.enabled` to `false` in user settings disables the surface. A trusted live Lead outside
Git can start workers only in the primary checkout named by the owner's `oxa.workers.repository`,
and only while that checkout is explicitly trusted. Neither project settings nor model arguments
can select it. The normal permission check covers both the source and the new worktree; the
launch still uses the same child permission ceiling and sandbox policy.

Dangerous commands are a floor prompt in the interactive run, and an unattended run (a worker, a subagent, a headless goal: no one to ask) never waits for an answer (#1512). `oxa.approval.unattended` is the owner setting, user settings only: `auto` (the default) refuses the command under the OS sandbox and files an approval request without one, `refuse` always refuses, `request` always files a request. Refusals are returned without writing an approval record, so repeated model-controlled commands cannot grow the desk. Requests live in `<agent dir>/approvals/` (directory 0700, one record per request at 0600, written atomically under a private temporary name, never following a symlink or an agent directory owned by another user) and are answered only by the owner through `oxa send <session> --approve <id> once|session|always|deny`, authenticated through the session's owner-confirm broker (or a direct terminal confirmation), or on the lead web approvals page, never by inbox text or model output. A `once` grant is consumed by the run that retries the command; `session` grants last only for the run that filed the request, including when the persisted session is resumed. `always` is recorded in the user-only `oxa.approval.always` list: its desk record also holds only in the run that was answered, and later runs read the list, so removing a line there revokes it. A command that reads its input from a file or stdin (an input redirection, `xargs`, `read`, `--files-from`) is refused unattended like shell input, and no saved grant covers it, because the reviewed text does not fix what it acts on. A grant is bound to the canonical directory it was reviewed in, a wipe takes the desk lock, the newest 500 `refused` and 500 `pending` records are kept (older ones of that status rotate out; answered grants never do), and listings escape control, bidi and zero-width format characters. A finding that asks for a default-off switch, for an unattended run to wait indefinitely, or for the desk to take an answer from a model or a project file is answered here.

## Loopback surfaces

Each loopback listener that can start a model request or spend cloud budget needs a client proof beyond the same-user peer. The in-process cloud gateway has no listener.

| Listener | Client proof |
| --- | --- |
| share (`oxa share`) | guest bearer |
| a2a server | capability token |
| oxe-server (started by oxa) | `OXE_API_KEY_FILE` key |
| `oxa lead serve`, local mode | per-start client token, cookie or bearer (#193) |

`oxa lead serve` mints a 256-bit token at start, prints `http://127.0.0.1:<port>/?t=<token>` once on the owner's foreground stdout (never the journal), and writes it to `<agent dir>/lead-client-token` (directory 0700, file 0600, replaced on every start, removed on clean exit). A request without it, page or POST, is refused with 401 and the fixed journal fact `reason=client_token`. Limit: under the interactive no-sandbox default, a same-user command that deliberately reads the agent directory can still obtain the token. The token stops the silent, accidental and scripted path (a `curl` to a known port) and every sandboxed or unattended run; no OS sandbox is added to close the rest.

## Sandbox standard (owner decision)

- **Interactive CLI:** no OS sandbox by default, on Linux and macOS. Dangerous commands ask the owner.
- **Yolo:** no OS sandbox for the session, nor for any worker, subagent, headless or unattended run it starts. Yolo refuses only publishing secrets and an OS-breaking wipe. A worker started by the `worker` tool without a `mode` inherits the parent's level (yolo parent, yolo worker); an explicit narrower mode wins, a wider one is clamped.
- **Workspace/auto publishing:** Publishing (owner decision 2026-11-19): in workspace and auto, without asking, the session repo's current non-default branch may be pushed to `origin` with an explicit source refspec (no force, no `+refspec`, no delete, no tag or mirror, no other remote), `gh pr create|edit|comment` without a PR argument, repo or head override, and read-only `gh` (`pr view|checks|diff`, `run view|watch`, `api` GET) run. A push with no refspec, a push to the default branch, a force push, branch deletion, `gh pr merge`, a push to another repo and release or tag pushes ask, or go to the lead's approvals desk for a worker with no human (a headless worker without a desk is refused). Every part of a command line must be in the allowed set: any other command chained with one (`curl x && git push`, `git push | tee f`, a substitution, `bash -c`) asks. A later sequential repository mutation does not change an earlier publish (`git push && git switch main`), but a repository mutation anywhere in the same concurrent pipeline asks because it can race the publish (`git push | git switch main`). The classifier asks for an unresolved Git subcommand (a name that is not a builtin recognized by the classifier, so it may be a configured alias or an external `git-*` executable), and whenever it cannot prove the case: unknown remote default branch, push-redirecting repo config (`[push]`, `remote.origin.push|pushurl`), a nested repository, a branch switch earlier in the line, `git -c/--config-env/--exec-path`, any `gh` command outside the read-only and own-PR set. Git metadata reads are bounded and refuse symlinks. The session repo's origin URLs (and URL rewrites) are recorded in memory at the first classified command; a push asks once they differ. The `bash -c` check is a conservative text heuristic, a known limit. Yolo is unchanged.- **Unattended runs without yolo:** keep bwrap by default.
- **Script filenames:** In a direct shell invocation such as `sh scripts/git-push.sh`, a literal script filename is not inline `git push` text. Normal mode permissions still apply. The publish floor still checks the remaining command line, and dangerous-command checks are unchanged.
- **In-process guards in every mode:** secret masking, private-path denials, user-only settings, offline mode.
- **Settings fail safe:** a user settings file that exists but cannot be read or parsed keeps offline on, bash network off, remote engines off and new projects untrusted, and warns the owner. Only a missing file or a valid file changes this.
- **Re-adding an OS sandbox to interactive or yolo paths** is a regression, not a hardening.

## Owner decisions (not findings)

The following decisions set the policy for current review and planned changes; they do not claim that the open issues are implemented:

- **Project trust (#1441):** Ask once per folder by default. Offer per-content re-approval as an opt-in; it is not required by default.
- **Privileged actions (#784):** Only the owner can configure named root actions that may run without a password. For other sudo commands, the owner enters the password directly through a separate terminal or secure owner-only input that passes it straight to sudo. The password never reaches the model, prompt, transcript, tool output, or logs.
- **Release payload (#1394):** Keep standalone `oxe` in the release allowlist for other inference clients and LAN sharing without an `oxa` agent.
- **Scripts (#2020):** Running a script follows the normal permission mode. Do not add a special confirmation merely because a script might publish. Yolo does not ask, auto runs within its existing rights, and standard applies its usual prompts. The existing secret and publication rules for direct commands still apply.
- **ChatGPT subscription limits (#1909):** Configured limits warn and record; they do not stop a request locally. The provider enforces its own limits.
- **Lead web approvals (#2117):** When only the owner can reach the UI over Tailscale, do not add a second presence check or passcode. Revisit this if the UI becomes reachable by others.
- **Approval and sandbox state (#1653, #1665):** An authenticated owner approval counts with or without an OS sandbox, whether given in the terminal, through `oxa send`, or in the owner-only Lead web UI over Tailscale. The sandbox is defense in depth, not a condition for approval. Record whether the sandbox was active; existing authentication and request binding still apply.
- **Secret masking without an OS sandbox (#1658):** Mask values oxa injects as secrets, such as provider keys and worker GitHub tokens, plus values of environment variables named `*KEY*`, `*SECRET*`, `*TOKEN*`, or `*PASSWORD*`. This is the scope of unsandboxed output masking, not a general secret detector or a change to the publication scanner.

The owner confirm broker answers a typed yes only for a request whose dialog was shown, and a consumer started by the model verifies the broker before it accepts an answer, by owner decision (2026-11-19, post-merge audit of #1572). A finding that asks for the old behaviour (a typed yes answering an unshown request, or a consumer trusting `OXA_CONFIRM_SOCK` as given) is answered here.
A background bash job that finishes on its own queues one follow-up message with the command as written, its exit code and last lines, by owner decision (#1448 W4): `oxa.jobs.wakeOnFinish` (default `true`, `false` opts out, user settings only so a project file cannot turn it on) is the control, and it wakes only the session the job started in. A job the model or the user stopped wakes nothing, and a completion retained behind a failed or interrupted turn stays queued, also across a session reopen, until the next prompt. A finding that asks for a default-off switch, for the output to stay silent, or for a cross-session wake is not a finding; it is answered here.
Limits warn and record by default, and the owner chooses a harder action, by owner decision (2026-11-19, refined 2026-11-20): the user-only `oxa.onExceed` sets `warn`, `pause` or `stop` per budget. Jobs and subagent children default to `warn`. A goal token budget (`--max-tokens`) counts known usage only, and a request with unknown usage (a reply without usage, a failed cloud request) is recorded as `unknown_requests`; under a goal token cap it pauses the goal with a recorded reason by default (`goalUnknownUsage`), `stop` blocks it and `warn` keeps the old warning only. A goal with no other budget pauses after two replies without a tool call by default (`goalIdle`), also recorded; `warn` only warns, and a goal that only warns gets one nudge and then waits itself between requests (no request, recorded), `oxa.goal.idleBackoffSeconds` (default 60, doubling, capped at 900 unless set higher, `0` turns the wait off, user settings only). A run that fails keeps its goal pursuing and records a warning with the error code; `/goal resume`, the a2a resume or a headless `--continue` continue it in the same session. Subscription limits stay warnings. `oxa.toolOutputCap`, `oxa.subagent.resultCap` and `oxa.subagent.softTokens` scale with the window of a cloud model when the owner left them unset (the local profile and owner values are unchanged). A finding that asks for these to refuse, stop or stay fixed is answered here.
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
- **Lane coverage:** what a green lane run counts as evidence for is data in `tools/ci/lane_coverage.json`, read by the Rust guard from the trusted base commit and never from the pushed head. A missing or invalid file covers nothing. The workflow and `tools/ci` files must also match the destination's base branch from the merge base.
- **Worker build cache:** a worker derives one writable path for itself, `<cache dir>/<hash>/cargo-target` per main repository, created 0700 and refused if a symlink or owned by another user. The settings are user-only. A held native worker runs online with runner-owned GitHub isolation (no credential, adapters or MCP, also in yolo) instead of being forced offline.
- **Headless inbox:** persisted headless print and JSON runs with a pursuing goal consume plain `oxa send` inbox messages with `oxa.inbox.headless` set to `true` by default. This user-only setting can opt out; project settings cannot enable or disable it. Non-persisted sessions, sessions without a pursuing goal, and the managed-steering path do not mount the plain inbox watcher.
- **GitHub updates, `oxa work` inbox and publish receipts:** an `oxa work` worker (`OXA_WORK_INBOX=1`) also watches its inbox, under the same user-only `oxa.inbox.headless` opt-out. Text the runner fetches for a worker (`oxa work` packet updates) and inbox messages are context, never an approval. The runner fetches; the worker stays GitHub-isolated. After a push the runner reads the destination back (`git ls-remote`) and records what it holds with a divergence class; it never force-pushes.
- **Bash and subagent limits:** `oxa.bash.maxJobs` (default 16 running background jobs) and `oxa.subagent.maxChildren` (default 8 running or queued children per parent, and 8 steps per call) are read from the user file and have a minimum of 1. Above them a call warns and records by default; `oxa.onExceed` (user-only) turns that into a refusal per budget. `oxa.tools.wait` and `oxa.toolOutputCap` are user-only too. Project settings for remote concurrency remain bounded by the user's child cap. An explicit bash `timeout: 0` disables the timeout for that call, including a background job. Sandbox and permission checks still apply.
- **Automated Go review** reads at most 200 GitHub comment pages (20,000 comments) when it checks the configured daily quota and prior same-diff receipts. If either listing is incomplete, review is deferred before quota reservation, provider authentication, or provider I/O. An incomplete history cannot authorize another token-spending review.
- **`oxa work` acceptance:** a task with zero or skipped checks remains `Unverified` and lacks a publishable acceptance receipt. Project files and worker replies cannot change the owner-only `oxa.work.localChecks` setting, which defaults to per-package `cargo check`. Rust acceptance requires hosted test evidence for the exact head, including for changed inputs outside packages in a tree with Cargo manifests. A worker declaration can add requirements to that evidence; review their coverage against the task's acceptance criteria.
- **Worker GitHub access:** the owner-only `oxa.work.workerGithub` setting decides (a project value is ignored): unset, yolo workers keep their own GitHub environment by owner policy (a finding that calls that a defect is not a finding), auto workers get no token (host-side tools follow), standard workers get the runner's packet only.
- **Repeat guard** (#1448): the third identical tool result gets a notice, a call that still repeats gets one persisted warning, and the run is not stopped. `oxa.maxTurns` is the configured cap against loops the guard cannot see.
- **Board evidence limits** degrade only the affected pull request. A connection that cannot be read completely produces a visible, ineligible row and cannot trigger review, landing, or refill actions; the other rows remain usable, and the limit is logged. An over-page merge queue retains its observed entries, while every unobserved pull request has unknown queue state and cannot trigger a queue action.
- **Report dedupe is one consent** (#1448): opting into `oxa.reportIssue.semanticDedupe` is the owner's consent to send the scrubbed report, a public source excerpt and candidate summaries to the configured model and to search the repository for matching issues, so the post asks one question: the final preview. New issues are limited to one per task. The per-worker daily count is the owner setting `oxa.reportIssue.dailyNew`: unset means no check, and at or over a configured cap the preview warns, the ledger records it and the owner's yes still posts.
Intentional removals must be stated in the PR body with a `Removes:` line; flag silent deletion of security checks as high.

The rolling dev release always requires its reproducible Linux artifacts. Its native macOS arm64
archive is best effort, so a failed macOS build cannot suppress the Linux release. When the macOS
archive is present, the workflow includes it in the versioned SHA256 checksum file and in the same
build-provenance attestation as the other dev assets before publishing it.
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
The broker's state machine, bounds and records do not depend on the extension host pump. A request is a structured, length-prefixed frame (title and body fields); an over-limit length is refused unread with a `reason` (`too_large`, `not_servable`, `closed`, `busy_limit`, `bad_request`). The session thread's `serve` picks a request, creates its per-dialog cancel ticket and marks it shown in one step under the broker's lock, so a typed yes that arrives before the UI has registered the dialog is neither lost nor able to dismiss another dialog. In yolo a request is answered at once, keeps no pending state and leaves a compact record (hash, size, first title line, a count for identical consecutive requests), never the text; a dialog or typed answer keeps the full text, which the owner paced. A finding that asks for yolo to retain every request body is answered here.
The broker bounds what a request can hold in the oxa process before accepting it for confirmation:
its size, the time to deliver it and the connections served at once (owner-configurable, generous
defaults: 8 MiB, 60 s, 64). An unbounded buffer, read or thread per request is high. Inside the
Linux sandbox the broker socket is mounted only where bwrap denies nested user namespaces
(`--disable-userns`), so a command cannot stack a fake listener over the trusted mount point; a
broker mount without that denial is high. Publishing hands `gh` the approved bytes through a sealed
descriptor (`--body-file /dev/fd/N`), never a pathname a command could swap after the yes.
 Lane-push approval uses this broker when a session launches `oxa work`, including through an
agent child; terminal and ambient-variable compatibility approval is unavailable to agent
processes.
Named cloud providers (`oxa.cloud.providers`) are user-only. A provider selects a distinct non-built-in key entry in the existing protected cloud secret store. The owner's HTTPS API base pins discovery and chat; redirects are not followed. A model list, project setting or per-model override cannot replace the base, protocol, key selector or billing class. Cached lists are bound to the base and credential, and catalog replies that echo the credential are refused. Public metadata lookup sends no key and cannot supply another provider's prices for a named route.
