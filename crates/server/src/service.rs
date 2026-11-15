//! Endpoint logic, independent of the HTTP library: a request (method, path, JSON body) gives
//! either a complete JSON response or a stream of SSE strings.

use crate::api::{self, ChatMeta};
use crate::engine::Engine;
use crate::generate::{GenError, GenParams, GenResult, Partial, Progress, Worker};
use chat::{ChatTemplate, ParserConfig, ServerChatOptions, Session};
use serde_json::{json, Value as Json};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokenizer::{Token, Tokenizer};
use tokio::sync::mpsc;

pub struct Config {
    pub model_name: String,
    pub model_path: String,
    pub chat_opts: ServerChatOptions,
    pub fingerprint: String,
    /// allow /slots POST (save/restore are not implemented yet)
    pub slots: bool,
    /// SX-2: prefill the next request's history form while the client is idle (`--idle-reprefill`)
    pub idle_reprefill: bool,
    /// engine facts for `/props` (Flash: `flash_cache` = the cache tier decision: chosen, reason, preference, margins)
    pub engine_status: Option<Json>,
}

pub struct State<E: Engine> {
    pub cfg: Config,
    pub tok: Arc<Tokenizer>,
    pub tmpl: Option<ChatTemplate>,
    pub worker: Mutex<Worker<E>>,
    pub n_ctx: usize,
    pub n_vocab: usize,
    /// token ids that open a user message in the rendered prompt (ChatML `<|im_start|>user`;
    /// llama-server's message-span delimiter), None if the template has none
    pub user_delim: Option<Vec<Token>>,
    /// SX-2: cancel flag of the pending/running idle re-prefill
    pub idle: Mutex<Option<Arc<AtomicBool>>>,
    /// requests started so far (an idle job that sees a newer request does not start)
    pub reqs: std::sync::atomic::AtomicU64,
    /// idle jobs finished (tests)
    pub idle_done: std::sync::atomic::AtomicU64,
    /// S8: a stream whose client takes no frame for this long is dropped and its generation
    /// cancelled (`--stream-stall-timeout`, env `OXE_STREAM_STALL_SECS`, default 30 s)
    pub stream_stall: std::time::Duration,
    /// LONG-APPEND (v0.1.3 #7): SSE comment interval while a streamed request waits for the worker
    /// or prefills (`--sse-keepalive SECS`, env `OXE_SSE_KEEPALIVE_SECS`, default 15; 0 = off)
    pub sse_keepalive: std::time::Duration,
    /// OpenAI M2: the output limit of a request that names none (`--default-max-tokens`, env
    /// `OXE_DEFAULT_MAX_TOKENS`; default min(16384, n_ctx / 4); 0 = none, the request then runs to
    /// the context end). An explicit client value always wins (-1 = no limit, 0 = H6).
    pub default_max_tokens: i64,
    /// `/props.speculative`: the speculative decoding this engine runs (decided at load)
    pub speculative: Json,
    /// CACHE-PERFECT ECHO records (see `echo.rs`); config from `OXE_ECHO*` (the `--echo*` flags)
    pub echo: Mutex<crate::echo::EchoStore>,
    /// token ids that open an assistant turn (`<|im_start|>assistant`), None if the template has none
    pub asst_delim: Option<Vec<Token>>,
    /// the turn end id (`<|im_end|>`)
    pub im_end: Option<Token>,
    /// CACHE-PERFECT P2: the worker has a state store (writes after each reply)
    pub state_cache: bool,
    /// cancel flag of the pending/running state-store write
    pub persist: Mutex<Option<Arc<AtomicBool>>>,
    /// state-store writes finished (tests)
    pub persist_done: std::sync::atomic::AtomicU64,
    /// v0.1.3: held by a post-reply state write through all its phases (the off-lock middle one
    /// included). The cache endpoints take it first, so they see a store with no write in flight;
    /// generation requests never take it. Writes queue behind each other in reply order.
    pub persist_gate: Mutex<()>,
    /// cancel flag of the running generation (SIGTERM cancels it so the bounded shutdown write runs)
    pub running: Mutex<Option<Arc<AtomicBool>>>,
    /// cancel flag of the pending/running preload (P2b; any request raises it)
    pub preload: Mutex<Option<Arc<AtomicBool>>>,
    /// the echo records' disk directory while the disk budget is 0 (Codex P2/P3 review 2)
    pub echo_dir_off: Mutex<Option<std::path::PathBuf>>,
    /// the last preload's status for `GET /state-cache` (idle|running|done|cancelled|failed|used|none|live)
    pub preload_status: Mutex<Json>,
    /// v0.1.4 (PRELOAD-RACE): long holders of the worker in flight: a request from the moment it
    /// queues for the worker, a state write and an idle job from their spawn, a preload from its
    /// start, each until it is done with the worker. Short holders (status reads, budget / clear,
    /// session routes) are not counted: they always let go, so a preload may wait for them.
    pub work: Arc<std::sync::atomic::AtomicUsize>,
    /// fair time-slicing between generations (`fair.rs`; `OXE_FAIR_QUANTUM_MS`, 0 = off)
    pub fair: crate::fair::Fair,
}

/// One unit of [`State::work`] (dropped = done with the worker).
pub struct WorkGuard(Arc<std::sync::atomic::AtomicUsize>);

impl Drop for WorkGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

pub enum Reply {
    Json(u16, Json),
    /// SSE stream: each item is a complete `data: ...\n\n` frame
    Stream(mpsc::Receiver<String>),
}

fn bad(msg: impl AsRef<str>) -> Reply {
    Reply::Json(400, api::error_json(400, msg.as_ref()))
}

impl<E: Engine> State<E> {
    pub fn new(cfg: Config, tok: Tokenizer, worker: Worker<E>) -> Result<State<E>, String> {
        let bos = tok
            .bos
            .and_then(|t| tok.token_text(t))
            .unwrap_or("")
            .to_string();
        let eos = tok
            .eos
            .and_then(|t| tok.token_text(t))
            .unwrap_or("")
            .to_string();
        let tmpl = match &tok.chat_template {
            Some(src) => Some(
                ChatTemplate::new(src, &bos, &eos).map_err(|e| format!("chat template: {e:#}"))?,
            ),
            None => None,
        };
        let n_ctx = worker.engine.n_ctx();
        let state_cache = worker.states.is_some();
        let n_vocab = worker.engine.n_vocab();
        let user_delim = tok
            .chat_template
            .as_deref()
            .filter(|t| t.contains("<|im_start|>"))
            .map(|_| tok.encode("<|im_start|>user", false, true))
            .filter(|d| d.len() >= 2);
        let asst_delim = tok
            .chat_template
            .as_deref()
            .filter(|t| t.contains("<|im_start|>"))
            .map(|_| tok.encode("<|im_start|>assistant", false, true))
            .filter(|d| d.len() >= 2);
        let im_end = tok.token_id("<|im_end|>");
        let speculative = speculative_json(
            worker.engine.spec_max_depth(),
            worker.engine.spec_fallback(),
        );
        Ok(State {
            cfg,
            tok: Arc::new(tok),
            tmpl,
            worker: Mutex::new(worker),
            n_ctx,
            n_vocab,
            user_delim,
            idle: Mutex::new(None),
            reqs: Default::default(),
            idle_done: Default::default(),
            stream_stall: std::time::Duration::from_secs_f64(
                std::env::var("OXE_STREAM_STALL_SECS")
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .unwrap_or(30.0),
            ),
            sse_keepalive: sse_keepalive_from_env(),
            default_max_tokens: default_max_tokens_from_env(n_ctx),
            speculative,
            echo: Mutex::new(crate::echo::EchoStore::new(
                crate::echo::EchoConfig::from_env(),
            )),
            asst_delim,
            im_end,
            state_cache,
            persist: Mutex::new(None),
            persist_done: Default::default(),
            persist_gate: Mutex::new(()),
            running: Mutex::new(None),
            preload: Mutex::new(None),
            preload_status: Mutex::new(json!({"status": "idle"})),
            echo_dir_off: Mutex::new(None),
            work: Default::default(),
            fair: crate::fair::Fair::from_env(),
        })
    }

    /// A request is starting: an idle re-prefill stops at its next chunk boundary (its partial
    /// state is a prefix of the conversation and stays reusable) or does not start at all.
    pub fn preempt_idle(&self) -> u64 {
        self.preempt_jobs();
        self.reqs.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Raise the cancel flags of the pending background jobs (idle re-prefill, state write,
    /// preload), as a starting request does; also a suspended generation whose turn came back.
    pub(crate) fn preempt_jobs(&self) {
        if let Some(f) = self.idle.lock().unwrap().take() {
            f.store(true, Ordering::Relaxed);
        }
        if let Some(f) = self.persist.lock().unwrap().take() {
            f.store(true, Ordering::Relaxed);
        }
        if let Some(f) = self.preload.lock().unwrap().take() {
            f.store(true, Ordering::Relaxed);
        }
    }

    /// Lock the worker. After a panic in an earlier holder the engine and the cache may disagree:
    /// reset them once and clear the poison flag so later holders reuse the cache normally.
    pub fn lock_worker(&self) -> std::sync::MutexGuard<'_, Worker<E>> {
        self.worker.lock().unwrap_or_else(|p| {
            let mut g = p.into_inner();
            let w = &mut *g;
            w.cache.reset(&mut w.engine);
            self.worker.clear_poison();
            g
        })
    }

    /// Count a long holder of the worker (see [`State::work`]) until the guard drops.
    pub fn begin_work(&self) -> WorkGuard {
        self.work.fetch_add(1, Ordering::SeqCst);
        WorkGuard(self.work.clone())
    }

    fn models(&self) -> Json {
        json!({
            "models": [{
                "name": self.cfg.model_name, "model": self.cfg.model_name, "modified_at": "", "size": "", "digest": "",
                "type": "model", "description": "", "tags": [""], "capabilities": ["completion"], "parameters": "",
                "details": {"parent_model": "", "format": "gguf", "family": "", "families": [""], "parameter_size": "", "quantization_level": ""}
            }],
            "object": "list",
            "data": [{
                "id": self.cfg.model_name, "aliases": [self.cfg.model_name], "tags": [], "object": "model",
                "created": api::now_secs(), "owned_by": "llamacpp",
                "meta": {"vocab_type": 2, "n_vocab": self.n_vocab, "n_ctx": self.n_ctx, "n_ctx_train": self.n_ctx, "max_output_tokens": self.max_output_json()},
                "context_length": self.n_ctx, "max_output_tokens": self.max_output_json()
            }]
        })
    }

    /// The default output limit as published (null = none).
    fn max_output_json(&self) -> Json {
        if self.default_max_tokens > 0 {
            json!(self.default_max_tokens)
        } else {
            Json::Null
        }
    }

    /// `GET /v1/models/{id}`: the OpenAI model object of this server's alias, else 404 model_not_found.
    fn model(&self, id: &str) -> Reply {
        if id != self.cfg.model_name {
            return Reply::Json(
                404,
                api::error_json_with(
                    404,
                    &format!("The model '{id}' does not exist"),
                    "model_not_found",
                    Some("model"),
                ),
            );
        }
        let mut row = self.models()["data"][0].clone();
        if let Some(o) = row.as_object_mut() {
            o.remove("aliases");
            o.remove("tags");
        }
        Reply::Json(200, row)
    }

    fn props(&self) -> Json {
        json!({
            "default_generation_settings": {"params": api::generation_settings(&GenParams { n_predict: self.default_max_tokens, ..GenParams::default() }), "n_ctx": self.n_ctx, "max_output_tokens": self.max_output_json()},
            "total_slots": 1,
            "model_alias": self.cfg.model_name,
            "model_path": self.cfg.model_path,
            "modalities": {"vision": false, "video": false, "audio": false},
            "chat_template": self.tmpl.as_ref().map(|t| t.source.clone()).unwrap_or_default(),
            "bos_token": self.tmpl.as_ref().map(|t| t.bos_token.clone()).unwrap_or_default(),
            "eos_token": self.tmpl.as_ref().map(|t| t.eos_token.clone()).unwrap_or_default(),
            "build_info": self.cfg.fingerprint,
            "is_sleeping": false,
            "engine_status": self.cfg.engine_status.clone().unwrap_or(Json::Null),
            // v0.2.0: the request hints this server implements (a client sends H4 / H6 only when listed)
            // `sampling`: the OpenAI sampler fields this server applies (others are refused or ignored)
            "oxe_hints": {"snapshot_keep": crate::cache::MAX_SNAP, "prefill_only": true, "hint_clear": true,
                          "sampling": ["presence_penalty", "frequency_penalty", "logit_bias"], "count_only": true},
            // speculative decoding in effect: {"active": "draft-mtp"|"lookup"|"none", "depth",
            // "requested", "fallback": why a requested draft-mtp is not running, else null}
            "speculative": self.speculative,
        })
    }

    /// Tokenize a `/completion` prompt: string, token array, or mixed array (first string
    /// part gets add_special).
    fn tokenize_prompt(&self, p: &Json, add_special: bool) -> Result<Vec<Token>, String> {
        match p {
            Json::String(s) => Ok(self.tok.encode(s, add_special, true)),
            Json::Array(a) => {
                let mut out = Vec::new();
                let mut first = true;
                for x in a {
                    match x {
                        // ids above u32 used to wrap onto valid ids (`as`)
                        Json::Number(n) => out.push(
                            n.as_u64()
                                .and_then(|v| Token::try_from(v).ok())
                                .ok_or("bad token")?,
                        ),
                        Json::String(s) => {
                            out.extend(self.tok.encode(s, add_special && first, true));
                            first = false;
                        }
                        _ => return Err("prompt array must contain strings or token ids".into()),
                    }
                }
                Ok(out)
            }
            _ => Err("\"prompt\" must be a string or an array".into()),
        }
    }

    /// Tokenize a rendered chat prompt and find its message starts and assistant turn spans from
    /// the template's own delimiter offsets (CACHE-PERFECT M1, `chat::request::template_marks`):
    /// delimiter text inside a message becomes delimiter tokens too, so it is never counted. The
    /// tokens are exactly `encode(prompt, true, true)`.
    fn render(&self, pc: &PreparedChat) -> Rendered {
        let text = &pc.applied.prompt;
        let marks = match (&self.tmpl, &self.asst_delim) {
            (Some(t), Some(_)) => chat::request::template_marks(t, &pc.inputs, text)
                .ok()
                .flatten(),
            _ => None,
        };
        let Some((starts, ends)) = marks else {
            let tokens = self.tok.encode(text, true, true);
            let (msg_starts, first_user) = match &self.user_delim {
                Some(d) => (
                    tokens
                        .iter()
                        .enumerate()
                        .filter(|(_, &t)| t == d[0])
                        .map(|(i, _)| i)
                        .collect(),
                    tokens.windows(d.len()).position(|w| w == &d[..]),
                ),
                None => (Vec::new(), None),
            };
            let last_user = self
                .user_delim
                .as_ref()
                .and_then(|d| tokens.windows(d.len()).rposition(|w| w == &d[..]));
            let user_starts = match &self.user_delim {
                Some(d) => tokens
                    .windows(d.len())
                    .enumerate()
                    .filter(|(_, w)| *w == &d[..])
                    .map(|(i, _)| i)
                    .collect(),
                None => Vec::new(),
            };
            return Rendered {
                tokens,
                asst: None,
                msg_starts,
                first_user,
                last_user,
                user_starts,
            };
        };
        let mut cuts: Vec<usize> = starts.iter().chain(&ends).copied().collect();
        cuts.sort_unstable();
        cuts.dedup();
        let (tokens, at) = self.tok.encode_cuts(text, true, true, &cuts);
        let tok_at = |b: usize| cuts.binary_search(&b).ok().and_then(|i| at[i]);
        let is = |b: usize, role: &str| text[b + chat::request::IM_START.len()..].starts_with(role);
        let msg_starts: Vec<usize> = starts.iter().filter_map(|&b| tok_at(b)).collect();
        let first_user = starts
            .iter()
            .find(|&&b| is(b, "user"))
            .and_then(|&b| tok_at(b));
        let last_user = starts
            .iter()
            .rev()
            .find(|&&b| is(b, "user"))
            .and_then(|&b| tok_at(b));
        let user_starts: Vec<usize> = starts
            .iter()
            .filter(|&&b| is(b, "user"))
            .filter_map(|&b| tok_at(b))
            .collect();
        let mut asst = Some(Vec::new());
        let n_starts = starts.len();
        for (k, &b) in starts
            .iter()
            .enumerate()
            .filter(|(_, &b)| is(b, "assistant"))
        {
            let next = starts.get(k + 1).copied().unwrap_or(text.len());
            match ends.iter().find(|&&e| e > b).filter(|&&e| e < next) {
                Some(&e) => match (tok_at(b), tok_at(e), asst.as_mut()) {
                    (Some(s), Some(t), Some(v)) => v.push((s, t)),
                    _ => asst = None,
                },
                // the generation prompt: the last opener, no turn end after it
                None if k + 1 == n_starts && pc.inputs.add_generation_prompt => {}
                None => asst = None,
            }
        }
        if starts.len() != msg_starts.len() {
            asst = None;
        }
        Rendered {
            tokens,
            asst,
            msg_starts,
            first_user,
            last_user,
            user_starts,
        }
    }

    /// v0.1.3 pre-think point of a fed chat prompt: thinking on (the generation prompt ends in an
    /// open `<think>`), the position just before that `<think>`, when the prompt ends with exactly
    /// the tokens of that tail. None otherwise (thinking off, another template).
    fn pre_think(&self, prompt: &[Token], generation_prompt: &str) -> Option<usize> {
        let at = generation_prompt.rfind("<think>")?;
        let tail = &generation_prompt[at..];
        if tail.contains("</think>") {
            return None;
        }
        let t = self.tok.encode(tail, false, true);
        (!t.is_empty() && t.len() < prompt.len() && prompt.ends_with(&t))
            .then(|| prompt.len() - t.len())
    }

    /// v0.2.0 H4: the END OF THE LAST MESSAGE in a fed chat prompt, as a token position: the prompt
    /// without the generation prompt the template appends after the last message (the assistant
    /// opener, with `<think>\n` or an empty think block depending on thinking). This is where any
    /// continuation of the conversation (one more message of any role) first differs from this
    /// prompt, so it is the deepest point a client can cut back to. The whole prompt when the
    /// request renders no generation prompt (an H6 prefill, `add_generation_prompt: false`). None
    /// when the prompt does not end with exactly the tokens of that opener (another template shape):
    /// no snapshot is kept and the reply reports `snap_pos: null`.
    fn snapshot_point(
        &self,
        prompt: &[Token],
        generation_prompt: &str,
        added: bool,
    ) -> Option<usize> {
        if !added || generation_prompt.is_empty() {
            return (!prompt.is_empty()).then_some(prompt.len());
        }
        let t = self.tok.encode(generation_prompt, false, true);
        (!t.is_empty() && t.len() < prompt.len() && prompt.ends_with(&t))
            .then(|| prompt.len() - t.len())
    }

    /// ECHO: the prompt to feed for a rendered chat prompt (see `echo.rs`).
    fn splice(
        &self,
        r: &Rendered,
        inputs: &chat::ChatInputs,
    ) -> (Vec<Token>, crate::echo::SpliceReport) {
        match (&self.asst_delim, self.im_end) {
            (Some(_), Some(_)) => self.echo.lock().unwrap_or_else(|p| p.into_inner()).splice(
                &r.tokens,
                &inputs.messages,
                &inputs.reasoning_sent,
                r.asst.as_deref(),
            ),
            _ => (r.tokens.clone(), Default::default()),
        }
    }

    /// ECHO: keep a finished turn (see `echo.rs`). `delivered`: the client received all of the
    /// turn's text (`r.text`); a streamed turn cut at the length limit may have held back a partial
    /// stop word or an unfinished UTF-8 sequence, and such a turn is not kept (V014-SL-1).
    fn echo_record(
        &self,
        prompt: &[Token],
        msg: &chat::ChatMsg,
        r: &GenResult,
        session: Option<&str>,
        delivered: bool,
    ) {
        let (Some(a), Some(e)) = (&self.asst_delim, self.im_end) else {
            return;
        };
        use crate::echo::TurnEnd;
        use crate::generate::StopType;
        let end = if r.stop == StopType::Eos && r.tokens.last().is_some_and(|&t| self.tok.is_eog(t))
        {
            TurnEnd::Eog
        } else if r.stop == StopType::Limit
            && delivered
            && !r.tokens.iter().any(|&t| self.tok.is_eog(t))
        {
            TurnEnd::Length
        } else {
            TurnEnd::Other
        };
        let (_, pending) = self
            .echo
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .record_deferred(prompt, a, e, msg, &r.tokens, end, session);
        // audit v0.1.2 N4: the file append and its fsync run after the store's mutex is released,
        // so the next request's splice never waits for the disk
        if let Some(p) = pending {
            p.write();
        }
    }

    /// HISTORY-REWRITE: keep the history turns this request fed as rendered (see `EchoStore::adopt`);
    /// the file appends run after the store's mutex is released, as for `echo_record`.
    fn echo_adopt(
        &self,
        prompt: &[Token],
        adopt: Option<&(Vec<(usize, usize, usize)>, Json, Vec<Option<String>>)>,
        session: Option<&str>,
    ) {
        let Some((turns, msgs, sent)) = adopt else {
            return;
        };
        let pending = self
            .echo
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .adopt(prompt, turns, msgs, sent, session);
        for p in pending {
            p.write();
        }
    }

    fn check_tokens(&self, toks: &[Token]) -> Result<(), String> {
        if let Some(t) = toks.iter().find(|&&t| t as usize >= self.n_vocab) {
            return Err(format!("token id {t} out of vocab"));
        }
        if toks.is_empty() {
            return Err("empty prompt".into());
        }
        Ok(())
    }
}

/// Route a request. Blocking work (generation) is done in `spawn_blocking`.
pub async fn handle<E: Engine>(
    st: Arc<State<E>>,
    method: &str,
    path: &str,
    query: &str,
    body: &[u8],
) -> Reply {
    handle_session(st, method, path, query, body, None).await
}

/// [`handle`] with the conversation's session hint (`X-Oxe-Session`, forwarded by the
/// controller). A hint only: echo records are found by their token chain, never by this id.
pub async fn handle_session<E: Engine>(
    st: Arc<State<E>>,
    method: &str,
    path: &str,
    query: &str,
    body: &[u8],
    session: Option<String>,
) -> Reply {
    handle_request(st, method, path, query, body, session, None).await
}

/// v0.2.0 H5: `X-Oxe-Turn-Kind` as written to the ledger: at most 32 characters of
/// `[A-Za-z0-9_.-]` (others dropped). A free label (`chat`, `compaction`, `branch_summary`,
/// `prefill`, `subagent`, anything a client uses); it changes nothing else.
pub fn ledger_label(k: &str) -> String {
    k.chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
        .take(32)
        .collect()
}

/// [`handle_session`] with the turn label (`X-Oxe-Turn-Kind`, v0.2.0 H5): a free label copied to
/// the ledger; it places no checkpoint and selects nothing (before v0.2.0 `compaction` placed a cut).
pub async fn handle_request<E: Engine>(
    st: Arc<State<E>>,
    method: &str,
    path: &str,
    query: &str,
    body: &[u8],
    session: Option<String>,
    turn_kind: Option<String>,
) -> Reply {
    handle_policy(st, method, path, query, body, session, turn_kind, false).await
}

/// [`handle_request`] with the controller's state policy (`X-Oxe-State-Policy: ephemeral`, LAN
/// share: a request from another device). An ephemeral generation has RAM-only, request-scoped
/// state (see [`GenParams::ephemeral`]): no echo splice or record, no session hint, no post-reply
/// state write, no idle re-prefill, no prefix reuse; the local live session is held and restored.
#[allow(clippy::too_many_arguments)]
pub async fn handle_policy<E: Engine>(
    st: Arc<State<E>>,
    method: &str,
    path: &str,
    query: &str,
    body: &[u8],
    session: Option<String>,
    turn_kind: Option<String>,
    ephemeral: bool,
) -> Reply {
    handle_hints(
        st,
        method,
        path,
        query,
        body,
        session,
        turn_kind,
        ephemeral,
        Hints::default(),
    )
    .await
}

/// v0.2.0 client-neutral request hints read from headers (HARNESS-RS-PLAN sec. 3; engine API doc
/// `docs/lab-notes/SERVER.md`). A hint never selects arithmetic for the tokens it does not touch and
/// is ignored by a server that lacks it. H6 (`max_tokens: 0`) is a body field, not a header.
#[derive(Clone, Copy, Debug, Default)]
pub struct Hints {
    /// H4 `X-Oxe-Snapshot: keep`: keep a snapshot at the deepest boundary the request processes on
    /// anyway (no new split: the hint never changes the arithmetic) at or before the end of its last
    /// message (chat; the prompt end for `/completion`) and not before that message's start; the
    /// newest 2 per session. The reply reports `timings.snap_pos` (null + `snap_reason` when none).
    pub snapshot_keep: bool,
}

/// [`handle_policy`] with the v0.2.0 hints.
#[allow(clippy::too_many_arguments)]
pub async fn handle_hints<E: Engine>(
    st: Arc<State<E>>,
    method: &str,
    path: &str,
    query: &str,
    body: &[u8],
    session: Option<String>,
    turn_kind: Option<String>,
    ephemeral: bool,
    hints: Hints,
) -> Reply {
    // exact paths like llama-server (REVIEW S4: a trimmed `/v1/chat/completions/` ran as a chat
    // completion while the controller did not treat it as inference)
    let parse = || -> Result<Json, Reply> {
        if body.is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_slice(body).map_err(|e| bad(format!("invalid JSON body: {e}")))
    };
    match (method, path) {
        ("GET", "/health") | ("GET", "/v1/health") => Reply::Json(200, json!({"status": "ok"})),
        ("GET", "/v1/models") | ("GET", "/models") => Reply::Json(200, st.models()),
        ("GET", p)
            if p.strip_prefix("/v1/models/")
                .is_some_and(|id| !id.is_empty()) =>
        {
            st.model(&percent_decode(&p["/v1/models/".len()..]))
        }
        // the OpenAI legacy completions (`text_completion`), built on the native completion path; the
        // engine's own /completion(s) keep llama-server's native shape
        ("POST", "/v1/completions") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            if ephemeral {
                v1_completion(st, b, None, true, false).await
            } else {
                v1_completion(st, b, session, false, hints.snapshot_keep).await
            }
        }
        ("GET", "/props") => Reply::Json(200, st.props()),
        ("GET", "/slots") => {
            if !st.cfg.slots {
                return Reply::Json(
                    501,
                    api::error_json(501, "This server does not support slots endpoint."),
                );
            }
            let busy = st.worker.try_lock().is_err();
            Reply::Json(
                200,
                json!([{"id": 0, "n_ctx": st.n_ctx, "speculative": false, "is_processing": busy}]),
            )
        }
        ("POST", p) if p.starts_with("/slots") => {
            let _ = query;
            Reply::Json(
                501,
                api::error_json(501, "slot save/restore is not supported by oxe-server yet"),
            )
        }
        ("POST", "/tokenize") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            let Some(content) = b.get("content").and_then(Json::as_str) else {
                return Reply::Json(200, json!({"tokens": []}));
            };
            let add_special = b
                .get("add_special")
                .and_then(Json::as_bool)
                .unwrap_or(false);
            let parse_special = b
                .get("parse_special")
                .and_then(Json::as_bool)
                .unwrap_or(true);
            let toks = st.tok.encode(content, add_special, parse_special);
            if b.get("with_pieces")
                .and_then(Json::as_bool)
                .unwrap_or(false)
            {
                let v: Vec<Json> = toks
                    .iter()
                    .map(|&t| {
                        let p = st.tok.piece(t, true);
                        let piece = match String::from_utf8(p.clone()) {
                            Ok(s) => Json::String(s),
                            Err(_) => p.iter().map(|&b| Json::from(b)).collect(),
                        };
                        json!({"id": t, "piece": piece})
                    })
                    .collect();
                Reply::Json(200, json!({"tokens": v}))
            } else {
                Reply::Json(200, json!({"tokens": toks}))
            }
        }
        ("POST", "/detokenize") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            let toks: Vec<Token> = b
                .get("tokens")
                .and_then(Json::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|x| {
                            x.as_u64().map(|v| Token::try_from(v).unwrap_or(Token::MAX))
                        })
                        .collect()
                })
                .unwrap_or_default();
            if let Err(e) =
                st.check_tokens(&toks)
                    .or_else(|e| if toks.is_empty() { Ok(()) } else { Err(e) })
            {
                return bad(e);
            }
            Reply::Json(200, json!({"content": st.tok.decode(&toks, true)}))
        }
        // CACHE-PERFECT P2 state store (the controller's /_local/cache routes forward here)
        ("GET", "/state-cache") => {
            use std::sync::TryLockError;
            let st2 = st.clone();
            // #213: never wait for a running generation (it holds the worker): report busy
            let r = tokio::task::spawn_blocking(move || {
                let _gate = match st2.persist_gate.try_lock() {
                    Ok(g) => g,
                    Err(TryLockError::Poisoned(p)) => p.into_inner(),
                    Err(TryLockError::WouldBlock) => return Err(()),
                };
                let w = match st2.worker.try_lock() {
                    Ok(w) => w,
                    Err(TryLockError::Poisoned(p)) => {
                        // recover with the guard we already hold (locking again would deadlock)
                        let mut g = p.into_inner();
                        let w = &mut *g;
                        w.cache.reset(&mut w.engine);
                        st2.worker.clear_poison();
                        g
                    }
                    Err(TryLockError::WouldBlock) => return Err(()),
                };
                Ok(w.states.as_ref().map(|s| {
                    let mut u = s.usage();
                    u["preload"] = st2.preload_status.lock().unwrap().clone();
                    u
                }))
            })
            .await
            .ok();
            match r {
                Some(Ok(Some(j))) => Reply::Json(200, j),
                Some(Err(())) => Reply::Json(
                    409,
                    api::error_json(409, "busy: a request or state write is running"),
                ),
                _ => Reply::Json(404, api::error_json(404, "state cache is off")),
            }
        }
        // P2b preload (T1-WARM contract): restore a session's newest snapshot into the idle engine
        // in the background; any request cancels it within one block (~5 ms) and it never queues
        // ahead of one (it only runs when it gets the worker at once and no request came since)
        ("POST", "/state-cache/preload") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            if !st.state_cache {
                return Reply::Json(404, api::error_json(404, "state cache is off"));
            }
            let hint = b.get("session").and_then(Json::as_str).map(str::to_string);
            let req_no = st.reqs.load(Ordering::SeqCst);
            let flag = Arc::new(AtomicBool::new(false));
            // v0.1.4 (PRELOAD-RACE): "busy" means a request, a state write, an idle job or another
            // preload is in flight ([`State::work`]) or a request came since; never a timeout. Any
            // other holder of the worker (a status read, budget / clear) is short: wait for it.
            let work = loop {
                if st.work.load(Ordering::SeqCst) > 0 || st.reqs.load(Ordering::SeqCst) != req_no {
                    return Reply::Json(200, json!({"status": "busy"}));
                }
                // decided without an await while the guard lives (the route's future stays Send)
                let step: Result<WorkGuard, Option<&str>> = {
                    let w = match st.worker.try_lock() {
                        Ok(w) => Some(w),
                        Err(std::sync::TryLockError::Poisoned(p)) => Some(p.into_inner()),
                        Err(std::sync::TryLockError::WouldBlock) => None,
                    };
                    match w {
                        None => Err(None),
                        Some(w) if !w.cache.tokens.is_empty() || w.preloaded.is_some() => {
                            Err(Some("live"))
                        }
                        Some(w)
                            if w.states
                                .as_ref()
                                .is_none_or(|s| s.budget == 0 || s.newest_plan(None).is_none()) =>
                        {
                            Err(Some("none"))
                        }
                        Some(w) => {
                            // counted and armed while the worker is held: a request that comes after
                            // this finds the flag (and its own counter bump makes the task give up)
                            let work = st.begin_work();
                            *st.preload.lock().unwrap() = Some(flag.clone());
                            *st.preload_status.lock().unwrap() = json!({"status": "running"});
                            drop(w);
                            Ok(work)
                        }
                    }
                };
                match step {
                    Ok(work) => break work,
                    Err(Some(status)) => return Reply::Json(200, json!({"status": status})),
                    Err(None) => tokio::time::sleep(std::time::Duration::from_millis(1)).await,
                }
            };
            let st2 = st.clone();
            tokio::task::spawn_blocking(move || {
                // never wait for a request: a request raises `flag` and bumps `reqs` before it
                // queues for the worker, so the preload never runs ahead of one. Any other holder
                // here is short (a status read, budget / clear: a state write or an idle job only
                // follows a request), so the task waits for it instead of giving up on a clock
                // (v0.1.4 FLAKY-PRELOAD-TEST, then PRELOAD-RACE: a preempted short holder outlived
                // the old 200 ms bound under CPU load).
                let r = (|| {
                    let mut w = loop {
                        if flag.load(Ordering::SeqCst) || st2.reqs.load(Ordering::SeqCst) != req_no
                        {
                            return ("cancelled", 0, 0.0);
                        }
                        match st2.worker.try_lock() {
                            Ok(w) => break w,
                            Err(std::sync::TryLockError::Poisoned(p)) => break p.into_inner(),
                            Err(std::sync::TryLockError::WouldBlock) => {
                                std::thread::sleep(std::time::Duration::from_millis(1))
                            }
                        }
                    };
                    if flag.load(Ordering::SeqCst) || st2.reqs.load(Ordering::SeqCst) != req_no {
                        return ("cancelled", 0, 0.0);
                    }
                    w.preload(hint.as_deref(), &flag)
                })();
                // PRELOAD-RACE: the status is published only after the worker is released and the
                // work unit is done, so whoever reads "done" finds the worker free (the old order
                // published it under the lock: a second preload right after saw a held worker)
                drop(work);
                let (status, pos, ms) = r;
                *st2.preload_status.lock().unwrap() =
                    json!({"status": status, "pos": pos, "ms": ms});
            });
            Reply::Json(202, json!({"status": "started"}))
        }
        ("POST", "/state-cache/budget") | ("POST", "/state-cache/clear") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            let clear = path.ends_with("/clear");
            let st2 = st.clone();
            let r = tokio::task::spawn_blocking(move || -> Result<Json, (u16, String)> {
                // after the state write in flight (if any), then refused while a request runs
                let _gate = st2.persist_gate.lock().unwrap_or_else(|p| p.into_inner());
                let mut w = st2.worker.try_lock().map_err(|_| (409, "a request is running".to_string()))?;
                // fair slicing: a suspended generation is a running request (a resume would undo a clear)
                if st2.fair.suspended() > 0 {
                    return Err((409, "a request is running".to_string()));
                }
                let w = &mut *w;
                if w.states.is_none() {
                    return Err((404, "state cache is off".into()));
                }
                let mut echo = st2.echo.lock().unwrap_or_else(|p| p.into_inner());
                if !clear {
                    let bytes = b.get("bytes").and_then(Json::as_u64).ok_or((400, "\"bytes\" must be a non-negative integer".to_string()))?;
                    let s = w.states.as_mut().unwrap();
                    let was_off = s.budget == 0;
                    let freed = s.set_budget(bytes);
                    // Codex P2/P3 review 2: budget 0 = no conversation state goes to disk at all: the
                    // segmented store, the echo records' files and the legacy prefix / session stores
                    // (parks fall back to RAM). Raising it again brings all of them back.
                    if bytes == 0 && !was_off {
                        *st2.echo_dir_off.lock().unwrap() = echo.set_disk(None);
                        w.legacy_off = Some((w.store.take(), w.sessions.take()));
                    } else if bytes > 0 && was_off {
                        if let Some(d) = st2.echo_dir_off.lock().unwrap().take() {
                            echo.set_disk(Some(d));
                        }
                        if let Some((p, se)) = w.legacy_off.take() {
                            w.store = p;
                            w.sessions = se;
                        }
                    }
                    let s = w.states.as_ref().unwrap();
                    return Ok(json!({"budget_bytes": s.budget, "used_bytes": s.used_bytes(), "freed_bytes": freed, "legacy_stores": if w.legacy_off.is_some() { "off" } else { "on" }}));
                }
                let scope = b.get("scope").and_then(Json::as_str).unwrap_or("all").to_string();
                let s = w.states.as_mut().unwrap();
                let other_model = b.get("model").and_then(Json::as_str).filter(|m| *m != s.ident.model_key());
                if scope == "session" {
                    // Codex P2/P3 review 1: purge the session from every tier, keep what others need
                    let sid = b.get("session").and_then(Json::as_str).ok_or((400, "session scope needs a session id".to_string()))?.to_string();
                    let (freed, ram, legacy, hint) = purge_session(w, &sid)?;
                    let s = w.states.as_ref().unwrap();
                    // the echo records of its hint go when no fork of it (same hint) is left
                    let echo_n = match &hint {
                        Some(h) if !s.hint_in_use(h) => echo.drop_hint_hash(h),
                        _ => 0,
                    };
                    return Ok(json!({"freed_bytes": freed, "used_bytes": s.used_bytes(), "ram_states": ram, "legacy_entries": legacy, "echo_records": echo_n}));
                }
                if scope == "hint" {
                    // v0.2.0 EG5: every session CREATED under this exact request hint (`X-Oxe-Session`),
                    // purged as one session clear each; blocks other sessions still name (a parent's
                    // or shared anchor, the system prompt) stay. Unknown hint: nothing to do (200).
                    let hint = b.get("hint").and_then(Json::as_str).filter(|h| valid_hint(h)).ok_or((400, "hint scope needs a hint of 1-128 characters [A-Za-z0-9._-]".to_string()))?;
                    let h = crate::state_store::hint_hash(hint);
                    // a session another hint also owns (continued or absorbed across hints) is kept:
                    // only this hint's ownership goes (Manifest::owners)
                    let sids = w.states.as_ref().unwrap().sessions_with_hint(&h);
                    let (mut freed, mut ram, mut legacy, mut purged, mut released) = (0u64, 0usize, 0usize, 0usize, 0usize);
                    for sid in &sids {
                        if w.states.as_mut().unwrap().release_owner(sid, &h).map_err(|e| (500, e))? {
                            released += 1;
                            continue;
                        }
                        let (f, r, l, _) = purge_session(w, sid)?;
                        freed += f;
                        ram += r;
                        legacy += l;
                        purged += 1;
                    }
                    let s = w.states.as_ref().unwrap();
                    let echo_n = if s.hint_in_use(&h) { 0 } else { echo.drop_hint_hash(&h) };
                    return Ok(json!({"freed_bytes": freed, "used_bytes": s.used_bytes(), "sessions": purged, "released": released, "ram_states": ram, "legacy_entries": legacy, "echo_records": echo_n}));
                }
                let freed = match (scope.as_str(), other_model) {
                    ("model", Some(key)) => {
                        // another model's directory under the root (only one in this store's format)
                        if key.len() != 16 || !key.bytes().all(|c| c.is_ascii_hexdigit()) {
                            return Err((400, "bad model key".into()));
                        }
                        let d = s.root.join(key);
                        let n = crate::state_store::dir_bytes(&d);
                        if d.join("identity.json").is_file() && std::fs::remove_dir_all(&d).is_ok() { n } else { 0 }
                    }
                    _ => s.clear(&scope, None).map_err(|e| (400, e))?,
                };
                let used = s.used_bytes();
                let mut legacy = 0;
                if other_model.is_none() {
                    // the RAM side of the same scope: live state, parked sessions, echo records, and the
                    // legacy disk stores' entries of this engine (Codex P2/P3 review 3)
                    w.cache.reset(&mut w.engine);
                    let _ = w.cache.drain_parked();
                    w.live_blocks.clear();
                    w.preloaded = None;
                    echo.clear();
                    let all = |_: &[u32]| true;
                    for ps in [w.store.as_mut(), w.sessions.as_mut()].into_iter().flatten() {
                        legacy += ps.remove_where(&all);
                    }
                    if let Some((p, se)) = w.legacy_off.as_mut() {
                        for ps in [p.as_mut(), se.as_mut()].into_iter().flatten() {
                            legacy += ps.remove_where(&all);
                        }
                    }
                }
                Ok(json!({"freed_bytes": freed, "used_bytes": used, "legacy_entries": legacy}))
            })
            .await
            .unwrap_or_else(|_| Err((500, "state cache task failed".into())));
            match r {
                Ok(j) => Reply::Json(200, j),
                Err((c, m)) => Reply::Json(c, api::error_json(c, &m)),
            }
        }
        ("POST", "/apply-template") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            match prepare_chat(&st, &b) {
                // CACHE-PERFECT contract checks (tools/cache_contract.py): `"return_tokens": true` adds
                // the prompt's token ids exactly as a chat request tokenizes it, `"echo": true` the ids
                // after the echo splice and its report; with `--engine stub` no weights are loaded
                Ok(pc) => {
                    // `"count_only": true` (v0.2.0): the exact prompt length and the context size only; the
                    // count is the length of the same render `return_tokens` reports, nothing else
                    if b.get("count_only").and_then(Json::as_bool).unwrap_or(false) {
                        let r = st.render(&pc);
                        let mut out = json!({"n_prompt_tokens": r.tokens.len(), "n_ctx": st.n_ctx});
                        if b.get("echo").and_then(Json::as_bool).unwrap_or(false) {
                            let (sp, _) = st.splice(&r, &pc.inputs);
                            out["n_echo_tokens"] = json!(sp.len());
                        }
                        return Reply::Json(200, out);
                    }
                    let mut out = json!({"prompt": pc.applied.prompt});
                    if b.get("return_tokens")
                        .and_then(Json::as_bool)
                        .unwrap_or(false)
                    {
                        let r = st.render(&pc);
                        out["generation_prompt"] =
                            Json::String(pc.applied.generation_prompt.clone());
                        out["msg_starts"] = json!(r.msg_starts);
                        out["assistant_spans"] = json!(r.asst);
                        if b.get("echo").and_then(Json::as_bool).unwrap_or(false) {
                            let (sp, rep) = st.splice(&r, &pc.inputs);
                            out["echo_tokens"] = json!(sp);
                            out["echo"] = rep.to_json();
                        }
                        out["tokens"] = json!(r.tokens);
                    }
                    Reply::Json(200, out)
                }
                Err(r) => r,
            }
        }
        ("POST", "/completion") | ("POST", "/completions") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            // as chat: an ephemeral (LAN share) request keeps no session and writes no state
            completion(
                st,
                b,
                if ephemeral { None } else { session },
                ephemeral,
                hints.snapshot_keep && !ephemeral,
            )
            .await
        }
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            if ephemeral {
                chat_completion(st, b, None, None, true, false).await
            } else {
                chat_completion(st, b, session, turn_kind, false, hints.snapshot_keep).await
            }
        }
        _ => Reply::Json(404, api::error_json(404, "File Not Found")),
    }
}

/// A rendered chat prompt: tokens and the message layout from the template's delimiter offsets.
struct Rendered {
    tokens: Vec<Token>,
    /// token range [`<|im_start|>`, `<|im_end|>`) of each history assistant turn; None = unknown layout
    asst: Option<Vec<(usize, usize)>>,
    /// every message start (template `<|im_start|>`)
    msg_starts: Vec<usize>,
    first_user: Option<usize>,
    last_user: Option<usize>,
    /// every user-message start
    user_starts: Vec<usize>,
}

struct PreparedChat {
    applied: chat::Applied,
    inputs: chat::ChatInputs,
}

fn prepare_chat<E: Engine>(st: &State<E>, body: &Json) -> Result<PreparedChat, Reply> {
    let Some(tmpl) = &st.tmpl else {
        return Err(Reply::Json(
            500,
            api::error_json(500, "model has no chat template"),
        ));
    };
    let inputs = chat::chat_inputs(body, &st.cfg.chat_opts, tmpl).map_err(|e| {
        if e.0.contains("(this model takes text only)") {
            Reply::Json(
                400,
                api::error_json_with(400, &e.0, "unsupported_content_type", Some("messages")),
            )
        } else {
            bad(e.0)
        }
    })?;
    let applied =
        chat::apply(tmpl, &inputs).map_err(|e| bad(format!("chat template error: {e:#}")))?;
    Ok(PreparedChat { applied, inputs })
}

/// Echo fields go into `timings` only for requests with assistant turns in their history (the
/// llama-server shape of a first turn stays as it was).
fn add_echo_timings(resp: &mut Json, echo: &Option<Json>) {
    if let (Some(Json::Object(t)), Some(Json::Object(e))) = (resp.get_mut("timings"), echo) {
        for (k, v) in e {
            t.insert(k.clone(), v.clone());
        }
    }
}

fn gen_error_reply(e: GenError) -> Reply {
    match e {
        GenError::ExceedContext(n, c) => Reply::Json(400, api::exceed_context_json(n, c)),
        GenError::Engine(e) => Reply::Json(500, api::error_json(500, &e.0)),
        GenError::Cancelled => Reply::Json(500, api::error_json(500, "cancelled")),
    }
}

fn gen_error_json(e: GenError) -> Json {
    match gen_error_reply(e) {
        Reply::Json(_, j) => j,
        _ => unreachable!(),
    }
}

/// Run the generation on a blocking thread. For streams the first event decides between an
/// HTTP error and a 200 SSE stream, like llama-server (which pre-fetches the first result).
async fn run_blocking<E: Engine, F>(
    st: Arc<State<E>>,
    prompt: Vec<Token>,
    mut gp: GenParams,
    mut on_partial: F,
) -> Result<GenResult, GenError>
where
    F: FnMut(Partial) -> bool + Send + 'static,
{
    // `timings.queue_ms`: from here until the generation first holds the worker
    let queued = std::time::Instant::now();
    /// Sets the cancel flag if this future is dropped before the generation returns: hyper
    /// drops a non-streamed request's future when its client disconnects.
    struct DropCancel(Arc<AtomicBool>, bool);
    impl Drop for DropCancel {
        fn drop(&mut self) {
            if !self.1 {
                self.0.store(true, Ordering::Relaxed);
            }
        }
    }
    let flag = gp
        .cancel
        .get_or_insert_with(|| Arc::new(AtomicBool::new(false)))
        .clone();
    let mut guard = DropCancel(flag, false);
    // PRELOAD-RACE: a long holder from now until the generation is done with the worker; a preload
    // armed after this request's start (it missed `preempt_idle`) is cancelled here
    let work = st.begin_work();
    if let Some(f) = st.preload.lock().unwrap().take() {
        f.store(true, Ordering::SeqCst);
    }
    let r = tokio::task::spawn_blocking(move || {
        let _work = work;
        let mut gp = gp;
        if st.fair.quantum().is_some() {
            // fair time-slicing between generations (fair.rs)
            let r = crate::fair::run_sliced(&st, &prompt, &mut gp, queued, &mut on_partial);
            if r.as_ref()
                .is_ok_and(|r| r.timings.resume_src != "cold" && r.timings.preload_ms.is_some())
            {
                *st.preload_status.lock().unwrap() = json!({"status": "used"});
            }
            return r;
        }
        let mut w = st.lock_worker();
        if gp.cancelled() {
            return Err(GenError::Cancelled); // the client left while queued for the worker
        }
        gp.queue_ms = crate::generate::ms_since(queued);
        // SIGTERM cancels the running generation (the bounded shutdown write needs the worker)
        *st.running.lock().unwrap() = gp.cancel.clone();
        /// Clears `running` on every exit, unwinding included.
        struct ClearRunning<'a>(&'a Mutex<Option<Arc<AtomicBool>>>);
        impl Drop for ClearRunning<'_> {
            fn drop(&mut self) {
                *self.0.lock().unwrap_or_else(|p| p.into_inner()) = None;
            }
        }
        let r = {
            let _running = ClearRunning(&st.running);
            w.generate(&st.tok, &prompt, &gp, &mut on_partial)
        };
        if w.last_resume.0 != "cold" && r.as_ref().is_ok_and(|r| r.timings.preload_ms.is_some()) {
            *st.preload_status.lock().unwrap() = json!({"status": "used"});
        }
        r
    })
    .await
    .unwrap_or_else(|e| {
        Err(GenError::Engine(crate::engine::EngineError(format!(
            "worker panicked: {e}"
        ))))
    });
    guard.1 = true;
    r
}

/// SX-2 idle speculative re-prefill. After a chat turn that ended the query (`stop`, no tool
/// call), the next request will be this conversation + the reply + a new user message, and the
/// template renders the earlier assistant turns differently once a user message follows (the
/// empty think blocks are dropped), so that request would re-prefill the whole query. Render that
/// history form now (messages + reply + an empty user turn, cut before the user message) and
/// prefill it while the client reads. Lossless: the tokens are the next prompt's prefix; the
/// raw session is parked (a retry still finds it); any request preempts the job.
fn spawn_idle_reprefill<E: Engine>(st: &Arc<State<E>>, b: &Json, reply: Json, req_no: u64) {
    if !st.cfg.idle_reprefill {
        return;
    }
    if st.user_delim.is_none() {
        return;
    }
    let mut body = b.clone();
    let Some(msgs) = body.get_mut("messages").and_then(Json::as_array_mut) else {
        return;
    };
    msgs.push(reply);
    msgs.push(json!({"role": "user", "content": ""}));
    let Ok(pc) = prepare_chat(st, &body) else {
        return;
    };
    let r = st.render(&pc);
    let (toks, rep) = st.splice(&r, &pc.inputs);
    let Some(cut) = r.last_user.map(|c| rep.map_pos(c)) else {
        return;
    };
    let hist = toks[..cut.min(toks.len())].to_vec();
    let req_no = st.fair.post_reply_req(req_no);
    let Ok(mut gp) = api::gen_params(&body, true) else {
        return;
    };
    gp.first_user = r.first_user.map(|c| rep.map_pos(c)).filter(|&c| c < cut);
    gp.msg_starts = r
        .msg_starts
        .iter()
        .map(|&c| rep.map_pos(c))
        .filter(|&c| c < cut)
        .collect();
    gp.query_start = false;
    let flag = Arc::new(AtomicBool::new(false));
    gp.cancel = Some(flag.clone());
    *st.idle.lock().unwrap() = Some(flag);
    let work = st.begin_work();
    let st2 = st.clone();
    tokio::task::spawn_blocking(move || {
        {
            let mut w = st2.lock_worker();
            // a newer request started (or is queued): leave the worker to it
            if gp.cancelled() || st2.reqs.load(Ordering::SeqCst) != req_no {
                drop(w);
            } else {
                match w.prefill_idle(&hist, &gp) {
                    Ok(n) if n > 0 => {
                        eprintln!("oxe-server: idle re-prefill: {n} tokens of the history form")
                    }
                    Ok(_) => {}
                    Err(GenError::Cancelled) => {
                        eprintln!("oxe-server: idle re-prefill: preempted by a request")
                    }
                    Err(e) => eprintln!("oxe-server: idle re-prefill failed: {e:?}"),
                }
            }
        }
        drop(work);
        st2.idle_done.fetch_add(1, Ordering::SeqCst);
    });
}

/// CACHE-PERFECT P2: write the live state to the state store after the reply (off the request
/// path, in the client's idle gap). A newer request preempts it between blocks; what was written
/// stays valid, the next reply writes the rest.
fn spawn_persist<E: Engine>(st: &Arc<State<E>>, session: Option<String>, req_no: u64) {
    if !st.state_cache {
        return;
    }
    // fair slicing: skipped when another generation was queued at this one's end (its switch path
    // writes this state); a sliced one that ended with nobody queued writes although newer requests
    // were counted while it was suspended (they all ended before it). Identity when slicing is off.
    let req_no = st.fair.post_reply_req(req_no);
    let flag = Arc::new(AtomicBool::new(false));
    *st.persist.lock().unwrap() = Some(flag.clone());
    let work = st.begin_work();
    let st2 = st.clone();
    tokio::task::spawn_blocking(move || {
        let _gate = st2.persist_gate.lock().unwrap_or_else(|p| p.into_inner());
        let t0 = std::time::Instant::now();
        // v0.1.3 (item 6): phase 1 copies the head snapshot and the new blocks under the lock ...
        let staged = {
            let mut w = st2.lock_worker();
            if flag.load(Ordering::Relaxed) || st2.reqs.load(Ordering::SeqCst) != req_no {
                None
            } else {
                let s = w.stage_persist(session.as_deref());
                if s.is_none() {
                    // nothing to stage (or more than the stage holds): the on-lock writer, as before
                    let n = w.persist_states(session.as_deref(), Some(&flag));
                    if n > 0 {
                        eprintln!(
                            "oxe-server: state cache: {n} files written in {:.0} ms",
                            t0.elapsed().as_secs_f64() * 1e3
                        );
                    }
                }
                s
            }
        };
        if let Some(s) = staged {
            // ... phase 2 writes them without it (a request that arrives now runs at once; the
            // write completes behind it), phase 3 commits the manifest under the lock again
            let written = s.write();
            let mut w = st2.lock_worker();
            let r = w.commit_staged(s, written, session.as_deref(), Some(&flag));
            if r.head.is_some() {
                eprintln!("oxe-server: state cache: {} files written in {:.0} ms (head off the worker lock)", r.files(), t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        drop(work);
        st2.persist_done.fetch_add(1, Ordering::SeqCst);
    });
}

/// S8 (REVIEW-v0.1.1): send one SSE frame from the generation thread; false = stop generating.
/// A client that keeps its socket open but stops reading fills the channel; `blocking_send`
/// then held the worker (and every later request) for ever. Now, after `stall` without room
/// for the next frame, the stream is given up: the generation is cancelled like a disconnect.
pub(crate) fn send_frame(
    tx: &mpsc::Sender<String>,
    frame: String,
    stall: std::time::Duration,
) -> bool {
    use tokio::sync::mpsc::error::TrySendError;
    let t0 = std::time::Instant::now();
    let mut frame = frame;
    loop {
        match tx.try_send(frame) {
            Ok(()) => return true,
            Err(TrySendError::Closed(_)) => return false,
            Err(TrySendError::Full(f)) => {
                if t0.elapsed() >= stall {
                    eprintln!("oxe-server: stream client took no data for {:.0} s: stream dropped, generation cancelled", stall.as_secs_f64());
                    return false;
                }
                frame = f;
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    }
}

/// The async side of [`send_frame`] (final frames after the worker is free): false when the
/// client is gone or has taken nothing for `stall`.
async fn send_final(tx: &mpsc::Sender<String>, frame: String, stall: std::time::Duration) -> bool {
    matches!(
        tokio::time::timeout(stall, tx.send(frame)).await,
        Ok(Ok(()))
    )
}

/// v0.2.0 EG5: a request hint as `/state-cache/clear {"scope": "hint"}` accepts it: 1-128 characters
/// of `[A-Za-z0-9._-]` (a client's `parent.s1` style child id fits; nothing else is a hint).
pub fn valid_hint(h: &str) -> bool {
    (1..=128).contains(&h.len())
        && h.bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
}

/// Purge stored session `sid` from every tier (Codex P2/P3 review 1): its manifest and the blocks no
/// other session names, the live and parked RAM state that is its own, the legacy stores' entries.
/// Returns (freed bytes, RAM states, legacy entries, its hint hash); the echo records are the
/// caller's. 400 "unknown session" for an id with no manifest.
fn purge_session<E: Engine>(
    w: &mut crate::generate::Worker<E>,
    sid: &str,
) -> Result<(u64, usize, usize, Option<String>), (u16, String)> {
    let s = w.states.as_mut().unwrap();
    let m = s
        .session_match(sid)
        .ok_or((400, "unknown session".to_string()))?;
    let freed = s.clear("session", Some(sid)).map_err(|e| (400, e))?;
    let owns = |t: &[u32]| m.owns(t);
    let mut ram = 0;
    let pre = w.preloaded.as_ref().is_some_and(|p| p.plan.sid == sid);
    if pre || (!w.cache.tokens.is_empty() && owns(&w.cache.tokens)) {
        w.cache.reset(&mut w.engine);
        w.live_blocks.clear();
        w.preloaded = None;
        ram += 1;
    }
    ram += w.cache.drop_parked_where(&owns);
    let mut legacy = 0;
    for ps in [w.store.as_mut(), w.sessions.as_mut()]
        .into_iter()
        .flatten()
    {
        legacy += ps.remove_where(&owns);
    }
    if let Some((p, se)) = w.legacy_off.as_mut() {
        for ps in [p.as_mut(), se.as_mut()].into_iter().flatten() {
            legacy += ps.remove_where(&owns);
        }
    }
    Ok((freed, ram, legacy, m.hint))
}

/// `usage.completion_tokens_details.reasoning_tokens`: the generated tokens up to and including the
/// think end (`</think>`) when the generation prompt opened a think block; else the span of a
/// `<think>` .. `</think>` block the model wrote itself; a think block cut by the limit counts to the
/// end. 0 for a template without single-token think markers.
pub fn reasoning_tokens(tok: &Tokenizer, tokens: &[Token], generation_prompt: &str) -> usize {
    let Some(end) = tok.token_id("</think>") else {
        return 0;
    };
    let opened = generation_prompt
        .rfind("<think>")
        .is_some_and(|i| !generation_prompt[i..].contains("</think>"));
    let start = if opened {
        Some(0)
    } else {
        tok.token_id("<think>")
            .and_then(|b| tokens.iter().position(|&t| t == b))
    };
    let Some(start) = start else { return 0 };
    match tokens[start..].iter().position(|&t| t == end) {
        Some(e) => e + 1,
        None => tokens.len() - start,
    }
}

/// Move the pending per-token log-probabilities onto a streamed chunk (nothing when none are
/// pending; logprobs not requested = never any).
fn attach_logprobs(chunk: &mut Json, pend: &mut Vec<crate::generate::TokenProbs>) {
    if pend.is_empty() {
        return;
    }
    if let Some(c) = chunk
        .pointer_mut("/choices/0")
        .and_then(Json::as_object_mut)
    {
        c.insert(
            "logprobs".into(),
            json!({"content": api::probs_json(pend, false)}),
        );
        pend.clear();
    }
}

/// Percent-decoding of one URL path segment (`%2F` etc.; invalid escapes are kept as they are).
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `/props.speculative` from the engine's load-time decision.
pub fn speculative_json(max_depth: usize, fallback: Option<String>) -> Json {
    let lookup = std::env::var("OXE_LOOKUP_LANES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .is_some_and(|l| l > 1);
    let (active, depth) = match max_depth {
        d @ 1..=8 => ("draft-mtp", d),
        _ if lookup => ("lookup", 0),
        _ => ("none", 0),
    };
    let requested = if fallback.is_some() {
        "draft-mtp"
    } else {
        active
    };
    json!({"active": active, "depth": depth, "requested": requested, "fallback": fallback})
}

/// The default output limit (see [`State::default_max_tokens`]): `OXE_DEFAULT_MAX_TOKENS` (0 = none),
/// else min(16384, n_ctx / 4) (at least 1). -1 = no default.
pub fn default_max_tokens_from_env(n_ctx: usize) -> i64 {
    default_max_tokens_for(
        std::env::var("OXE_DEFAULT_MAX_TOKENS").ok().as_deref(),
        n_ctx,
    )
}

/// [`default_max_tokens_from_env`] for a given setting.
pub fn default_max_tokens_for(setting: Option<&str>, n_ctx: usize) -> i64 {
    match setting.and_then(|v| v.trim().parse::<i64>().ok()) {
        Some(0) => -1,
        Some(n) if n > 0 => n,
        _ => (n_ctx as i64 / 4).clamp(1, 16384),
    }
}

/// LONG-APPEND default SSE keepalive interval (seconds).
pub const SSE_KEEPALIVE_SECS: f64 = 15.0;

/// `OXE_SSE_KEEPALIVE_SECS` (0 = off; unset or unparsable: [`SSE_KEEPALIVE_SECS`]).
pub fn sse_keepalive_from_env() -> std::time::Duration {
    let s = std::env::var("OXE_SSE_KEEPALIVE_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(SSE_KEEPALIVE_SECS);
    std::time::Duration::from_secs_f64(s)
}

/// LONG-APPEND (v0.1.3 #7): the SSE comment the keepalive sends ("queued" before the worker takes
/// the request, then the prefill position). v0.2.0 F10: the queued line has the one format the lab
/// controller's queue uses too, `: queued [pos=<n>] wait_s=<s>` (key=value fields, whole seconds
/// since the request arrived, rounded down; `pos` only where a component keeps an ordered queue: the
/// engine's worker is a lock, not a queue, so it sends no `pos`). See SERVER.md "Long prefills".
pub fn keepalive_comment(pr: &Progress, elapsed: std::time::Duration) -> String {
    let (done, total) = (
        pr.done.load(Ordering::Relaxed),
        pr.total.load(Ordering::Relaxed),
    );
    // fair slicing: a suspended generation waits for its turn again
    if total == 0 || pr.suspended.load(Ordering::Relaxed) {
        format!(": queued wait_s={}\n\n", elapsed.as_secs())
    } else {
        format!(": prefill {done}/{total} {:.0}s\n\n", elapsed.as_secs_f64())
    }
}

/// LONG-APPEND (v0.1.3 #7): a streamed request sends nothing while it waits for the worker or
/// prefills, which at depth can take many minutes (28.5K tokens at ~83K: > 600 s). Every `every`
/// this sends one SSE comment line (`: prefill <done>/<total> <s>s`): SSE parsers ignore comments
/// (WHATWG EventSource, eventsource-parser, Effect's Sse parser in agent clients, the OpenAI SDKs), so
/// clients see no change, but byte-level idle-read timeouts (sockets, proxies, the lab controller)
/// are reset and the controller can show progress. Event-level timeouts are NOT reset (the
/// client's SSE chunk timeout, e.g. 900 s by default, counts only `data:` events): those clients
/// need a generous setting (the lab's client overlay sets 3600 s). Stops at the first decoded token (`decoding`), when the response
/// ends (a weak sender: it never keeps the stream open) or when the client is gone; never blocks
/// the generation (a full channel just skips a tick).
fn spawn_keepalive(tx: &mpsc::Sender<String>, pr: Arc<Progress>, every: std::time::Duration) {
    if every.is_zero() {
        return;
    }
    let weak = tx.downgrade();
    let t0 = std::time::Instant::now();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval_at(tokio::time::Instant::now() + every, every);
        loop {
            tick.tick().await;
            if pr.decoding.load(Ordering::Relaxed) && !pr.fair.load(Ordering::Relaxed) {
                return;
            }
            let Some(tx) = weak.upgrade() else { return };
            // fair slicing: a generation can be suspended after its first token; the comments then
            // go on while it waits for its turn (`: queued`), and only then
            if pr.decoding.load(Ordering::Relaxed) && !pr.suspended.load(Ordering::Relaxed) {
                if tx.is_closed() {
                    return;
                }
                continue;
            }
            if let Err(mpsc::error::TrySendError::Closed(_)) =
                tx.try_send(keepalive_comment(&pr, t0.elapsed()))
            {
                return;
            }
        }
    });
}

/// Streams: cancel when the client's end of the SSE channel is dropped (the client went
/// away), also while the prompt is still being prefilled.
fn cancel_on_close(gp: &mut GenParams, tx: &mpsc::Sender<String>) {
    let flag = Arc::new(AtomicBool::new(false));
    gp.cancel = Some(flag.clone());
    // A weak handle: holding a Sender here would keep the channel open after the response
    // ends (the client would wait for the end of the stream forever).
    let weak = tx.downgrade();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_millis(100));
        loop {
            tick.tick().await;
            match weak.upgrade() {
                None => return, // the response finished: every sender is gone
                Some(tx) => {
                    if tx.is_closed() {
                        flag.store(true, Ordering::Relaxed);
                        return;
                    }
                }
            }
        }
    });
}

/// Structured-output fields that this server does not implement: refuse them instead of
/// silently ignoring them (ENGINE-AUDIT F7).
fn unsupported_constraint(b: &Json) -> Option<String> {
    for k in ["response_format", "grammar", "json_schema"] {
        match b.get(k) {
            None | Some(Json::Null) => {}
            Some(Json::Object(o)) if k == "response_format" && o.get("type").and_then(Json::as_str) == Some("text") => {}
            Some(_) => return Some(format!("\"{k}\" (constrained/structured output) is not supported by oxe-server; remove it or use tools")),
        }
    }
    None
}

async fn completion<E: Engine>(
    st: Arc<State<E>>,
    b: Json,
    session: Option<String>,
    ephemeral: bool,
    snapshot_keep: bool,
) -> Reply {
    let req_no = st.preempt_idle();
    if let Some(m) = unsupported_constraint(&b) {
        return bad(m);
    }
    let mut gp = match api::gen_params(&b, false) {
        Ok(g) => g,
        Err(e) => return bad(e),
    };
    if !api::has_length_field(&b) {
        gp.n_predict = st.default_max_tokens;
    }
    gp.ephemeral = ephemeral;
    gp.session = session.clone();
    gp.req_no = Some(req_no);
    let prompt_json = b
        .get("prompt")
        .cloned()
        .unwrap_or(Json::String(String::new()));
    let t_tok = std::time::Instant::now();
    let prompt = match st.tokenize_prompt(&prompt_json, true) {
        Ok(p) => p,
        Err(e) => return bad(e),
    };
    gp.tokenize_ms = crate::generate::ms_since(t_tok);
    if let Err(e) = st.check_tokens(&prompt) {
        return bad(e);
    }
    if prompt.len() >= st.n_ctx {
        return Reply::Json(400, api::exceed_context_json(prompt.len(), st.n_ctx));
    }
    // H4 on a raw prompt: there are no messages, so the point is the prompt end
    gp.snap_keep = snapshot_keep;
    gp.snap_at = snapshot_keep.then_some(prompt.len());
    let prompt_text = match &prompt_json {
        Json::String(s) => s.clone(),
        _ => st.tok.decode(&prompt, true),
    };
    let stream = b.get("stream").and_then(Json::as_bool).unwrap_or(false);
    let post = b
        .get("post_sampling_probs")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let return_tokens = b
        .get("return_tokens")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let model = st.cfg.model_name.clone();
    if !stream {
        return match run_blocking(st.clone(), prompt, gp.clone(), |_| true).await {
            Ok(r) => {
                // CACHE-PERFECT P2 after the reply, as chat (the write runs off the request path)
                if !ephemeral {
                    spawn_persist(&st, session.clone(), req_no);
                }
                Reply::Json(
                    200,
                    api::completion_final(
                        &model,
                        &prompt_text,
                        &r,
                        false,
                        &gp,
                        return_tokens,
                        post,
                    ),
                )
            }
            Err(e) => gen_error_reply(e),
        };
    }
    let (tx, rx) = mpsc::channel::<String>(256);
    cancel_on_close(&mut gp, &tx);
    let pr = Arc::new(Progress::default());
    gp.progress = Some(pr.clone());
    spawn_keepalive(&tx, pr.clone(), st.sse_keepalive);
    let n_prompt = prompt.len();
    let tx2 = tx.clone();
    let st2 = st.clone();
    let st3 = st.clone();
    let session3 = session.clone();
    let stall = st.stream_stall;
    tokio::spawn(async move {
        let sent = Arc::new(std::sync::atomic::AtomicUsize::new(0)); // probability rows streamed
        let sent2 = sent.clone();
        let res = run_blocking(st2, prompt, gp.clone(), move |p: Partial| {
            pr.decoding.store(true, Ordering::Relaxed);
            sent2.fetch_add(p.probs.len(), Ordering::Relaxed);
            let j = api::completion_partial(&p.text, p.n_decoded, n_prompt, &p.probs, post);
            send_frame(&tx2, api::sse(&j), stall)
        })
        .await;
        let frame = match res {
            Ok(r) => {
                if !ephemeral {
                    spawn_persist(&st3, session3, req_no);
                }
                let mut fin =
                    api::completion_final(&model, &prompt_text, &r, true, &gp, return_tokens, post);
                // rows of the tokens after the last partial (a held-back UTF-8 tail, a stop word)
                let n = sent.load(Ordering::Relaxed).min(r.probs.len());
                if n < r.probs.len() {
                    fin["completion_probabilities"] = api::probs_json(&r.probs[n..], post);
                }
                api::sse(&fin)
            }
            Err(GenError::Cancelled) => return,
            Err(e) => api::sse(&gen_error_json(e)),
        };
        let _ = send_final(&tx, frame, stall).await;
    });
    Reply::Stream(rx)
}

/// The one prompt of a legacy request: a string, an array of one string, an array of token ids, or
/// an array of one token-id array. Several prompts are refused (`unsupported_value`), a malformed
/// one is `invalid_value`; both name `prompt`.
fn legacy_prompt(b: &Json) -> Result<Json, Reply> {
    let refuse =
        |code: &str, m: &str| Reply::Json(400, api::error_json_with(400, m, code, Some("prompt")));
    const SEVERAL: &str = "One prompt per request: send a string, or an array with one prompt.";
    match b.get("prompt") {
        Some(p @ Json::String(_)) => Ok(p.clone()),
        Some(Json::Array(a)) if a.is_empty() => Err(refuse("invalid_value", "\"prompt\" is empty")),
        Some(p @ Json::Array(a)) if a.iter().all(Json::is_number) => Ok(p.clone()),
        Some(Json::Array(a)) if a.iter().all(Json::is_string) || a.iter().all(Json::is_array) => {
            match a.len() {
                1 => Ok(a[0].clone()),
                _ => Err(refuse("unsupported_value", SEVERAL)),
            }
        }
        _ => Err(refuse(
            "invalid_value",
            "\"prompt\" must be a string, an array of one string, or an array of token ids",
        )),
    }
}

/// Running state of one legacy stream: the echo text not yet sent, the character offset of the next
/// logprobs row, the rows and the text bytes already sent.
struct LegacyTally {
    echo: Option<String>,
    offset: usize,
    rows: usize,
    sent: usize,
}

/// OpenAI legacy `POST /v1/completions` (`text_completion`). The same generation as `completion`
/// (tokenisation, default output limit, session and hint handling, state write after the reply,
/// cancel on close, keepalive); only the body and the reply differ. One prompt; `suffix` (insert
/// mode) is refused; `logprobs` is 0..=5; `echo` prefixes the prompt text to the reply.
async fn v1_completion<E: Engine>(
    st: Arc<State<E>>,
    b: Json,
    session: Option<String>,
    ephemeral: bool,
    snapshot_keep: bool,
) -> Reply {
    if let Err(j) = api::check_n(&b) {
        return Reply::Json(400, j);
    }
    let req_no = st.preempt_idle();
    if let Some(m) = unsupported_constraint(&b) {
        return bad(m);
    }
    let prompt_json = match legacy_prompt(&b) {
        Ok(p) => p,
        Err(r) => return r,
    };
    match b.get("suffix") {
        None | Some(Json::Null) => {}
        Some(Json::String(s)) if s.is_empty() => {}
        Some(_) => {
            return Reply::Json(
                400,
                api::error_json_with(
                    400,
                    "\"suffix\" (insert mode) is not supported by this model server.",
                    "unsupported_value",
                    Some("suffix"),
                ),
            )
        }
    }
    // absent / null: no logprobs object. An explicit 0 is valid: the chosen token's logprob with an
    // empty top list (the sampler then reports one candidate, which the reply leaves out)
    let top_n: Option<usize> = match b.get("logprobs") {
        None | Some(Json::Null) => None,
        Some(v) => match v.as_u64() {
            Some(n) if n <= 5 => Some(n as usize),
            _ => {
                return Reply::Json(
                    400,
                    api::error_json_with(
                        400,
                        "\"logprobs\" must be an integer from 0 to 5",
                        "invalid_value",
                        Some("logprobs"),
                    ),
                )
            }
        },
    };
    let echo = match b.get("echo") {
        None | Some(Json::Null) | Some(Json::Bool(false)) => None,
        Some(Json::Bool(true)) => match &prompt_json {
            Json::String(s) => Some(s.clone()),
            _ => Some(String::new()),
        },
        Some(_) => {
            return Reply::Json(
                400,
                api::error_json_with(
                    400,
                    "\"echo\" must be a boolean",
                    "invalid_value",
                    Some("echo"),
                ),
            )
        }
    };
    // the native parameters, with the legacy `logprobs` count as `n_probs`
    let mut gb = b.clone();
    if let Some(o) = gb.as_object_mut() {
        o.remove("logprobs");
        o.insert("n_probs".into(), Json::from(top_n.map_or(0, |n| n.max(1))));
    }
    let mut gp = match api::gen_params(&gb, false) {
        Ok(g) => g,
        Err(e) => return bad(e),
    };
    if !api::has_length_field(&b) {
        gp.n_predict = st.default_max_tokens;
    }
    gp.ephemeral = ephemeral;
    gp.session = session.clone();
    gp.req_no = Some(req_no);
    let t_tok = std::time::Instant::now();
    let prompt = match st.tokenize_prompt(&prompt_json, true) {
        Ok(p) => p,
        Err(e) => {
            return Reply::Json(
                400,
                api::error_json_with(400, &e, "invalid_value", Some("prompt")),
            )
        }
    };
    gp.tokenize_ms = crate::generate::ms_since(t_tok);
    if let Err(e) = st.check_tokens(&prompt) {
        return Reply::Json(
            400,
            api::error_json_with(400, &e, "invalid_value", Some("prompt")),
        );
    }
    if prompt.len() >= st.n_ctx {
        return Reply::Json(400, api::exceed_context_json(prompt.len(), st.n_ctx));
    }
    gp.snap_keep = snapshot_keep;
    gp.snap_at = snapshot_keep.then_some(prompt.len());
    let stream = b.get("stream").and_then(Json::as_bool).unwrap_or(false);
    let include_usage = b
        .pointer("/stream_options/include_usage")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let meta = Arc::new(ChatMeta {
        id: format!("cmpl-{}", &api::random_hex()[..24]),
        model: st.cfg.model_name.clone(),
        created: api::now_secs(),
        fingerprint: st.cfg.fingerprint.clone(),
    });
    let echo_chars = echo.as_deref().map_or(0, |e| e.chars().count());
    if !stream {
        let r = match run_blocking(st.clone(), prompt, gp, |_| true).await {
            Ok(r) => r,
            Err(e) => return gen_error_reply(e),
        };
        // CACHE-PERFECT P2 after the reply, as the native route (the write runs off the request path)
        if !ephemeral {
            spawn_persist(&st, session.clone(), req_no);
        }
        let mut offset = echo_chars;
        let logprobs = top_n
            .filter(|_| !r.probs.is_empty())
            .map(|n| api::legacy_logprobs(&r.probs, n, &mut offset));
        let text = format!("{}{}", echo.unwrap_or_default(), r.text);
        return Reply::Json(200, api::text_completion(&meta, &text, &r, logprobs));
    }
    let (tx, rx) = mpsc::channel::<String>(256);
    cancel_on_close(&mut gp, &tx);
    let pr = Arc::new(Progress::default());
    gp.progress = Some(pr.clone());
    spawn_keepalive(&tx, pr.clone(), st.sse_keepalive);
    let tally = Arc::new(Mutex::new(LegacyTally {
        echo,
        offset: echo_chars,
        rows: 0,
        sent: 0,
    }));
    let tally2 = tally.clone();
    let meta2 = meta.clone();
    let tx2 = tx.clone();
    let st2 = st.clone();
    let st3 = st.clone();
    let session3 = session.clone();
    let stall = st.stream_stall;
    tokio::spawn(async move {
        let tally3 = tally2.clone();
        let res = run_blocking(st2, prompt, gp, move |p: Partial| {
            pr.decoding.store(true, Ordering::Relaxed);
            let mut t = tally2.lock().unwrap_or_else(|q| q.into_inner());
            t.sent += p.text.len();
            let mut text = p.text;
            if let Some(e) = t.echo.take() {
                text = format!("{e}{text}");
            }
            let logprobs = top_n
                .filter(|_| !p.probs.is_empty())
                .map(|n| api::legacy_logprobs(&p.probs, n, &mut t.offset));
            t.rows += p.probs.len();
            drop(t);
            send_frame(
                &tx2,
                api::sse(&api::text_chunk(&meta2, &text, None, logprobs)),
                stall,
            )
        })
        .await;
        let r = match res {
            Ok(r) => r,
            Err(GenError::Cancelled) => return,
            Err(e) => {
                let _ = send_final(&tx, api::sse(&gen_error_json(e)), stall).await;
                return;
            }
        };
        if !ephemeral {
            spawn_persist(&st3, session3, req_no);
        }
        // what the partials did not carry: the tail text, the rows after the last partial (a held-back
        // UTF-8 tail, a stop word) and the echo text when no partial went out
        let frames = {
            let mut t = tally3.lock().unwrap_or_else(|q| q.into_inner());
            let head = t.echo.take().unwrap_or_default();
            let tail_text = r.text.get(t.sent..).unwrap_or("");
            let n = t.rows.min(r.probs.len());
            let logprobs = top_n
                .filter(|_| n < r.probs.len())
                .map(|k| api::legacy_logprobs(&r.probs[n..], k, &mut t.offset));
            let fin = api::text_chunk(
                &meta,
                &format!("{head}{tail_text}"),
                Some(api::finish_reason(r.stop, false)),
                logprobs,
            );
            let mut frames = vec![fin];
            if include_usage {
                frames.push(api::text_usage_chunk(&meta, &r));
            }
            if let Some(last) = frames.last_mut().and_then(Json::as_object_mut) {
                last.insert("timings".into(), api::timings_json(&r.timings));
            }
            frames
        };
        for f in frames {
            if !send_final(&tx, api::sse(&f), stall).await {
                return;
            }
        }
        let _ = send_final(&tx, "data: [DONE]\n\n".to_string(), stall).await;
    });
    Reply::Stream(rx)
}

async fn chat_completion<E: Engine>(
    st: Arc<State<E>>,
    mut b: Json,
    session: Option<String>,
    turn_kind: Option<String>,
    ephemeral: bool,
    snapshot_keep: bool,
) -> Reply {
    if let Err(j) = api::check_n(&b) {
        return Reply::Json(400, j);
    }
    let req_no = st.preempt_idle();
    if let Some(m) = unsupported_constraint(&b) {
        return bad(m);
    }
    let mut gp = match api::gen_params(&b, true) {
        Ok(g) => g,
        Err(e) => return bad(e),
    };
    if !api::has_length_field(&b) {
        gp.n_predict = st.default_max_tokens;
    }
    // v0.2.0 H6 (`max_tokens: 0`): prefill only. The prompt is the conversation itself, without the
    // assistant opener a generating request renders, so the next request (this conversation + one
    // more message) continues it exactly (its cache_n = this prompt_tokens). An explicit
    // `add_generation_prompt` from the client is kept.
    let prefill_only = gp.n_predict == 0;
    if prefill_only {
        if let Some(o) = b.as_object_mut() {
            o.entry("add_generation_prompt")
                .or_insert(Json::Bool(false));
        }
    }
    let t_render = std::time::Instant::now();
    let pc = match prepare_chat(&st, &b) {
        Ok(p) => p,
        Err(r) => return r,
    };
    gp.render_ms = crate::generate::ms_since(t_render);
    gp.special = false;
    let t_tok = std::time::Instant::now();
    let rendered = st.render(&pc);
    // ECHO: earlier assistant turns as their exact generated ids (cache keeps every token); an
    // ephemeral request never looks at this machine's echo records (its prompt is the plain render)
    let (prompt, echo) = if ephemeral {
        (rendered.tokens.clone(), Default::default())
    } else {
        st.splice(&rendered, &pc.inputs)
    };
    gp.tokenize_ms = crate::generate::ms_since(t_tok);
    gp.ephemeral = ephemeral;
    let echo_json = (echo.blocks > 0).then(|| echo.to_json());
    // HISTORY-REWRITE: the history turns fed as rendered, kept after the request (EchoStore::adopt)
    let adopt = (!echo.unspliced.is_empty()).then(|| {
        (
            echo.unspliced.clone(),
            pc.inputs.messages.clone(),
            pc.inputs.reasoning_sent.clone(),
        )
    });
    gp.echo = echo_json.clone();
    if let Err(e) = st.check_tokens(&prompt) {
        return bad(e);
    }
    if prompt.len() >= st.n_ctx {
        return Reply::Json(400, api::exceed_context_json(prompt.len(), st.n_ctx));
    }
    gp.first_user = rendered.first_user.map(|c| echo.map_pos(c));
    gp.msg_starts = rendered
        .msg_starts
        .iter()
        .map(|&c| echo.map_pos(c))
        .collect();
    gp.user_starts = rendered
        .user_starts
        .iter()
        .map(|&c| echo.map_pos(c))
        .collect();
    gp.pre_think = st.pre_think(&prompt, &pc.applied.generation_prompt);
    // v0.2.0 H5: a free label for the ledger only (any value; kept to a safe charset there)
    gp.turn_kind = turn_kind
        .map(|k| ledger_label(&k))
        .filter(|k| !k.is_empty());
    drop(rendered);
    gp.query_start = b
        .get("messages")
        .and_then(Json::as_array)
        .and_then(|m| m.last())
        .and_then(|m| m.get("role"))
        .and_then(Json::as_str)
        == Some("user");
    gp.session = session.clone();
    gp.req_no = Some(req_no);
    // H4: the exact end of the last message (`snapshot_point`); none for an ephemeral request
    gp.snap_keep = snapshot_keep && !ephemeral;
    if gp.snap_keep {
        gp.snap_at = st.snapshot_point(
            &prompt,
            &pc.applied.generation_prompt,
            pc.inputs.add_generation_prompt,
        );
    }
    let tmpl = st.tmpl.as_ref().unwrap();
    let pcfg = ParserConfig {
        format: pc.applied.format,
        tools: pc.inputs.tools.clone(),
        tool_choice: pc.inputs.tool_choice,
        parallel_tool_calls: pc.inputs.parallel_tool_calls,
        reasoning_format: pc.inputs.reasoning_format,
        supports_thinking: tmpl.supports_thinking,
        // a valid call with its parameters in another order is still a call (llama.cpp's
        // grammar drops it: required parameters first)
        lenient: true,
    };
    let meta = Arc::new(ChatMeta {
        id: format!("chatcmpl-{}", chat::random_string()),
        model: st.cfg.model_name.clone(),
        created: api::now_secs(),
        fingerprint: st.cfg.fingerprint.clone(),
    });
    let stream = b.get("stream").and_then(Json::as_bool).unwrap_or(false);
    let include_usage = b
        .pointer("/stream_options/include_usage")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let want_probs = gp.n_probs > 0;
    let gen_prompt = pc.applied.generation_prompt.clone();
    let prompt_kept = prompt.clone();
    if !stream {
        let mut r = match run_blocking(st.clone(), prompt, gp, |_| true).await {
            Ok(r) => r,
            Err(e) => return gen_error_reply(e),
        };
        r.reasoning_tokens = reasoning_tokens(&st.tok, &r.tokens, &gen_prompt);
        let mut sess = Session::new(pcfg, gen_prompt, Box::new(chat::random_string));
        sess.incremental = std::env::var_os("OXE_SERVER_LEGACY").is_none();
        if !prefill_only {
            if let Err(e) = sess.feed(&r.text, true) {
                return Reply::Json(500, api::error_json(500, &e));
            }
        }
        if !ephemeral {
            st.echo_adopt(&prompt_kept, adopt.as_ref(), session.as_deref());
            // the whole text was parsed into the reply: nothing held back. H6: no turn, no record
            if !prefill_only {
                st.echo_record(&prompt_kept, sess.msg(), &r, session.as_deref(), true);
            }
            spawn_persist(&st, session.clone(), req_no);
        }
        if !ephemeral
            && !prefill_only
            && api::finish_reason(r.stop, !sess.msg().tool_calls.is_empty()) == "stop"
        {
            spawn_idle_reprefill(&st, &b, sess.msg().to_json(), req_no);
        }
        let mut resp = api::chat_response(&meta, sess.msg(), &r, want_probs, false);
        add_echo_timings(&mut resp, &echo_json);
        return Reply::Json(200, resp);
    }
    let (tx, rx) = mpsc::channel::<String>(256);
    cancel_on_close(&mut gp, &tx);
    let pr = Arc::new(Progress::default());
    gp.progress = Some(pr.clone());
    spawn_keepalive(&tx, pr.clone(), st.sse_keepalive);
    let gen_prompt_r = gen_prompt.clone();
    let tok_r = st.tok.clone();
    let mut sess0 = Session::new(pcfg, gen_prompt, Box::new(chat::random_string));
    // OXE_SERVER_LEGACY=1 (A/B diagnostic only): full re-parse every token
    sess0.incremental = std::env::var_os("OXE_SERVER_LEGACY").is_none();
    let sess = Arc::new(Mutex::new(sess0));
    let st2 = st.clone();
    let stall = st.stream_stall;
    // bytes of the turn's text fed to the stream parser (V014-SL-1: a length-cut turn is kept only
    // when that is all of it)
    let fed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let fed2 = fed.clone();
    tokio::spawn(async move {
        let tx2 = tx.clone();
        let m2 = meta.clone();
        let s2 = sess.clone();
        // streamed logprobs (OpenAI: choices[0].logprobs.content on the delta chunks): each token's
        // entry rides on the next delta chunk; a token the parser held back joins a later one, and
        // whatever is left goes on the finish chunk
        let pend = Arc::new(Mutex::new(Vec::<crate::generate::TokenProbs>::new()));
        let pend2 = pend.clone();
        let rows_seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let rows_seen2 = rows_seen.clone();
        let res = run_blocking(st2, prompt, gp, move |p: Partial| {
            pr.decoding.store(true, Ordering::Relaxed);
            fed2.fetch_add(p.text.len(), Ordering::Relaxed);
            let mut frames = Vec::new();
            if p.n_decoded == 1 {
                frames.push(api::chat_chunk(
                    &m2,
                    json!({"role": "assistant", "content": null}),
                    None,
                ));
            }
            rows_seen2.fetch_add(p.probs.len(), Ordering::Relaxed);
            pend2.lock().unwrap().extend(p.probs);
            let diffs = s2.lock().unwrap().feed(&p.text, false);
            match diffs {
                Ok(ds) => {
                    for d in ds {
                        let mut c = api::chat_chunk(&m2, d.to_delta_json(), None);
                        attach_logprobs(&mut c, &mut pend2.lock().unwrap());
                        frames.push(c);
                    }
                }
                Err(e) => {
                    let _ = send_frame(&tx2, api::sse(&api::error_json(500, &e)), stall);
                    return false;
                }
            }
            for f in frames {
                if !send_frame(&tx2, api::sse(&f), stall) {
                    return false;
                }
            }
            true
        })
        .await;
        let mut r = match res {
            Ok(r) => r,
            Err(GenError::Cancelled) => return,
            Err(e) => {
                let _ = send_final(&tx, api::sse(&gen_error_json(e)), stall).await;
                return;
            }
        };
        r.reasoning_tokens = reasoning_tokens(&tok_r, &r.tokens, &gen_prompt_r);
        // the rows of tokens after the last partial (a held-back UTF-8 tail, a stop word): every row
        // of the reply is streamed, in order, exactly once
        {
            let n = rows_seen.load(Ordering::Relaxed).min(r.probs.len());
            pend.lock().unwrap().extend(r.probs[n..].iter().cloned());
        }
        let (diffs, has_calls) = if prefill_only {
            (Ok(Vec::new()), false)
        } else {
            let mut s = sess.lock().unwrap();
            let d = s.feed("", true);
            let has = !s.msg().tool_calls.is_empty();
            (d, has)
        };
        let mut frames = Vec::new();
        if r.n_decoded == 0 {
            // H6: no token, so no partial sent the role chunk; the stream still opens with it
            frames.push(api::chat_chunk(
                &meta,
                json!({"role": "assistant", "content": null}),
                None,
            ));
        }
        match diffs {
            Ok(ds) => {
                for d in ds {
                    let mut c = api::chat_chunk(&meta, d.to_delta_json(), None);
                    attach_logprobs(&mut c, &mut pend.lock().unwrap());
                    frames.push(c);
                }
            }
            Err(e) => {
                let _ = send_final(&tx, api::sse(&api::error_json(500, &e)), stall).await;
                return;
            }
        }
        let mut fin = api::chat_chunk(
            &meta,
            json!({}),
            Some(api::finish_reason(r.stop, has_calls)),
        );
        attach_logprobs(&mut fin, &mut pend.lock().unwrap());
        frames.push(fin);
        if !ephemeral {
            st.echo_adopt(&prompt_kept, adopt.as_ref(), session.as_deref());
            if !prefill_only {
                let msg = sess.lock().unwrap().msg().clone();
                let delivered = fed.load(Ordering::Relaxed) == r.text.len();
                st.echo_record(&prompt_kept, &msg, &r, session.as_deref(), delivered);
            }
            spawn_persist(&st, session.clone(), req_no);
        }
        if !ephemeral && !prefill_only && api::finish_reason(r.stop, has_calls) == "stop" {
            let reply = sess.lock().unwrap().msg().to_json();
            spawn_idle_reprefill(&st, &b, reply, req_no);
        }
        if include_usage {
            frames.push(api::chat_usage_chunk(&meta, &r));
        }
        if let Some(Json::Object(last)) = frames.last_mut() {
            last.insert("timings".into(), api::timings_json(&r.timings));
        }
        if let Some(last) = frames.last_mut() {
            add_echo_timings(last, &echo_json);
        }
        for f in frames {
            if !send_final(&tx, api::sse(&f), stall).await {
                return;
            }
        }
        let _ = send_final(&tx, "data: [DONE]\n\n".to_string(), stall).await;
    });
    Reply::Stream(rx)
}
