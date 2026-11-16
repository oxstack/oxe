//! Request parameters and response JSON in llama-server's shapes (key order included:
//! serde_json is built with preserve_order).

use crate::generate::{GenParams, GenResult, StopType, Timings, TokenProbs};
use crate::sampling::SamplingParams;
use serde_json::{json, Map, Value as Json};

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The OpenAI error object `{"error":{"code","message","param","type"}}` for an HTTP status:
/// `code` a string (OpenAI's own where one exists), `param` null. See [`error_json_with`].
pub fn error_json(status: u16, message: &str) -> Json {
    let code = match status {
        400 => "invalid_request",
        401 => "invalid_api_key",
        403 => "forbidden",
        404 => "not_found",
        409 => "conflict",
        411 => "length_required",
        413 => "request_too_large",
        501 => "not_supported",
        503 => "unavailable",
        _ => "server_error",
    };
    error_json_with(status, message, code, None)
}

/// [`error_json`] with an explicit string `code` and `param` (the request field at fault).
pub fn error_json_with(status: u16, message: &str, code: &str, param: Option<&str>) -> Json {
    let ty = match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        501 => "not_supported_error",
        503 => "unavailable_error",
        _ => "server_error",
    };
    json!({"error": {"code": code, "message": message, "param": param, "type": ty}})
}

/// Context overflow: OpenAI's code `context_length_exceeded` (param `messages`); the type and the
/// message text stay llama-server's (`exceed_context_size_error`; clients match on them), plus
/// `n_prompt_tokens` / `n_ctx`.
pub fn exceed_context_json(n_prompt: usize, n_ctx: usize) -> Json {
    json!({"error": {
        "code": "context_length_exceeded",
        "message": format!("request ({n_prompt} tokens) exceeds the available context size ({n_ctx} tokens), try increasing it"),
        "param": "messages",
        "type": "exceed_context_size_error",
        "n_prompt_tokens": n_prompt,
        "n_ctx": n_ctx,
    }})
}

/// The request's own output limit, if it names one (`n_predict`, `max_completion_tokens`,
/// `max_tokens`; null = absent). Without one the server's default output limit applies.
pub fn has_length_field(body: &Json) -> bool {
    ["n_predict", "max_completion_tokens", "max_tokens"]
        .iter()
        .any(|k| body.get(*k).is_some_and(|v| !v.is_null()))
}

fn num<T: Copy>(body: &Json, keys: &[&str], f: impl Fn(&Json) -> Option<T>, def: T) -> T {
    for k in keys {
        if let Some(v) = body.get(*k) {
            if let Some(x) = f(v) {
                return x;
            }
        }
    }
    def
}

pub const MAX_N_PROBS: usize = 100;

/// REVIEW-v0.1.1 B2: out-of-range sampling values are a 400, not a server abort. Ranges: the
/// OpenAI API's where it has one (temperature 0..=2 is NOT enforced: llama-server treats any
/// value <= 0 as greedy and allows > 2), else llama-server's meaning: top_p / min_p / typical_p
/// in 0..=1, top_k an integer (<= 0 = off), penalties finite (presence / frequency -2..=2,
/// repeat >= 0). A present field of the wrong type is a 400 too (null = absent).
pub fn check_sampling(body: &Json) -> Result<(), String> {
    let get = |k: &str| body.get(k).filter(|v| !v.is_null());
    let float = |k: &str, lo: f64, hi: f64| -> Result<(), String> {
        let Some(v) = get(k) else { return Ok(()) };
        let x = v
            .as_f64()
            .ok_or_else(|| format!("\"{k}\" must be a number"))?;
        // the engine computes in f32: a value that is finite in f64 but not in f32 is out of range
        if !(x as f32).is_finite() || x < lo || x > hi {
            return Err(format!("\"{k}\" = {x} is out of range ({lo} to {hi})"));
        }
        Ok(())
    };
    let big = f32::MAX as f64;
    float("temperature", -big, big)?;
    float("top_p", 0.0, 1.0)?;
    float("min_p", 0.0, 1.0)?;
    float("typical_p", 0.0, 1.0)?;
    float("presence_penalty", -2.0, 2.0)?;
    float("frequency_penalty", -2.0, 2.0)?;
    float("repeat_penalty", 0.0, big)?;
    if let Some(v) = get("top_k") {
        match v.as_i64() {
            Some(k) if i32::try_from(k).is_ok() => {}
            _ => {
                return Err("\"top_k\" must be an integer in the 32-bit range (<= 0 = off)".into())
            }
        }
    }
    if let Some(v) = get("seed") {
        if v.as_i64().is_none() && v.as_u64().is_none() {
            return Err("\"seed\" must be an integer".into());
        }
    }
    Ok(())
}

/// OpenAI `logit_bias`: an object of token id (decimal string) -> bias in -100..=100, added to that
/// token's logit before sampling. Null or absent = none. Also llama-server's array form
/// `[[id, bias], ...]` (a `false` bias = never). Sorted by id, one entry per id.
pub fn logit_bias(body: &Json) -> Result<Vec<(u32, f32)>, String> {
    let bad = |m: String| Err(m);
    let mut out: Vec<(u32, f32)> = Vec::new();
    let mut put = |id: Option<u64>, b: Option<f64>| -> Result<(), String> {
        let id = id
            .and_then(|i| u32::try_from(i).ok())
            .ok_or("\"logit_bias\": a key is not a token id")?;
        let b = b.ok_or("\"logit_bias\": a bias is not a number")?;
        if !(-100.0..=100.0).contains(&b) && b != f64::NEG_INFINITY {
            return Err(format!(
                "\"logit_bias\" = {b} for token {id} is out of range (-100 to 100)"
            ));
        }
        out.retain(|e| e.0 != id);
        out.push((id, b as f32));
        Ok(())
    };
    match body.get("logit_bias") {
        None | Some(Json::Null) => {}
        Some(Json::Object(m)) => {
            for (k, v) in m {
                put(k.trim().parse::<u64>().ok(), v.as_f64())?;
            }
        }
        Some(Json::Array(a)) => {
            for e in a {
                let pair = e.as_array().filter(|p| p.len() == 2);
                let Some(p) = pair else {
                    return bad("\"logit_bias\": array entries are [id, bias]".into());
                };
                let b = if p[1] == Json::Bool(false) {
                    Some(f64::NEG_INFINITY)
                } else {
                    p[1].as_f64()
                };
                put(p[0].as_u64(), b)?;
            }
        }
        Some(_) => return bad("\"logit_bias\" must be an object of token id -> bias".into()),
    }
    // a zero bias changes nothing: dropped, so an all-zero map keeps the exact fast path
    out.retain(|e| e.1 != 0.0);
    out.sort_by_key(|e| e.0);
    Ok(out)
}

/// Caps on `stop` (OpenAI allows 4; generous, but bounded so a request cannot make every token scan huge lists).
const MAX_STOP_STRINGS: usize = 16;
const MAX_STOP_BYTES: usize = 1024;

/// Sampling / stopping parameters common to /completion and /v1/chat/completions
/// (`params_from_json_cmpl`).
pub fn gen_params(body: &Json, chat: bool) -> Result<GenParams, String> {
    let d = SamplingParams::default();
    let mut p = GenParams::default();
    p.n_predict = num(
        body,
        &["n_predict", "max_completion_tokens", "max_tokens"],
        Json::as_i64,
        -1,
    );
    if let Some(v) = body.get("n_predict").and_then(Json::as_i64) {
        p.n_predict = v;
    }
    if let Some(value) = body.get("thinking_budget").filter(|v| !v.is_null()) {
        if !chat {
            return Err("thinking_budget is supported on chat completions only".into());
        }
        p.thinking_budget = Some(
            value
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("thinking_budget must be a non-negative integer or null")?,
        );
        if p.n_predict == 1 {
            return Err(
                "thinking_budget needs at least two output tokens (closing marker and answer)"
                    .into(),
            );
        }
    }
    p.sampling = SamplingParams {
        temperature: num(
            body,
            &["temperature"],
            |v| v.as_f64().map(|x| x as f32),
            d.temperature,
        ),
        top_k: num(
            body,
            &["top_k"],
            |v| {
                v.as_i64()
                    .map(|x| x.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            },
            d.top_k,
        ),
        top_p: num(body, &["top_p"], |v| v.as_f64().map(|x| x as f32), d.top_p),
        min_p: num(body, &["min_p"], |v| v.as_f64().map(|x| x as f32), d.min_p),
        seed: num(body, &["seed"], |v| v.as_i64().map(|x| x as u64), d.seed),
        presence_penalty: num(
            body,
            &["presence_penalty"],
            |v| v.as_f64().map(|x| x as f32),
            0.0,
        ),
        frequency_penalty: num(
            body,
            &["frequency_penalty"],
            |v| v.as_f64().map(|x| x as f32),
            0.0,
        ),
        logit_bias: logit_bias(body)?,
    };
    check_sampling(body)?;
    p.stop = match body.get("stop") {
        Some(Json::String(s)) => vec![s.clone()],
        Some(Json::Array(a)) => a
            .iter()
            .filter_map(Json::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    };
    if p.stop.len() > MAX_STOP_STRINGS || p.stop.iter().any(|s| s.len() > MAX_STOP_BYTES) {
        return Err(format!(
            "\"stop\": at most {MAX_STOP_STRINGS} strings of {MAX_STOP_BYTES} bytes each"
        ));
    }
    p.cache_prompt = body
        .get("cache_prompt")
        .and_then(Json::as_bool)
        .unwrap_or(true);
    p.ignore_eos = body
        .get("ignore_eos")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    // per-request speculation switch (the controller's off-switch): "speculative": false, or a
    // depth via "spec_draft_n_max" (llama-server's flag name) / "speculative": {"n_max": n}
    p.spec_depth = match body.get("speculative") {
        Some(Json::Bool(false)) => Some(0),
        Some(Json::Object(o)) => o.get("n_max").and_then(Json::as_u64).map(|n| n as usize),
        _ => None,
    }
    .or_else(|| {
        body.get("spec_draft_n_max")
            .and_then(Json::as_u64)
            .map(|n| n as usize)
    });
    if chat {
        if body
            .get("logprobs")
            .and_then(Json::as_bool)
            .unwrap_or(false)
        {
            p.n_probs = body
                .get("top_logprobs")
                .and_then(Json::as_u64)
                .unwrap_or(20) as usize;
        } else if body
            .get("top_logprobs")
            .map(|v| !v.is_null())
            .unwrap_or(false)
        {
            return Err("top_logprobs requires logprobs to be set to true".into());
        }
        if let Some(n) = body.get("n_probs").and_then(Json::as_u64) {
            p.n_probs = n as usize;
        }
    } else {
        p.n_probs = num(body, &["n_probs", "logprobs"], Json::as_u64, 0) as usize;
    }
    // every generated token keeps n_probs entries (vocabulary-sized for a huge value): bound the
    // memory a request can ask for (OpenAI allows top_logprobs <= 20)
    if p.n_probs > MAX_N_PROBS {
        return Err(format!(
            "n_probs / top_logprobs = {} is above the limit {MAX_N_PROBS}",
            p.n_probs
        ));
    }
    Ok(p)
}

pub fn timings_json(t: &Timings) -> Json {
    let per_tok = |ms: f64, n: usize| if n > 0 { ms / n as f64 } else { 0.0 };
    let per_s = |ms: f64, n: usize| if ms > 0.0 { 1e3 / ms * n as f64 } else { 0.0 };
    let mut j = timings_base(t, per_tok, per_s);
    // request-path timings (ms), always reported: the queue, the route's render and tokenisation, the
    // prefix restore and the checkpoints the request took
    j["queue_ms"] = json!(t.queue_ms);
    j["render_ms"] = json!(t.render_ms);
    j["tokenize_ms"] = json!(t.tokenize_ms);
    j["restore_ms"] = json!(t.restore_ms);
    j["ckpt_ms"] = json!(t.ckpt_ms);
    if t.draft_n > 0 {
        // llama-server's speculative counters
        j["draft_n"] = json!(t.draft_n);
        j["draft_n_accepted"] = json!(t.draft_n_accepted);
    }
    // CACHE-PERFECT P2b (design sec. 7): where the reused prefix came from (live/ram/disk/cold)
    if !t.resume_src.is_empty() {
        j["resume_src"] = json!(t.resume_src);
        j["resume_ms"] = json!(t.resume_ms);
    }
    if let Some(ms) = t.preload_ms {
        j["preload_src"] = json!("disk");
        j["preload_ms"] = json!(ms);
    }
    // v0.2.0 H4 (`X-Oxe-Snapshot: keep`): the position kept, null when none could be (only with the hint)
    if t.snap_keep {
        j["snap_pos"] = json!(t.snap_pos);
        if let Some(why) = t.snap_reason {
            j["snap_reason"] = json!(why);
        }
    }
    // fair slicing (`fair.rs`): only for a generation that gave the engine away at least once
    if t.fair_yields > 0 {
        j["fair_yields"] = json!(t.fair_yields);
        j["fair_wait_ms"] = json!(t.fair_wait_ms);
    }
    j
}

fn timings_base(
    t: &Timings,
    per_tok: impl Fn(f64, usize) -> f64,
    per_s: impl Fn(f64, usize) -> f64,
) -> Json {
    json!({
        "cache_n": t.cache_n,
        "prompt_n": t.prompt_n,
        "prompt_ms": t.prompt_ms,
        "prompt_per_token_ms": per_tok(t.prompt_ms, t.prompt_n),
        "prompt_per_second": per_s(t.prompt_ms, t.prompt_n),
        "predicted_n": t.predicted_n,
        "predicted_ms": t.predicted_ms,
        "predicted_per_token_ms": per_tok(t.predicted_ms, t.predicted_n),
        "predicted_per_second": per_s(t.predicted_ms, t.predicted_n),
    })
}

pub fn usage_json(r: &GenResult) -> Json {
    json!({
        "completion_tokens": r.n_decoded,
        "prompt_tokens": r.n_prompt,
        "total_tokens": r.n_decoded + r.n_prompt,
        "prompt_tokens_details": {"cached_tokens": r.timings.cache_n},
        "completion_tokens_details": {"reasoning_tokens": r.reasoning_tokens},
    })
}

fn logprob(p: f32) -> f32 {
    if p == 0.0 {
        f32::MIN
    } else {
        p.ln()
    }
}

fn piece_str(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// `completion_token_output::probs_vector_to_json` (post_sampling_probs=false shape;
/// with post_sampling the keys become prob/top_probs).
pub fn probs_json(v: &[TokenProbs], post_sampling: bool) -> Json {
    let (pk, tk) = if post_sampling {
        ("prob", "top_probs")
    } else {
        ("logprob", "top_logprobs")
    };
    let val = |p: f32| {
        if post_sampling {
            Json::from(p)
        } else {
            Json::from(logprob(p))
        }
    };
    Json::Array(
        v.iter()
            .map(|t| {
                let mut o = Map::new();
                o.insert("id".into(), t.id.into());
                o.insert("token".into(), piece_str(&t.piece).into());
                o.insert(
                    "bytes".into(),
                    t.piece.iter().map(|&b| Json::from(b)).collect(),
                );
                o.insert(pk.into(), val(t.prob));
                o.insert(
                    tk.into(),
                    t.top
                        .iter()
                        .map(|e| {
                            let mut x = Map::new();
                            x.insert("id".into(), e.id.into());
                            x.insert("token".into(), piece_str(&e.piece).into());
                            x.insert(
                                "bytes".into(),
                                e.piece.iter().map(|&b| Json::from(b)).collect(),
                            );
                            x.insert(pk.into(), val(e.prob));
                            Json::Object(x)
                        })
                        .collect(),
                );
                Json::Object(o)
            })
            .collect(),
    )
}

pub fn finish_reason(stop: StopType, has_tool_calls: bool) -> &'static str {
    match stop {
        StopType::Word | StopType::Eos => {
            if has_tool_calls {
                "tool_calls"
            } else {
                "stop"
            }
        }
        _ => "length",
    }
}

pub struct ChatMeta {
    pub id: String,
    pub model: String,
    pub created: u64,
    pub fingerprint: String,
}

pub fn chat_response(
    m: &ChatMeta,
    msg: &chat::ChatMsg,
    r: &GenResult,
    n_probs: bool,
    post_sampling: bool,
) -> Json {
    let mut choice = Map::new();
    choice.insert(
        "finish_reason".into(),
        finish_reason(r.stop, !msg.tool_calls.is_empty()).into(),
    );
    choice.insert("index".into(), 0.into());
    choice.insert("message".into(), msg.to_json());
    if n_probs && !r.probs.is_empty() {
        choice.insert(
            "logprobs".into(),
            json!({"content": probs_json(&r.probs, post_sampling)}),
        );
    }
    json!({
        "choices": [Json::Object(choice)],
        "created": m.created,
        "model": m.model,
        "system_fingerprint": m.fingerprint,
        "object": "chat.completion",
        "usage": usage_json(r),
        "id": m.id,
        "timings": timings_json(&r.timings),
    })
}

pub fn chat_chunk(m: &ChatMeta, delta: Json, finish: Option<&str>) -> Json {
    json!({
        "choices": [{"finish_reason": finish, "index": 0, "delta": delta}],
        "created": m.created,
        "id": m.id,
        "model": m.model,
        "system_fingerprint": m.fingerprint,
        "object": "chat.completion.chunk",
    })
}

pub fn chat_usage_chunk(m: &ChatMeta, r: &GenResult) -> Json {
    json!({
        "choices": [],
        "created": m.created,
        "id": m.id,
        "model": m.model,
        "system_fingerprint": m.fingerprint,
        "object": "chat.completion.chunk",
        "usage": usage_json(r),
    })
}

/// native /completion final object (`to_json_non_oaicompat`)
pub fn completion_final(
    model: &str,
    prompt: &str,
    r: &GenResult,
    stream: bool,
    gp: &GenParams,
    return_tokens: bool,
    post_sampling: bool,
) -> Json {
    let mut o = Map::new();
    o.insert("index".into(), 0.into());
    o.insert(
        "content".into(),
        if stream {
            "".into()
        } else {
            r.text.clone().into()
        },
    );
    o.insert(
        "tokens".into(),
        if return_tokens && !stream {
            r.tokens.iter().map(|&t| Json::from(t)).collect()
        } else {
            Json::Array(vec![])
        },
    );
    o.insert("id_slot".into(), 0.into());
    o.insert("stop".into(), true.into());
    o.insert("model".into(), model.into());
    o.insert("tokens_predicted".into(), r.n_decoded.into());
    o.insert("tokens_evaluated".into(), r.n_prompt.into());
    o.insert("generation_settings".into(), generation_settings(gp));
    o.insert("prompt".into(), prompt.into());
    o.insert("has_new_line".into(), r.text.contains('\n').into());
    o.insert("truncated".into(), r.truncated.into());
    o.insert("stop_type".into(), r.stop.as_str().into());
    o.insert("stopping_word".into(), r.stopping_word.clone().into());
    o.insert(
        "tokens_cached".into(),
        (r.n_prompt + r.n_decoded.saturating_sub(1)).into(),
    );
    o.insert("timings".into(), timings_json(&r.timings));
    if !stream && !r.probs.is_empty() {
        o.insert(
            "completion_probabilities".into(),
            probs_json(&r.probs, post_sampling),
        );
    }
    Json::Object(o)
}

pub fn generation_settings(gp: &GenParams) -> Json {
    json!({
        "n_predict": gp.n_predict,
        "seed": gp.sampling.seed,
        "temperature": gp.sampling.temperature,
        "top_k": gp.sampling.top_k,
        "top_p": gp.sampling.top_p,
        "min_p": gp.sampling.min_p,
        "max_tokens": gp.n_predict,
        "n_probs": gp.n_probs,
        "stop": gp.stop,
        "ignore_eos": gp.ignore_eos,
        "cache_prompt": gp.cache_prompt,
    })
}

/// native /completion streamed partial
pub fn completion_partial(
    text: &str,
    n_decoded: usize,
    n_prompt: usize,
    probs: &[TokenProbs],
    post_sampling: bool,
) -> Json {
    let mut o = Map::new();
    o.insert("index".into(), 0.into());
    o.insert("content".into(), text.into());
    o.insert("tokens".into(), Json::Array(vec![]));
    o.insert("stop".into(), false.into());
    o.insert("id_slot".into(), 0.into());
    o.insert("tokens_predicted".into(), n_decoded.into());
    o.insert("tokens_evaluated".into(), n_prompt.into());
    if !probs.is_empty() {
        o.insert(
            "completion_probabilities".into(),
            probs_json(probs, post_sampling),
        );
    }
    Json::Object(o)
}

/// The message of a refused `n` above 1.
const N_REFUSED: &str =
    "\"n\" > 1 is not supported: this local model server returns one choice per request. \
Send n=1 (or omit it) and repeat the request for more choices.";

/// `n` (choices per request). Absent, null or 1 is the one choice this server returns. An integer
/// above 1 is refused (`unsupported_value`); any other value is `invalid_value`. Both are 400 with
/// `param: "n"`. Checked on the chat and the legacy completions routes.
pub fn check_n(body: &Json) -> Result<(), Json> {
    let int = match body.get("n") {
        None | Some(Json::Null) => return Ok(()),
        Some(Json::Number(x)) => x
            .as_i64()
            .map(i128::from)
            .or_else(|| x.as_u64().map(i128::from)),
        Some(_) => None,
    };
    match int {
        Some(1) => Ok(()),
        Some(v) if v > 1 => Err(error_json_with(
            400,
            N_REFUSED,
            "unsupported_value",
            Some("n"),
        )),
        _ => Err(error_json_with(
            400,
            "\"n\" must be a positive integer.",
            "invalid_value",
            Some("n"),
        )),
    }
}

/// 32 lower-case hex digits from two `RandomState` hashes. `RandomState::new()` takes its keys from
/// the OS random source (and bumps a per-thread counter), so the digits are unpredictable to a
/// client and differ on every call. No new dependency. Used for ids, never for secrets.
pub fn random_hex() -> String {
    use std::hash::{BuildHasher, Hasher};
    let one = || {
        std::collections::hash_map::RandomState::new()
            .build_hasher()
            .finish()
    };
    format!("{:016x}{:016x}", one(), one())
}

/// OpenAI legacy `logprobs` object for generated rows: parallel arrays, `top_logprobs` a map from
/// token text to logprob, at most `top_n` candidates per row (0: the chosen token's logprob only).
/// `offset` is the character position of the first row in the reply text;
/// it is advanced past each row's token, so rows sent in several pieces keep one running offset.
pub fn legacy_logprobs(rows: &[TokenProbs], top_n: usize, offset: &mut usize) -> Json {
    let (mut tokens, mut lps, mut tops, mut offs) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for t in rows {
        let token = piece_str(&t.piece);
        offs.push(Json::from(*offset));
        *offset += token.chars().count();
        lps.push(Json::from(logprob(t.prob)));
        let mut top = Map::new();
        for e in t.top.iter().take(top_n) {
            top.insert(piece_str(&e.piece), Json::from(logprob(e.prob)));
        }
        tops.push(Json::Object(top));
        tokens.push(Json::from(token));
    }
    json!({"tokens": tokens, "token_logprobs": lps, "top_logprobs": tops, "text_offset": offs})
}

/// One `text_completion` stream chunk (OpenAI legacy completions): the text of one piece, the
/// finish reason on the last choice chunk, the logprobs of the piece's rows (null when none).
pub fn text_chunk(m: &ChatMeta, text: &str, finish: Option<&str>, logprobs: Option<Json>) -> Json {
    json!({
        "id": m.id,
        "object": "text_completion",
        "created": m.created,
        "model": m.model,
        "system_fingerprint": m.fingerprint,
        "choices": [{"text": text, "index": 0, "logprobs": logprobs, "finish_reason": finish}],
    })
}

/// The `include_usage` chunk of a legacy stream: no choices, the usage of the whole reply.
pub fn text_usage_chunk(m: &ChatMeta, r: &GenResult) -> Json {
    json!({
        "id": m.id,
        "object": "text_completion",
        "created": m.created,
        "model": m.model,
        "system_fingerprint": m.fingerprint,
        "choices": [],
        "usage": usage_json(r),
    })
}

/// The non-streamed OpenAI legacy completion (`text_completion`).
pub fn text_completion(m: &ChatMeta, text: &str, r: &GenResult, logprobs: Option<Json>) -> Json {
    let mut v = text_chunk(m, text, Some(finish_reason(r.stop, false)), logprobs);
    v["usage"] = usage_json(r);
    v["timings"] = timings_json(&r.timings);
    v
}

pub fn sse(v: &Json) -> String {
    format!("data: {}\n\n", serde_json::to_string(v).unwrap())
}
