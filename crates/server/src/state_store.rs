//! CACHE-PERFECT P2: the disk state store.
//!
//! A conversation's engine state is kept as immutable 256-token KV blocks on a token hash chain
//! plus snapshots (recurrent + conv state, MTP driver state, KV rows of the partial block) at a
//! few positions, and one manifest per session. Sessions that share a prefix (the static system
//! prompt + tools, a fork, a retry, a subagent) share its blocks. A restore is a byte copy of
//! states this engine computed, so the resumed run equals the uninterrupted one.
//!
//! Identity I = SHA-256 of: model GGUF sha256, state ABI (`Engine::state_id`: layout, KV dtype,
//! MTP head), [`NUMERICS_EPOCH`], prefill ubatch, the effective arithmetic flags
//! (`Engine::arith_id`: the arithmetic knobs and the kernel path, [`kernel_path`]) and `rope=gguf` (oxe has no rope scaling: the rope config is the GGUF's,
//! covered by the model sha). The binary's own hash is NOT in it (O-96): installs keep their
//! caches; `numerics_epoch` is bumped by hand and a golden-state test in CI fails when the
//! numerics change without a bump (`model::tests::numerics_epoch_golden_state`). The build sha goes in
//! `identity.json`; `--state-cache-strict-build` puts it in the identity (gates).
//!
//! Chain: h_0 = I, h_i = SHA-256(h_{i-1}, tokens of block i) (hex). Snapshot key at position p =
//! SHA-256(h_k, tokens[256k..p]) with k = p / 256 (h_0 = I when k = 0).
//!
//! Layout under `ROOT/<model_key>/` (model_key = first 16 hex of I; dirs 0700, files 0600):
//! - `identity.json`: version, identity text and hash, model sha, numerics epoch, build sha.
//! - `blocks/<hh>/<h>.<c>.kv`: block with chain hash h, content checksum c (16 hex Check64 of
//!   the payload). The same chain computed along another path (other ubatch boundaries, decode vs
//!   prefill) has other bytes, so another `c`: states of different origin are never mixed.
//! - `snaps/<key>.<c>.st`: one snapshot.
//! - `sessions/<sid>.json`: manifest (blocks [[h, c]], snaps [{pos, class, key, c, clean, bytes}],
//!   n_tokens, created, atime, hint hash). Every snapshot in a manifest is consistent with the
//!   manifest's block list (an edit that re-computes blocks drops the snapshots above it).
//! - `echo/<group>.jsonl`: echo records (echo.rs, M3).
//! Block / snapshot file = magic (`OXEBLK01` / `OXESNP01`), u32 version, 64 hex identity hash,
//! u64 position (block start / snapshot position), u64 payload length, u64 payload Check64, then
//! the payload (block: `kv_rows_save` of 256 rows; snapshot: u64 kv bytes, `kv_rows_save` of the
//! partial block, `snap_save`). Everything is verified on load; a bad file is deleted with the
//! manifest entries that need it, the engine is reset and the lookup falls back.
//!
//! Crash consistency: every file is written to `.pending`, fsync'd, renamed, then the directory is
//! fsync'd; manifests are written last. Start-up sweeps `.pending` files and blocks / snapshots no
//! manifest references, and removes the directories of older identities of the same model.
//!
//! Budget: one byte budget over every model under ROOT (`--state-budget-bytes`, runtime.json
//! `cache.disk_budget_bytes`, default 64 GB). Eviction: TTL-expired sessions first, then the
//! lowest value = Cost(n) / bytes x recency, Cost(p) = 3.0 p + 0.112 p^2 / 1000 ms; sessions used
//! in the last 10 minutes are pinned; blocks shared with a kept session stay (refcount), so a kept
//! block always has its prefix. v0.1.3 (`--state-budget-scope global`, the default): the loaded
//! engine evicts across EVERY identity under ROOT (other models, older epochs still in their TTL),
//! in one value order over optional snapshots and whole sessions (a session's head, its deepest
//! state, goes only with the session); each identity's shared blocks go last (refcount within that
//! identity).
//!
//! v0.1.3 off-lock writes ([`Staged`]): after a reply the head snapshot and the new blocks are copied
//! to RAM under the worker lock, written without it (a request arriving meanwhile runs at once), and
//! the manifest is committed under the lock again. Files on their way are listed in `inflight`, so
//! a sweep in between never takes them.
//!
//! Privacy: state files encode the conversation. No prompt text or token ids in logs or names.

use crate::engine::{Engine, EngineError};
use crate::prefix_store::{check64, sha256_hex, Check64, Sha256};
use serde_json::{json, Value as Json};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// The numerics epoch (defined next to the golden-state test that guards it: `model::engine`).
pub use model::engine::NUMERICS_EPOCH;
/// Positions per block (one prefill ubatch of the MoE profiles).
pub const BLOCK: usize = 256;
const MAGIC_BLOCK: &[u8; 8] = b"OXEBLK01";
const MAGIC_SNAP: &[u8; 8] = b"OXESNP01";
const VERSION: u32 = 1;
/// snapshot file format: 2 = the MTP driver state in full (P2b, G3); P2's version 1 is refused
const SNAP_VERSION: u32 = 2;
const HDR: u64 = 8 + 4 + 64 + 8 + 8 + 8;
/// bytes of a state file header (a snapshot file is this + its payload)
pub const HDR_BYTES: u64 = HDR;
/// snapshots kept per session at most (design sec. 2; the default of `--state-snaps`)
pub const MAX_SNAPS: usize = 12;
/// an off-lock write stages at most this many new blocks in RAM (88 MiB at Ornith's row size);
/// more go through the on-lock writer (a long prefill without progress writes)
pub const STAGE_MAX_BLOCKS: usize = 16;
/// a manifest larger than this is refused unread (and removed): never an unbounded read
pub const MAX_MANIFEST_BYTES: u64 = 8 << 20;
/// sessions used this recently are never evicted
const PIN_SECS: u64 = 600;

/// Environment knobs that select an arithmetic path of the oxe engine.
pub(crate) const ARITH_KNOBS: &[&str] = &[
    "OXE_ATTN",
    "OXE_ATTN_TILED_MIN_KV",
    "OXE_DEC_ATTN_PAD",
    "OXE_DEC_FUSED_DN",
    "OXE_KV_VPAD",
    "OXE_PF_ATTN_PB",
    "OXE_PF_GROUP",
    "OXE_PF_GROUP_T",
    "OXE_PF_GROUP_UNALIGNED",
    "OXE_PF_MMID_MIN",
    "OXE_PF_TILES",
    "OXE_SMALL_PREFILL",
    "OXE_VL",
    "OXE_KV",
    "OXE_KV_TYPE",
    // P2 review: the knobs the first list missed (`--pf-group-t1/-t2` set the first two)
    "OXE_PF_GROUP_T1",
    "OXE_PF_GROUP_T2",
    "OXE_KV8_FUSED",
    "OXE_KV8_PV",
    // v0.1.3 (lane A, PREFILL-LONG): the prefill attention path and the fused-attention cutoffs.
    // An engine that picks fa2 by itself (auto at 256K, no variable set) must key on its actual
    // path too: OxeEngine::arith_id appends ` pf_attn=fa2` from `Engine::pf_attn_fa2()`
    "OXE_PF_ATTN",
    "OXE_ATTN_FUSED_MIN_KV",
    "OXE_ATTN_FUSED_1LANE_MIN_KV",
    // v0.1.3 APPEND-PREFILL: the fa2x kernel's f16 chain length and chunk (arith_id also appends
    // ` pf_attn=fa2x.f<fold>.c<ch>` from the engine's effective parameters)
    "OXE_PF_FA2X_FOLD",
    "OXE_PF_FA2X_CH",
    // v0.1.3 PREFILL-POLICY: what `auto` resolves to (arith_id also appends the resolved kernel
    // and `.k<cut>` when the cutoff is not PFA_MIN_KV)
    "OXE_PF_ATTN_AUTO_MIN_KV",
    "OXE_PF_ATTN_AUTO_KERNEL",
    // v0.1.4 PFG: the pfg kernel's folds, chunk, unit size and micro-kernel (arith_id also appends
    // ` pf_attn=pfg.<tag>` from the engine's effective parameters)
    "OXE_PF_PFG_FOLD",
    "OXE_PF_PFG_CH",
    "OXE_PF_PFG_TB",
    "OXE_PF_PFG_MK",
    // NOT here (v0.1.4 DEC-PB): OXE_ATTN_OUT / --attn-out (legacy | pb). The two V^T . p kernels
    // are bit-identical (kernels-sys dec_pb_*, model mini_dec_pb_attn_out_is_bit_identical), so
    // the state identity must stay the same and states move freely between them.
];

/// Environment knobs that select an arithmetic path of the oxe engine (sorted `K=V`, set ones only).
pub fn arith_env() -> String {
    arith_env_with(|k| std::env::var(k).ok())
}

/// [`arith_env`] over a given reader (tests: no process-wide environment change, which the
/// engines of parallel tests would read at construction).
pub(crate) fn arith_env_with(get: impl Fn(&str) -> Option<String>) -> String {
    ARITH_KNOBS
        .iter()
        .filter_map(|k| get(k).map(|v| format!("{k}={v}")))
        .collect::<Vec<_>>()
        .join(",")
}

/// The kernel path (moved to `model::engine` next to the golden-state test that keys on it).
pub use model::engine::kernel_path;

/// One `key=value` field of an identity text (`state=` runs up to ` epoch=`: the layout has spaces).
fn ident_field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let start = text.find(&format!("{key}="))? + key.len() + 1;
    let rest = &text[start..];
    let end = if key == "state" {
        rest.find(" epoch=")?
    } else {
        rest.find(' ').unwrap_or(rest.len())
    };
    Some(&rest[..end])
}

/// Whether another identity directory's `identity.json` can never be read by any build of this
/// store format: another format version, numerics epoch, state ABI or block size. Unknown or
/// unparseable fields count as stale (a directory this build cannot even describe).
pub(crate) fn stale_identity(other: &Json, mine: &Identity) -> bool {
    if other.get("version").and_then(Json::as_u64) != Some(VERSION as u64) {
        return true;
    }
    if other.get("numerics_epoch") != Some(&json!(NUMERICS_EPOCH)) {
        return true;
    }
    let Some(text) = other.get("identity").and_then(Json::as_str) else {
        return true;
    };
    ["state", "block"].iter().any(|k| {
        ident_field(text, k).is_none() || ident_field(text, k) != ident_field(&mine.text, k)
    })
}

/// Re-prefill cost of p tokens in ms (design sec. 4, [E]).
pub fn cost_ms(p: usize) -> f64 {
    let p = p as f64;
    3.0 * p + 0.112 * p * p / 1000.0
}

pub fn now_s() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Clone, Debug)]
pub struct Identity {
    pub text: String,
    /// SHA-256 hex of `text` (h_0 of every chain)
    pub hash: String,
    pub model: String,
    pub build: String,
}

impl Identity {
    /// `model`: the GGUF sha256 (or a file id), `state_id`: Engine::state_id, `arith`:
    /// Engine::arith_id; `build`: the binary's sha (in the identity only when `strict_build`).
    pub fn new(
        model: &str,
        state_id: &str,
        ubatch: usize,
        arith: &str,
        build: &str,
        strict_build: bool,
    ) -> Identity {
        let mut text = format!("model={model} state={state_id} epoch={NUMERICS_EPOCH} ubatch={ubatch} block={BLOCK} rope=gguf arith=[{arith}]");
        if strict_build {
            text.push_str(&format!(" build={build}"));
        }
        Identity {
            hash: sha256_hex(text.as_bytes()),
            text,
            model: model.to_string(),
            build: build.to_string(),
        }
    }
    pub fn model_key(&self) -> &str {
        &self.hash[..16]
    }
}

/// The file stem shared by a session's manifest `hint` field and its echo file
/// (`echo/<stem>.jsonl`): the first 32 hex of SHA-256(session hint). echo.rs names its group files
/// with it and the controller's GC (`state_cache.py`) finds a session's echo file by it.
pub fn hint_hash(hint: &str) -> String {
    sha256_hex(hint.as_bytes())[..32].to_string()
}

fn hex_name(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn tok_bytes(t: &[u32]) -> Vec<u8> {
    t.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Chain hashes h_1..h_k of the full blocks of `tokens` (index i holds h_{i+1}).
pub fn chain(id_hash: &str, tokens: &[u32]) -> Vec<String> {
    let mut prev = id_hash.to_string();
    tokens
        .as_chunks::<BLOCK>()
        .0
        .iter()
        .map(|b| {
            let mut s = Sha256::default();
            s.update(prev.as_bytes());
            s.update(&tok_bytes(b));
            prev = s.hex();
            prev.clone()
        })
        .collect()
}

/// Snapshot key at position `pos` (see the module docs); `hs` = chain(tokens) at least to pos/256.
pub fn snap_key(id_hash: &str, hs: &[String], tokens: &[u32], pos: usize) -> String {
    let k = pos / BLOCK;
    let base = if k == 0 { id_hash } else { hs[k - 1].as_str() };
    let mut s = Sha256::default();
    s.update(base.as_bytes());
    s.update(&(pos as u64).to_le_bytes());
    s.update(&tok_bytes(&tokens[k * BLOCK..pos]));
    s.hex()
}

#[derive(Clone, Debug, PartialEq)]
pub struct SnapRef {
    pub pos: usize,
    /// anchor | head | turn | query | tail
    pub class: String,
    pub key: String,
    pub c: String,
    pub clean: bool,
    pub bytes: u64,
}

#[derive(Clone, Debug)]
pub struct Manifest {
    pub sid: String,
    pub hint: Option<String>,
    /// v0.2.0 EG5: the hint hashes of every request hint whose conversation this session holds (the
    /// creator first). A session continued or absorbed under another hint keeps the earlier owners;
    /// a hint clear deletes the session only when it is the sole owner, else drops that owner.
    /// A request without a hint owns as [`ANON_OWNER`], so a hinted request continuing an unhinted
    /// conversation never becomes its sole owner.
    pub owners: Vec<String>,
    pub created: u64,
    pub atime: u64,
    pub n_tokens: usize,
    pub blocks: Vec<(String, String)>,
    pub snaps: Vec<SnapRef>,
}

impl Manifest {
    fn to_json(&self, id_hash: &str) -> Json {
        json!({
            "version": VERSION, "identity_hash": id_hash, "sid": self.sid, "hint": self.hint, "owners": self.owners, "created": self.created,
            "atime": self.atime, "n_tokens": self.n_tokens,
            "blocks": self.blocks.iter().map(|(h, c)| json!([h, c])).collect::<Vec<_>>(),
            "snaps": self.snaps.iter().map(|s| json!({"pos": s.pos, "class": s.class, "key": s.key, "c": s.c, "clean": s.clean, "bytes": s.bytes})).collect::<Vec<_>>(),
        })
    }
    fn from_json(v: &Json, id_hash: &str) -> Option<Manifest> {
        if v.get("version")?.as_u64()? != VERSION as u64
            || v.get("identity_hash")?.as_str()? != id_hash
        {
            return None;
        }
        let hex = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit());
        let blocks = v
            .get("blocks")?
            .as_array()?
            .iter()
            .map(|b| {
                Some((
                    b.get(0)?.as_str()?.to_string(),
                    b.get(1)?.as_str()?.to_string(),
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        let snaps = v
            .get("snaps")?
            .as_array()?
            .iter()
            .map(|s| {
                Some(SnapRef {
                    pos: s.get("pos")?.as_u64()? as usize,
                    class: s.get("class")?.as_str()?.to_string(),
                    key: s.get("key")?.as_str()?.to_string(),
                    c: s.get("c")?.as_str()?.to_string(),
                    clean: s.get("clean")?.as_bool()?,
                    bytes: s.get("bytes")?.as_u64()?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        if !blocks.iter().all(|(h, c)| hex(h, 64) && hex(c, 16))
            || !snaps
                .iter()
                .all(|s| hex(&s.key, 64) && hex(&s.c, 16) && s.pos / BLOCK <= blocks.len())
        {
            return None;
        }
        let sid = v.get("sid")?.as_str()?.to_string();
        if !hex(&sid, 32) {
            return None;
        }
        let hint = v.get("hint").and_then(Json::as_str).map(str::to_string);
        // owners: the recorded ones (32 hex each), else (an older manifest) its hint
        let mut owners: Vec<String> = match v.get("owners").and_then(Json::as_array) {
            Some(a) => a
                .iter()
                .filter_map(Json::as_str)
                .filter(|h| hex(h, 32) || *h == ANON_OWNER)
                .map(str::to_string)
                .collect(),
            None => Vec::new(),
        };
        if owners.is_empty() {
            // an older manifest: its hint, or (no hint) the anonymous owner
            match hint.as_ref().filter(|h| hex(h, 32)) {
                Some(h) => owners.push(h.clone()),
                None => owners.push(ANON_OWNER.to_string()),
            }
        }
        Some(Manifest {
            sid,
            hint,
            owners,
            created: v.get("created")?.as_u64()?,
            atime: v.get("atime")?.as_u64()?,
            n_tokens: v.get("n_tokens")?.as_u64()? as usize,
            blocks,
            snaps,
        })
    }
}

/// v0.2.0 EG5: the owner a request without a session hint writes as (never a hint hash: those are
/// 32 hex), so a later hinted continuation shares the session instead of owning it alone.
pub const ANON_OWNER: &str = "-";

/// v0.2.0 EG5: the owners of a session written now: every owner of the manifests it continues or
/// absorbs (in that order, the creator first; one without owners counts as its hint, else
/// anonymous), then the request's own (its hint hash, or [`ANON_OWNER`] without a hint).
/// `request`: None = no request identity to add (a commit merging recorded owners).
pub fn merge_owners<'a>(
    from: impl Iterator<Item = &'a Manifest>,
    request: Option<Option<String>>,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut add = |o: String| {
        if !out.contains(&o) {
            out.push(o);
        }
    };
    for m in from {
        if m.owners.is_empty() {
            add(m.hint.clone().unwrap_or_else(|| ANON_OWNER.to_string()));
        }
        for o in &m.owners {
            add(o.clone());
        }
    }
    if let Some(r) = request {
        add(r.unwrap_or_else(|| ANON_OWNER.to_string()));
    }
    out
}

/// A lookup result: load these blocks, then this snapshot, to be at `snap.pos`.
#[derive(Clone, Debug)]
pub struct Plan {
    pub sid: String,
    pub blocks: Vec<(String, String)>,
    pub snap: SnapRef,
}

#[derive(Clone, Debug, Default)]
pub struct StoreStats {
    pub loads: u64,
    pub load_ms: f64,
    pub load_errors: u64,
    pub blocks_written: u64,
    pub snaps_written: u64,
    pub bytes_written: u64,
    pub evicted_bytes: u64,
}

/// Cached sizes, shared with off-lock writers. Only startup and explicit cache management
/// reconcile external changes; normal writes replace a path's size (never double-count a shared
/// block or an overwritten snapshot). No file descriptors or state payloads are retained.
#[derive(Clone, Default)]
pub(crate) struct DiskUsage(std::sync::Arc<std::sync::Mutex<DiskFiles>>);

#[derive(Default)]
struct DiskFiles {
    bytes: u64,
    sizes: HashMap<PathBuf, u64>,
}

impl DiskFiles {
    fn set(&mut self, path: PathBuf, size: Option<u64>) {
        if let Some(old) = self.sizes.remove(&path) {
            self.bytes = self.bytes.saturating_sub(old);
        }
        if let Some(size) = size {
            self.bytes += size;
            self.sizes.insert(path, size);
        }
    }
}

impl DiskUsage {
    fn lock(&self) -> std::sync::MutexGuard<'_, DiskFiles> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn set(&self, path: &Path, size: Option<u64>) {
        self.lock().set(path.to_path_buf(), size);
    }

    pub(crate) fn refresh(&self, path: &Path) {
        match fs::symlink_metadata(path) {
            Ok(m) if m.is_file() => self.set(path, Some(m.len())),
            Ok(_) => self.set(path, None),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => self.set(path, None),
            Err(_) => {}
        }
    }

    fn resync(&self, root: &Path) {
        fn scan(p: &Path, files: &mut DiskFiles) {
            let Ok(rd) = read_dir(p) else { return };
            for d in rd.flatten() {
                match d.file_type() {
                    Ok(t) if t.is_dir() => scan(&d.path(), files),
                    Ok(t) if t.is_file() => {
                        if let Ok(m) = d.metadata() {
                            files.set(d.path(), Some(m.len()));
                        }
                    }
                    _ => {}
                }
            }
        }
        // Serialize with off-lock publication: its final size update is idempotent even if
        // the scan already saw the renamed file.
        let mut files = self.lock();
        *files = DiskFiles::default();
        scan(root, &mut files);
    }
}

pub struct StateStore {
    pub(crate) usage: DiskUsage,
    pub root: PathBuf,
    pub dir: PathBuf,
    pub ident: Identity,
    /// bytes over every model under `root` (0 = disk tier off)
    pub budget: u64,
    pub ttl_secs: u64,
    /// sessions shorter than this are not written (title requests, tiny chats)
    pub min_tokens: usize,
    pub manifests: HashMap<String, Manifest>,
    pub stats: StoreStats,
    /// a write is skipped when the file system has less than this free
    pub min_free_bytes: u64,
    /// a write is skipped while MemAvailable is below this (P2b, M2: the save floor; a write holds
    /// at most a 4 MiB buffer and one 5.5 MiB block, never a state-sized one)
    pub min_mem_bytes: u64,
    /// MemAvailable reader (tests replace it)
    pub mem_available: fn() -> Option<u64>,
    /// crash-test hook (see [`Kill`])
    pub kill: Kill,
    /// v0.1.3: snapshots kept per session at most (`--state-snaps`, default [`MAX_SNAPS`])
    pub max_snaps: usize,
    /// v0.1.3: the budget evicts across every identity under the root (`--state-budget-scope global`)
    pub global_budget: bool,
    /// v0.1.3: files an off-lock write is putting on disk that no manifest names yet (sweeps keep them)
    pub inflight: std::sync::Arc<std::sync::Mutex<HashSet<PathBuf>>>,
    /// test hook: an off-lock write sleeps this long before it writes (a slow disk)
    pub stage_delay_ms: u64,
    /// mutation-test hook (0 in production): 1 = no sweep on every enforce, 2 = a live block a sweep
    /// removed counts as on disk, 3 = sweeps ignore `inflight`. The v0.1.3 tests check that each
    /// broken sweep fails the tests that guard it.
    pub mutant: u8,
}

fn io(e: std::io::Error) -> String {
    e.to_string()
}

fn mkdir(p: &Path) -> Result<(), String> {
    fs::create_dir_all(p).map_err(io)?;
    fs::set_permissions(p, fs::Permissions::from_mode(0o700)).map_err(io)
}

pub(crate) fn fsync_dir(p: &Path) {
    #[cfg(test)]
    DIR_SYNCS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(p.to_path_buf());
    if let Ok(f) = fs::File::open(p) {
        let _ = f.sync_all();
    }
}

/// Test record of every directory fsync (tests filter by their own temp directory).
#[cfg(test)]
pub(crate) static DIR_SYNCS: std::sync::Mutex<Vec<PathBuf>> = std::sync::Mutex::new(Vec::new());

/// Crash-test hook (design sec. 8, "kill at each write step"): every file-system step of the
/// store (a file's payload written, fsync'd, renamed; a removal) counts one step; at step `at` the
/// process "dies": that step and every later one fail and leave the disk as a kill would (a
/// half-written `.pending` stays, a removal does not happen). Off (`at` None) outside tests.
#[derive(Default)]
pub struct Kill {
    pub at: Option<usize>,
    pub steps: std::sync::atomic::AtomicUsize,
}

const KILLED: &str = "killed (test hook)";

impl Kill {
    fn step(&self) -> Result<(), String> {
        let n = self.steps.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match self.at {
            Some(a) if n >= a => Err(KILLED.into()),
            _ => Ok(()),
        }
    }
    /// steps taken so far
    pub fn taken(&self) -> usize {
        self.steps.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Remove a file (one kill step). True when it is gone now.
fn rm(k: &Kill, usage: &DiskUsage, p: &Path) -> bool {
    if k.step().is_err() {
        return false;
    }
    match fs::remove_file(p) {
        Ok(()) => {
            usage.set(p, None);
            true
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                usage.set(p, None);
            }
            false
        }
    }
}

/// Write a file atomically: `<tmp>.pending` (0600, O_NOFOLLOW) streamed through a 4 MiB
/// buffer, fsync, rename to `final_name(payload checksum)` in `dir`, directory fsync. The payload
/// goes through a checksum while it is written (no state-sized buffer: the shutdown write's
/// memory floor), and `header(len, sum)` (HDR bytes, or 0 for a plain file) is written over the
/// placeholder at the start before the fsync. Returns (final path, payload length, checksum).
/// Kill steps: payload written, fsync'd, renamed.
#[allow(clippy::too_many_arguments)]
fn atomic_stream(
    k: &Kill,
    usage: &DiskUsage,
    dir: &Path,
    tmp: &str,
    hdr_len: u64,
    header: &dyn Fn(u64, u64) -> Vec<u8>,
    final_name: &dyn Fn(u64) -> String,
    write: &mut dyn FnMut(&mut dyn Write) -> Result<(), String>,
) -> Result<(PathBuf, u64, u64), String> {
    use std::io::{Seek, SeekFrom};
    let pending = dir.join(format!("{tmp}.pending"));
    let mut published = None;
    let r = (|| -> Result<(PathBuf, u64, u64), String> {
        let f = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&pending)
            .map_err(io)?;
        let mut w = BufWriter::with_capacity(4 << 20, f);
        w.write_all(&vec![0u8; hdr_len as usize]).map_err(io)?;
        let mut sw = Sum {
            w: &mut w,
            c: Check64::default(),
        };
        write(&mut sw)?;
        let (len, sum) = (sw.c.len, sw.c.finish());
        let mut f = w.into_inner().map_err(|e| e.into_error().to_string())?;
        if hdr_len > 0 {
            f.seek(SeekFrom::Start(0)).map_err(io)?;
            f.write_all(&header(len, sum)).map_err(io)?;
        }
        if let Err(e) = k.step() {
            // killed while the page cache held the data: half of it reached the disk
            let _ = f.set_len((hdr_len + len) / 2);
            return Err(e);
        }
        f.sync_all().map_err(io)?;
        // SAFETY: valid fd; advisory (drop the written pages: no state in the page cache).
        unsafe {
            libc::posix_fadvise(
                std::os::fd::AsRawFd::as_raw_fd(&f),
                0,
                0,
                libc::POSIX_FADV_DONTNEED,
            );
        }
        k.step()?;
        let path = dir.join(final_name(sum));
        fs::rename(&pending, &path).map_err(io)?;
        published = Some((path.clone(), hdr_len + len));
        k.step()?;
        fsync_dir(dir);
        Ok((path, len, sum))
    })();
    if let Some((path, size)) = published {
        usage.set(&path, Some(size));
        usage.set(&pending, None);
    } else if r.is_err() {
        if !matches!(&r, Err(e) if e == KILLED) {
            let _ = fs::remove_file(&pending);
        }
        // A kill or a failed unlink may leave a partial file: count it until a sweep succeeds.
        usage.refresh(&pending);
    }
    r
}

/// [`atomic_stream`] of a small plain file at `path` (manifests, identity.json).
fn atomic_write(k: &Kill, usage: &DiskUsage, path: &Path, data: &[u8]) -> Result<(), String> {
    let dir = path.parent().ok_or("no parent")?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("bad name")?
        .to_string();
    atomic_stream(
        k,
        usage,
        dir,
        &name,
        0,
        &|_, _| Vec::new(),
        &|_| name.clone(),
        &mut |w| w.write_all(data).map_err(io),
    )
    .map(|_| ())
}

/// Writer that checksums what passes through.
struct Sum<'a> {
    w: &'a mut dyn Write,
    c: Check64,
}

impl Write for Sum<'_> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        let n = self.w.write(b)?;
        self.c.update(&b[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

/// Reader over exactly `left` payload bytes that checksums them.
struct Checked<'a> {
    r: &'a mut dyn Read,
    left: u64,
    c: Check64,
}

impl Read for Checked<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = (buf.len() as u64).min(self.left) as usize;
        if n == 0 {
            return Ok(0);
        }
        let k = self.r.read(&mut buf[..n])?;
        self.c.update(&buf[..k]);
        self.left -= k as u64;
        Ok(k)
    }
}

fn version_of(magic: &[u8; 8]) -> u32 {
    if magic == MAGIC_SNAP {
        SNAP_VERSION
    } else {
        VERSION
    }
}

fn header(magic: &[u8; 8], id_hash: &str, pos: u64, len: u64, sum: u64) -> Vec<u8> {
    let mut h = Vec::with_capacity(HDR as usize);
    h.extend_from_slice(magic);
    h.extend_from_slice(&version_of(magic).to_le_bytes());
    h.extend_from_slice(id_hash.as_bytes());
    for x in [pos, len, sum] {
        h.extend_from_slice(&x.to_le_bytes());
    }
    h
}

/// Open a state file, check its header against (magic, identity, position) and its length;
/// returns the reader positioned at the payload, the payload length and checksum.
fn open_checked(
    path: &Path,
    magic: &[u8; 8],
    id_hash: &str,
    pos: u64,
) -> Result<(BufReader<fs::File>, u64, u64), String> {
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(io)?;
    let file_len = f.metadata().map_err(io)?.len();
    let mut r = BufReader::with_capacity(4 << 20, f);
    let mut h = vec![0u8; HDR as usize];
    r.read_exact(&mut h)
        .map_err(|_| "truncated header".to_string())?;
    let u64at = |o: usize| u64::from_le_bytes(h[o..o + 8].try_into().unwrap());
    if &h[..8] != magic || u32::from_le_bytes(h[8..12].try_into().unwrap()) != version_of(magic) {
        return Err("bad magic or version".into());
    }
    if &h[12..76] != id_hash.as_bytes() {
        return Err("identity mismatch".into());
    }
    if u64at(76) != pos {
        return Err("position mismatch".into());
    }
    let (len, sum) = (u64at(84), u64at(92));
    if file_len != HDR + len {
        return Err(format!(
            "length {file_len} != header {HDR} + payload {len} (truncated?)"
        ));
    }
    Ok((r, len, sum))
}

impl StateStore {
    /// Open the store of identity `ident` under `root` (created 0700). `root` must lie inside the
    /// user's home (the encrypted home, design sec. 6) unless `allow_anywhere` (tests).
    pub fn open(
        root: &Path,
        ident: Identity,
        budget: u64,
        ttl_secs: u64,
        allow_anywhere: bool,
    ) -> Result<StateStore, String> {
        if !root.is_absolute() {
            return Err("state cache: the directory must be absolute".into());
        }
        if !allow_anywhere {
            let home = std::env::var_os("HOME")
                .map(PathBuf::from)
                .ok_or("state cache: HOME is not set")?;
            let home = fs::canonicalize(&home).map_err(io)?;
            let mut probe = root.to_path_buf();
            let base = loop {
                if let Ok(c) = fs::canonicalize(&probe) {
                    break c;
                }
                probe = probe.parent().ok_or("state cache: bad path")?.to_path_buf();
            };
            if !base.starts_with(&home) {
                return Err(format!(
                    "state cache: {} is outside the home directory; refused",
                    root.display()
                ));
            }
        }
        mkdir(root)?;
        let dir = root.join(ident.model_key());
        for sub in ["blocks", "snaps", "sessions", "echo"] {
            mkdir(&dir.join(sub))?;
        }
        let idj = json!({"version": VERSION, "identity": ident.text, "identity_hash": ident.hash, "model": ident.model, "numerics_epoch": NUMERICS_EPOCH, "build": ident.build});
        let idp = dir.join("identity.json");
        if fs::read(&idp)
            .ok()
            .and_then(|b| serde_json::from_slice::<Json>(&b).ok())
            != Some(idj.clone())
        {
            atomic_write(
                &Kill::default(),
                &DiskUsage::default(),
                &idp,
                idj.to_string().as_bytes(),
            )?;
        }
        let usage = DiskUsage::default();
        usage.resync(root);
        let mut st = StateStore {
            usage,
            root: root.to_path_buf(),
            dir,
            ident,
            budget,
            ttl_secs,
            min_tokens: 1024,
            manifests: HashMap::new(),
            stats: StoreStats::default(),
            min_free_bytes: 5 << 30,
            min_mem_bytes: 512 << 20,
            mem_available: crate::prefix_store::proc_mem_available,
            kill: Kill::default(),
            max_snaps: MAX_SNAPS,
            // `--state-budget-scope own` (main.rs sets the variable before the store opens)
            global_budget: std::env::var("OXE_STATE_BUDGET_SCOPE").map_or(true, |v| v != "own"),
            inflight: Default::default(),
            stage_delay_ms: 0,
            mutant: 0,
        };
        st.load_manifests();
        st.sweep();
        st.orphan_sweep();
        st.enforce(None);
        Ok(st)
    }

    fn block_path(&self, h: &str, c: &str) -> PathBuf {
        block_path_in(&self.dir, h, c)
    }
    fn snap_path(&self, key: &str, c: &str) -> PathBuf {
        snap_path_in(&self.dir, key, c)
    }
    fn manifest_path(&self, sid: &str) -> PathBuf {
        self.dir.join("sessions").join(format!("{sid}.json"))
    }

    /// Is this snapshot file on disk?
    pub fn has_snap(&self, key: &str, c: &str) -> bool {
        self.snap_path(key, c).is_file()
    }

    fn load_manifests(&mut self) {
        let Ok(rd) = read_dir(self.dir.join("sessions")) else {
            return;
        };
        for de in rd.flatten() {
            let p = de.path();
            if p.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            // bounded (Codex P2/P3 review): a manifest is a few KiB; anything over 8 MiB is not ours
            let small = fs::symlink_metadata(&p)
                .is_ok_and(|m| m.is_file() && m.len() <= MAX_MANIFEST_BYTES);
            let m = small
                .then(|| fs::read(&p).ok())
                .flatten()
                .and_then(|b| serde_json::from_slice::<Json>(&b).ok())
                .and_then(|v| Manifest::from_json(&v, &self.ident.hash));
            match m {
                Some(m) if p.file_stem().and_then(|s| s.to_str()) == Some(m.sid.as_str()) => {
                    self.manifests.insert(m.sid.clone(), m);
                }
                // unreadable or another identity's: a bad manifest is a miss, never a guess
                _ => {
                    rm(&self.kill, &self.usage, &p);
                }
            }
        }
    }

    /// Delete `.pending` files and blocks / snapshots no manifest references (files of an off-lock
    /// write on its way are kept: `inflight`).
    pub fn sweep(&mut self) -> u64 {
        if self.mutant == 3 {
            return sweep_dir(
                &self.dir,
                &self.manifests,
                &self.kill,
                &self.usage,
                &Default::default(),
                &|| false,
            );
        }
        sweep_dir(
            &self.dir,
            &self.manifests,
            &self.kill,
            &self.usage,
            &self.inflight,
            &|| false,
        )
    }

    /// Remove the directories of STALE identities of the same model: another store format
    /// (`version`), numerics epoch, state ABI (`state=` layout) or block size. Nothing can ever
    /// read them. Same-epoch variants (other arithmetic knobs, deep/max context modes, tiled/fa2
    /// prefill, ubatch) are kept: another start may use them again; the global budget and the TTL
    /// age them out (v0.1.3 integration decision). Returns the bytes freed (logged).
    pub fn orphan_sweep(&mut self) -> u64 {
        let mut freed = 0;
        let Ok(rd) = read_dir(&self.root) else {
            return 0;
        };
        for de in rd.flatten() {
            let p = de.path();
            if p == self.dir || !de.file_type().is_ok_and(|t| t.is_dir()) {
                continue;
            }
            let id = fs::read(p.join("identity.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<Json>(&b).ok());
            let same_model = id
                .as_ref()
                .and_then(|v| v.get("model"))
                .and_then(Json::as_str)
                == Some(self.ident.model.as_str());
            if same_model && id.as_ref().is_some_and(|v| stale_identity(v, &self.ident)) {
                let b = dir_bytes(&p);
                if fs::remove_dir_all(&p).is_ok() {
                    let mut files = self.usage.lock();
                    let removed: Vec<PathBuf> = files
                        .sizes
                        .keys()
                        .filter(|f| f.starts_with(&p))
                        .cloned()
                        .collect();
                    for f in removed {
                        files.set(f, None);
                    }
                    freed += b;
                    eprintln!(
                        "oxe-server: state cache: removed an older identity of this model ({} MiB)",
                        b >> 20
                    );
                }
            }
        }
        freed
    }

    /// Cached bytes of every model's state under the root (one budget over all models).
    pub fn used_bytes(&self) -> u64 {
        self.usage.lock().bytes
    }

    /// Reconcile external writers/deletions at an explicit cache-management boundary.
    pub fn refresh_usage(&self) -> u64 {
        self.usage.resync(&self.root);
        self.used_bytes()
    }

    /// Bytes per model directory (model_key -> bytes).
    pub fn usage(&self) -> Json {
        self.usage.resync(&self.root);
        let mut models = serde_json::Map::new();
        if let Ok(rd) = read_dir(&self.root) {
            for de in rd
                .flatten()
                .filter(|d| d.file_type().is_ok_and(|t| t.is_dir()))
            {
                models.insert(
                    de.file_name().to_string_lossy().into_owned(),
                    json!(dir_bytes(&de.path())),
                );
            }
        }
        let now = now_s();
        let sessions: Vec<Json> = self
            .manifests
            .values()
            .map(|m| json!({"sid": m.sid, "n_tokens": m.n_tokens, "atime": m.atime, "snaps": m.snaps.len(), "blocks": m.blocks.len(), "bytes": self.session_bytes(m), "pinned": now.saturating_sub(m.atime) < PIN_SECS}))
            .collect();
        json!({
            "budget_bytes": self.budget, "used_bytes": self.used_bytes(), "models": models, "model_key": self.ident.model_key(),
            "sessions": sessions, "ttl_secs": self.ttl_secs, "numerics_epoch": NUMERICS_EPOCH,
            "stats": {"loads": self.stats.loads, "load_ms": self.stats.load_ms, "load_errors": self.stats.load_errors, "blocks_written": self.stats.blocks_written,
                      "snaps_written": self.stats.snaps_written, "bytes_written": self.stats.bytes_written, "evicted_bytes": self.stats.evicted_bytes},
        })
    }

    fn session_bytes(&self, m: &Manifest) -> u64 {
        let blocks: u64 = m
            .blocks
            .iter()
            .filter_map(|(h, c)| fs::metadata(self.block_path(h, c)).ok())
            .map(|md| md.len())
            .sum();
        blocks + m.snaps.iter().map(|s| s.bytes).sum::<u64>()
    }

    /// Longest-match lookup: the deepest snapshot (<= `limit`, > `beat`) whose blocks are the
    /// prompt's chain and whose key matches the prompt's tokens.
    pub fn lookup(&self, prompt: &[u32], limit: usize, beat: usize) -> Option<Plan> {
        if self.manifests.is_empty() || limit <= beat {
            return None;
        }
        let hs = chain(&self.ident.hash, &prompt[..limit]);
        let mut best: Option<(usize, &Manifest, &SnapRef)> = None;
        for m in self.manifests.values() {
            let matched = m
                .blocks
                .iter()
                .zip(&hs)
                .take_while(|((h, _), x)| h == *x)
                .count();
            for s in &m.snaps {
                if s.pos > limit
                    || s.pos <= beat
                    || s.pos / BLOCK > matched
                    || best.is_some_and(|b| b.0 >= s.pos)
                {
                    continue;
                }
                if snap_key(&self.ident.hash, &hs, prompt, s.pos) == s.key {
                    best = Some((s.pos, m, s));
                }
            }
        }
        best.map(|(pos, m, s)| Plan {
            sid: m.sid.clone(),
            blocks: m.blocks[..pos / BLOCK].to_vec(),
            snap: s.clone(),
        })
    }

    /// Load `plan` into the engine (reset first). On error the bad file and every manifest entry
    /// that needs it are dropped; the engine state is then undefined (the caller resets it).
    pub fn load<E: Engine>(&mut self, eng: &mut E, plan: &Plan) -> Result<f64, String> {
        self.load_cancellable(eng, plan, None)
    }

    /// The deepest snapshot of the newest session (of the sessions with hint hash `hint` when
    /// one is given and has any): the preload's choice (P2b).
    pub fn newest_plan(&self, hint: Option<&str>) -> Option<Plan> {
        let pick = |f: &dyn Fn(&Manifest) -> bool| {
            self.manifests
                .values()
                .filter(|m| f(m) && !m.snaps.is_empty())
                .max_by_key(|m| (m.atime, m.n_tokens, m.sid.clone()))
                .cloned()
        };
        let m = hint
            .and_then(|h| pick(&|m: &Manifest| m.hint.as_deref() == Some(h)))
            .or_else(|| pick(&|_| true))?;
        let snap = m.snaps.iter().max_by_key(|s| s.pos)?.clone();
        Some(Plan {
            sid: m.sid.clone(),
            blocks: m.blocks[..snap.pos / BLOCK].to_vec(),
            snap,
        })
    }

    /// [`StateStore::load`] that stops between blocks when `cancel` rises (Err "cancelled"; the
    /// engine state is then undefined: the caller resets it). Each block is ~5 ms at 5.5 MiB.
    pub fn load_cancellable<E: Engine>(
        &mut self,
        eng: &mut E,
        plan: &Plan,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<f64, String> {
        let t0 = std::time::Instant::now();
        eng.reset();
        let id = self.ident.hash.clone();
        for (i, (h, c)) in plan.blocks.iter().enumerate() {
            if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
                return Err("state cache: load cancelled".into());
            }
            let p = self.block_path(h, c);
            let a = i * BLOCK;
            let r = (|| -> Result<(), String> {
                let (mut r, len, sum) = open_checked(&p, MAGIC_BLOCK, &id, a as u64)?;
                if len != (BLOCK * eng.kv_row_bytes()) as u64 {
                    return Err("block payload of another size".into());
                }
                let mut cr = Checked {
                    r: &mut r,
                    left: len,
                    c: Check64::default(),
                };
                eng.kv_rows_load(a, a + BLOCK, &mut cr).map_err(|e| e.0)?;
                if cr.left != 0 || cr.c.finish() != sum || format!("{sum:016x}") != *c {
                    return Err("block checksum mismatch".into());
                }
                Ok(())
            })();
            if let Err(e) = r {
                self.drop_bad_block(h, c);
                self.stats.load_errors += 1;
                return Err(format!("state cache: block {i}: {e}"));
            }
        }
        let s = &plan.snap;
        let p = self.snap_path(&s.key, &s.c);
        let r = (|| -> Result<(), String> {
            let (mut r, len, sum) = open_checked(&p, MAGIC_SNAP, &id, s.pos as u64)?;
            let mut cr = Checked {
                r: &mut r,
                left: len,
                c: Check64::default(),
            };
            let mut u = [0u8; 8];
            cr.read_exact(&mut u).map_err(io)?;
            let kvb = u64::from_le_bytes(u);
            let a = s.pos / BLOCK * BLOCK;
            if kvb != ((s.pos - a) * eng.kv_row_bytes()) as u64 {
                return Err("snapshot rows of another size".into());
            }
            if s.pos > a {
                eng.kv_rows_load(a, s.pos, &mut cr).map_err(|e| e.0)?;
            }
            eng.snap_load(&mut cr, s.pos).map_err(|e| e.0)?;
            if cr.left != 0 || cr.c.finish() != sum || format!("{sum:016x}") != s.c {
                return Err("snapshot checksum mismatch".into());
            }
            Ok(())
        })();
        if let Err(e) = r {
            rm(&self.kill, &self.usage, &p);
            for m in self.manifests.values_mut() {
                m.snaps.retain(|x| !(x.key == s.key && x.c == s.c));
            }
            self.rewrite_all();
            self.stats.load_errors += 1;
            return Err(format!("state cache: snapshot: {e}"));
        }
        if eng.n_past() != s.pos {
            return Err("state cache: the engine is not at the snapshot position".into());
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        self.stats.loads += 1;
        self.stats.load_ms += ms;
        if let Some(m) = self.manifests.get_mut(&plan.sid) {
            m.atime = now_s();
            let m = m.clone();
            let _ = self.write_manifest(&m);
        }
        Ok(ms)
    }

    fn drop_bad_block(&mut self, h: &str, c: &str) {
        rm(&self.kill, &self.usage, &self.block_path(h, c));
        for m in self.manifests.values_mut() {
            if let Some(i) = m.blocks.iter().position(|(x, y)| x == h && y == c) {
                m.blocks.truncate(i);
                m.snaps.retain(|s| s.pos / BLOCK <= i);
            }
        }
        self.rewrite_all();
    }

    fn rewrite_all(&mut self) {
        let ms: Vec<Manifest> = self.manifests.values().cloned().collect();
        for m in ms {
            if m.snaps.is_empty() {
                self.remove_session(&m.sid);
            } else {
                let _ = self.write_manifest(&m);
            }
        }
    }

    fn write_manifest(&self, m: &Manifest) -> Result<(), String> {
        let b = m.to_json(&self.ident.hash).to_string();
        atomic_write(
            &self.kill,
            &self.usage,
            &self.manifest_path(&m.sid),
            b.as_bytes(),
        )
    }

    fn remove_session(&mut self, sid: &str) {
        remove_manifest_in(&self.dir, &mut self.manifests, sid, &self.kill, &self.usage);
    }

    fn free_space_ok(&self) -> bool {
        // SAFETY: statvfs on a valid C path into a zeroed struct.
        unsafe {
            let Ok(c) = std::ffi::CString::new(self.dir.as_os_str().as_encoded_bytes()) else {
                return false;
            };
            let mut s: libc::statvfs = std::mem::zeroed();
            if libc::statvfs(c.as_ptr(), &mut s) != 0 {
                return true;
            }
            (s.f_bavail as u64).saturating_mul(s.f_frsize as u64) >= self.min_free_bytes
        }
    }

    /// MemAvailable at or above the save floor (unknown: yes).
    pub fn mem_ok(&self) -> bool {
        self.min_mem_bytes == 0 || (self.mem_available)().is_none_or(|m| m >= self.min_mem_bytes)
    }

    pub(crate) fn write_allowed(&self) -> Result<(), String> {
        if !self.free_space_ok() {
            return Err("disk free below the floor: write skipped".into());
        }
        if !self.mem_ok() {
            return Err(format!(
                "MemAvailable below the save floor ({} MiB): write skipped",
                self.min_mem_bytes >> 20
            ));
        }
        Ok(())
    }

    /// Write block `i` of the live state (positions [256i, 256i+256)) unless present; returns its c.
    /// `buf` holds the one block (5.5 MiB at Ornith's row size): its checksum names the file, so a
    /// block already on disk (the shared prefix) is never written twice.
    pub fn write_block<E: Engine>(
        &mut self,
        eng: &mut E,
        h: &str,
        i: usize,
        buf: &mut Vec<u8>,
    ) -> Result<String, String> {
        let a = i * BLOCK;
        buf.clear();
        eng.kv_rows_save(a, a + BLOCK, buf).map_err(|e| e.0)?;
        let sum = check64(buf);
        let c = format!("{sum:016x}");
        let p = self.block_path(h, &c);
        if p.exists() {
            return Ok(c);
        }
        self.write_allowed()?;
        let sub = p.parent().unwrap().to_path_buf();
        if !sub.is_dir() {
            mkdir(&sub)?;
            // the new directory entry must be durable before a manifest names a block in it
            fsync_dir(&self.dir.join("blocks"));
        }
        let id = self.ident.hash.clone();
        let name = format!("{h}.{c}.kv");
        let data: &[u8] = buf;
        atomic_stream(
            &self.kill,
            &self.usage,
            &sub,
            &format!("{h}.w.kv"),
            HDR,
            &|len, sum| header(MAGIC_BLOCK, &id, a as u64, len, sum),
            &|_| name.clone(),
            &mut |w| w.write_all(data).map_err(io),
        )?;
        self.stats.blocks_written += 1;
        self.stats.bytes_written += HDR + buf.len() as u64;
        Ok(c)
    }

    /// Write a snapshot at `pos`: KV rows of the partial block from the live state (unchanged below
    /// the live position), then the checkpoint's non-KV state (`snap`, or the live state when
    /// None, written straight from the engine's buffers). Streamed: the file is written through a
    /// 4 MiB buffer and checksummed on the way (P2b, M2: no 63-190 MiB snapshot buffer, so the
    /// shutdown write fits under memory pressure). Returns (c, file bytes).
    pub fn write_snap<E: Engine>(
        &mut self,
        eng: &mut E,
        key: &str,
        pos: usize,
        snap: Option<&E::Snapshot>,
    ) -> Result<(String, u64), String> {
        self.write_snap_c(eng, key, pos, snap, None)
    }

    /// [`StateStore::write_snap`] that gives up inside the file when `cancel` rises (v0.1.3: a request
    /// that arrives during a snapshot write waits for at most one 4 MiB buffer, not the whole file);
    /// the half-written `.pending` is removed and nothing is published.
    pub fn write_snap_c<E: Engine>(
        &mut self,
        eng: &mut E,
        key: &str,
        pos: usize,
        snap: Option<&E::Snapshot>,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<(String, u64), String> {
        self.write_snap_until(eng, key, pos, snap, cancel, None)
    }

    /// [`StateStore::write_snap_c`] that also gives up inside the file once `deadline` has passed
    /// (audit v0.1.2 N2: the shutdown snapshot had no bound inside the file; the controller's
    /// unload grace then had to end it with SIGKILL). Same cut: the `.pending` is removed and
    /// nothing is published.
    pub fn write_snap_until<E: Engine>(
        &mut self,
        eng: &mut E,
        key: &str,
        pos: usize,
        snap: Option<&E::Snapshot>,
        cancel: Option<&std::sync::atomic::AtomicBool>,
        deadline: Option<std::time::Instant>,
    ) -> Result<(String, u64), String> {
        self.write_allowed()?;
        let id = self.ident.hash.clone();
        let dir = self.dir.join("snaps");
        let (_, len, sum) = atomic_stream(
            &self.kill,
            &self.usage,
            &dir,
            &format!("{key}.w.st"),
            HDR,
            &|len, sum| header(MAGIC_SNAP, &id, pos as u64, len, sum),
            &|sum| format!("{key}.{sum:016x}.st"),
            &mut |w| {
                let mut cw = Cancellable {
                    w,
                    cancel,
                    deadline,
                };
                snap_payload(eng, pos, snap, &mut cw)
            },
        )?;
        let bytes = HDR + len;
        self.stats.snaps_written += 1;
        self.stats.bytes_written += bytes;
        Ok((format!("{sum:016x}"), bytes))
    }

    /// v0.1.3 off-lock write: block `i` of the live state copied to RAM (checksum hex, payload; no
    /// payload when that exact file is on disk already).
    pub fn stage_block<E: Engine>(
        &self,
        eng: &mut E,
        h: &str,
        i: usize,
    ) -> Result<(String, Option<Vec<u8>>), String> {
        let a = i * BLOCK;
        let mut buf = Vec::with_capacity(BLOCK * eng.kv_row_bytes());
        eng.kv_rows_save(a, a + BLOCK, &mut buf).map_err(|e| e.0)?;
        let c = format!("{:016x}", check64(&buf));
        Ok(if self.block_path(h, &c).exists() {
            (c, None)
        } else {
            (c, Some(buf))
        })
    }

    /// v0.1.3 off-lock write: the live state's snapshot payload at `pos` (= the engine position)
    /// copied to RAM: (checksum hex, payload), the bytes [`StateStore::write_snap`] would stream.
    pub fn stage_snap<E: Engine>(
        &self,
        eng: &mut E,
        pos: usize,
    ) -> Result<(String, Vec<u8>), String> {
        let mut buf = Vec::new();
        snap_payload(eng, pos, None, &mut buf)?;
        Ok((format!("{:016x}", check64(&buf)), buf))
    }

    /// Store / replace a session's manifest (written last: blocks and snapshots are on disk).
    pub fn put_manifest(&mut self, m: Manifest) -> Result<(), String> {
        self.write_manifest(&m)?;
        self.manifests.insert(m.sid.clone(), m);
        Ok(())
    }

    /// Is this block file on disk (a sweep removes blocks no manifest names yet)?
    pub fn has_block(&self, h: &str, c: &str) -> bool {
        self.mutant == 2 || self.block_path(h, c).is_file()
    }

    pub fn manifest(&self, sid: &str) -> Option<&Manifest> {
        self.manifests.get(sid)
    }

    pub fn drop_session(&mut self, sid: &str) {
        self.remove_session(sid);
    }

    /// [`SessionMatch`] of a stored session (None: unknown id).
    pub fn session_match(&self, sid: &str) -> Option<SessionMatch> {
        let m = self.manifests.get(sid)?;
        let head = m.snaps.iter().max_by_key(|s| s.pos)?;
        let shared = self
            .manifests
            .values()
            .filter(|o| o.sid != sid)
            .map(|o| {
                o.blocks
                    .iter()
                    .zip(&m.blocks)
                    .take_while(|(a, b)| a.0 == b.0)
                    .count()
            })
            .max()
            .unwrap_or(0);
        Some(SessionMatch {
            id_hash: self.ident.hash.clone(),
            blocks: m.blocks.iter().map(|b| b.0.clone()).collect(),
            head: (head.pos, head.key.clone()),
            shared,
            hint: m.hint.clone(),
        })
    }

    /// v0.2.0 EG5: the stored sessions the request hint with this hint hash owns (alone or with
    /// others; `Manifest::owners`), in id order.
    pub fn sessions_with_hint(&self, h: &str) -> Vec<String> {
        let mut v: Vec<String> = self
            .manifests
            .values()
            .filter(|m| m.owners.iter().any(|o| o == h) || m.hint.as_deref() == Some(h))
            .map(|m| m.sid.clone())
            .collect();
        v.sort();
        v
    }

    /// v0.2.0 EG5: drop hint hash `h` from session `sid`'s owners when another hint still owns it
    /// (the manifest is rewritten; its `hint` moves to the earliest remaining owner) and return
    /// true; false when `h` is its sole owner (the caller purges the session) or it is unknown.
    pub fn release_owner(&mut self, sid: &str, h: &str) -> Result<bool, String> {
        let Some(m) = self.manifests.get(sid) else {
            return Ok(false);
        };
        let rest: Vec<String> = m.owners.iter().filter(|o| *o != h).cloned().collect();
        if rest.is_empty() {
            return Ok(false);
        }
        let mut m = m.clone();
        if m.hint.as_deref() == Some(h) || m.hint.is_none() {
            m.hint = rest.iter().find(|o| o.as_str() != ANON_OWNER).cloned();
        }
        m.owners = rest;
        self.put_manifest(m)?;
        Ok(true)
    }

    /// Is any manifest still naming this hint hash?
    pub fn hint_in_use(&self, h: &str) -> bool {
        self.manifests
            .values()
            .any(|m| m.hint.as_deref() == Some(h))
    }

    /// Set a new budget and evict down to it at once; returns the bytes freed.
    pub fn set_budget(&mut self, bytes: u64) -> u64 {
        self.usage.resync(&self.root);
        self.budget = bytes;
        self.enforce(None)
    }

    /// Clear: `all` (every model under the root), `model` (this model) or one `session`.
    pub fn clear(&mut self, scope: &str, sid: Option<&str>) -> Result<u64, String> {
        self.usage.resync(&self.root);
        let before = self.used_bytes();
        match scope {
            "session" => {
                let sid = sid.ok_or("session scope needs a session id")?;
                if !self.manifests.contains_key(sid) {
                    return Err("unknown session".into());
                }
                self.remove_session(sid);
                self.sweep();
            }
            "model" | "all" => {
                let sids: Vec<String> = self.manifests.keys().cloned().collect();
                for s in sids {
                    self.remove_session(&s);
                }
                self.sweep();
                if let Ok(rd) = read_dir(self.dir.join("echo")) {
                    for d in rd.flatten() {
                        let _ = fs::remove_file(d.path());
                    }
                }
                if scope == "all" {
                    if let Ok(rd) = read_dir(&self.root) {
                        for d in rd.flatten().filter(|d| {
                            d.path() != self.dir && d.file_type().is_ok_and(|t| t.is_dir())
                        }) {
                            // only directories in this store's format (identity.json present)
                            if d.path().join("identity.json").is_file() {
                                let _ = fs::remove_dir_all(d.path());
                            }
                        }
                    }
                }
            }
            _ => return Err("scope must be all, model or session".into()),
        }
        self.usage.resync(&self.root);
        Ok(before.saturating_sub(self.used_bytes()))
    }

    /// TTL, then the budget: evict by value until every model's bytes fit. With `global_budget`
    /// (v0.1.3) every identity under the root is a candidate, else only this one's sessions (the
    /// others are only counted). Order: TTL, then one value order over the optional snapshots and
    /// whole sessions (store-depth: re-prefill lost x recency per byte still needed; a head, anchor
    /// or cut goes only with its session); a block shared with a kept session of its identity
    /// stays. `keep`: a session never evicted now (the one just written). Returns bytes freed.
    pub fn enforce(&mut self, keep: Option<&str>) -> u64 {
        self.enforce_cancellable(keep, None)
    }

    /// Budget maintenance yields between sessions and eviction steps when a request arrives.
    /// Already-published manifests and their referenced files remain valid when interrupted.
    pub fn enforce_cancellable(
        &mut self,
        keep: Option<&str>,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> u64 {
        self.enforce_until(keep, &|| {
            cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        })
    }

    fn enforce_until(&mut self, keep: Option<&str>, stop: &dyn Fn() -> bool) -> u64 {
        if stop() {
            return 0;
        }
        let before = self.used_bytes();
        let now = now_s();
        // audit v0.1.2 N3: a last use in the future (the wall clock stepped back) would read as age
        // 0 until the clock caught up, pinning the session against TTL and budget eviction; it is
        // last used "now" instead and ages normally from here (persisted, so a restart agrees)
        let future: Vec<String> = self
            .manifests
            .values()
            .filter(|m| m.atime > now)
            .map(|m| m.sid.clone())
            .collect();
        for sid in future {
            if stop() {
                break;
            }
            if let Some(m) = self.manifests.get_mut(&sid) {
                m.atime = now;
            }
            let _ = self.write_manifest(&self.manifests[&sid]);
        }
        let expired: Vec<String> = self
            .manifests
            .values()
            .filter(|m| now.saturating_sub(m.atime) > self.ttl_secs && Some(m.sid.as_str()) != keep)
            .map(|m| m.sid.clone())
            .collect();
        for s in &expired {
            if stop() {
                break;
            }
            self.remove_session(s);
        }
        // Retire replaced heads before choosing a live session to evict. This sweep visits the
        // in-memory size index, not the directory tree (Progress can defer it while under budget).
        if self.mutant != 1 {
            let empty = Default::default();
            let inflight = if self.mutant == 3 {
                &empty
            } else {
                &*self.inflight
            };
            sweep_dir(
                &self.dir,
                &self.manifests,
                &self.kill,
                &self.usage,
                inflight,
                stop,
            );
        }
        let mut used = self.used_bytes();
        if used > self.budget && !stop() {
            let mut tiers = vec![Tier {
                dir: self.dir.clone(),
                id_hash: self.ident.hash.clone(),
                manifests: std::mem::take(&mut self.manifests),
            }];
            if self.global_budget {
                for mut t in foreign_tiers(&self.root, &self.dir, stop) {
                    if stop() {
                        break;
                    }
                    for m in t.manifests.values_mut().filter(|m| m.atime > now) {
                        if stop() {
                            break;
                        }
                        m.atime = now;
                        let _ = write_manifest_in(&t.dir, &t.id_hash, m, &self.kill, &self.usage);
                    }
                    // their TTL too: an older epoch's sessions are nobody else's to expire
                    let gone: Vec<String> = t
                        .manifests
                        .values()
                        .filter(|m| now.saturating_sub(m.atime) > self.ttl_secs)
                        .map(|m| m.sid.clone())
                        .collect();
                    for s in &gone {
                        if stop() {
                            break;
                        }
                        remove_manifest_in(&t.dir, &mut t.manifests, s, &self.kill, &self.usage);
                    }
                    // A cancelled eviction may have published a manifest removal (or snapshot
                    // thinning) without unlinking its files. Reclaim those indexed orphans before
                    // selecting another live session, even when no manifest expires this pass.
                    sweep_dir(
                        &t.dir,
                        &t.manifests,
                        &self.kill,
                        &self.usage,
                        &self.inflight,
                        stop,
                    );
                    tiers.push(t);
                }
                used = self.used_bytes();
            }
            let mut eviction: Vec<Eviction> = tiers
                .iter()
                .map(|t| Eviction::new(t, &self.usage))
                .collect();
            let pinned = |ti: usize, m: &Manifest| {
                (ti == 0 && Some(m.sid.as_str()) == keep) || now.saturating_sub(m.atime) < PIN_SECS
            };
            // v0.1.3 store-depth: one value order over optional snapshots and whole sessions (the
            // v0.1.2 order thinned EVERY unpinned session, a deep recent one too, before any shallow
            // old session went). Each step drops the item that loses the least re-prefill per byte
            // still needed: loss = Cost(depth it saves) x recency, / min(bytes it frees, bytes over
            // budget) (a 2.3 GB session is not dropped to free 60 MB while a 60 MB point would do).
            // An optional snapshot saves Cost(pos) - Cost(next point below it in its session); a
            // session saves Cost(n_tokens). A session's head (its deepest state), anchor and cut
            // points go only with the whole session.
            while used > self.budget && !stop() {
                let need = (used - self.budget) as f64;
                let mut best: Option<(f64, usize, String, Option<usize>)> = None;
                for (ti, t) in tiers.iter().enumerate() {
                    for m in t.manifests.values().filter(|m| !pinned(ti, m)) {
                        let rec = recency(m, now);
                        let own = eviction[ti].own.get(&m.sid).copied().unwrap_or(0).max(1) as f64;
                        let mut cands = vec![(cost_ms(m.n_tokens) * rec / own.min(need), None)];
                        let head = m.snaps.iter().map(|x| x.pos).max().unwrap_or(0);
                        for x in m
                            .snaps
                            .iter()
                            .filter(|x| x.pos != head && x.class != "anchor" && x.class != "cut")
                        {
                            let below = m
                                .snaps
                                .iter()
                                .map(|y| y.pos)
                                .filter(|&p| p < x.pos)
                                .max()
                                .unwrap_or(0);
                            cands.push((
                                (cost_ms(x.pos) - cost_ms(below)) * rec
                                    / (x.bytes.max(1) as f64).min(need),
                                Some(x.pos),
                            ));
                        }
                        for (v, pos) in cands {
                            if best.as_ref().is_none_or(|b| v < b.0) {
                                best = Some((v, ti, m.sid.clone(), pos));
                            }
                        }
                    }
                }
                let Some((_, ti, sid, pos)) = best else { break };
                let t = &mut tiers[ti];
                let old = t.manifests[&sid].clone();
                match pos {
                    None => {
                        remove_manifest_in(&t.dir, &mut t.manifests, &sid, &self.kill, &self.usage)
                    }
                    Some(p) => {
                        let mut m2 = t.manifests[&sid].clone();
                        m2.snaps.retain(|x| x.pos != p);
                        if write_manifest_in(&t.dir, &t.id_hash, &m2, &self.kill, &self.usage)
                            .is_err()
                        {
                            break;
                        }
                        t.manifests.insert(sid.clone(), m2);
                    }
                }
                eviction[ti].removed(t, &old, &self.kill, &self.usage, &self.inflight, stop);
                used = self.used_bytes();
            }
            self.manifests = std::mem::take(&mut tiers[0].manifests);
        }
        let freed = before.saturating_sub(self.used_bytes());
        self.stats.evicted_bytes += freed;
        freed
    }
}

fn block_path_in(dir: &Path, h: &str, c: &str) -> PathBuf {
    dir.join("blocks").join(&h[..2]).join(format!("{h}.{c}.kv"))
}

fn snap_path_in(dir: &Path, key: &str, c: &str) -> PathBuf {
    dir.join("snaps").join(format!("{key}.{c}.st"))
}

fn write_manifest_in(
    dir: &Path,
    id_hash: &str,
    m: &Manifest,
    k: &Kill,
    usage: &DiskUsage,
) -> Result<(), String> {
    atomic_write(
        k,
        usage,
        &dir.join("sessions").join(format!("{}.json", m.sid)),
        m.to_json(id_hash).to_string().as_bytes(),
    )
}

/// Drop session `sid` of the store at `dir`: its manifest, and its echo file (`echo/<hint hash>.jsonl`,
/// one scheme with echo.rs and the controller's state_cache.py) with the last manifest naming its hint.
fn remove_manifest_in(
    dir: &Path,
    manifests: &mut HashMap<String, Manifest>,
    sid: &str,
    k: &Kill,
    usage: &DiskUsage,
) {
    let gone = manifests.remove(sid);
    // a removed manifest or echo file must not come back after a crash (a cleared session would be
    // resurrected): the directory is fsync'd after the unlink. Blocks and snapshots are not: one
    // that comes back is named by no manifest and the next sweep removes it.
    if rm(k, usage, &dir.join("sessions").join(format!("{sid}.json"))) {
        fsync_dir(&dir.join("sessions"));
    }
    if let Some(h) = gone.and_then(|m| m.hint) {
        if hex_name(&h, 32)
            && !manifests
                .values()
                .any(|m| m.hint.as_deref() == Some(h.as_str()))
            && rm(k, usage, &dir.join("echo").join(format!("{h}.jsonl")))
        {
            fsync_dir(&dir.join("echo"));
        }
    }
}

/// Delete `.pending` files and blocks / snapshots no manifest of `manifests` references, in the store
/// at `dir`; files in `inflight` (an off-lock write on its way) are kept. Returns the bytes freed.
fn sweep_dir(
    dir: &Path,
    manifests: &HashMap<String, Manifest>,
    kill: &Kill,
    usage: &DiskUsage,
    inflight: &std::sync::Mutex<HashSet<PathBuf>>,
    stop: &dyn Fn() -> bool,
) -> u64 {
    let mut want: HashSet<PathBuf> = HashSet::new();
    for m in manifests.values() {
        for (h, c) in &m.blocks {
            want.insert(block_path_in(dir, h, c));
        }
        for s in &m.snaps {
            want.insert(snap_path_in(dir, &s.key, &s.c));
        }
    }
    let busy = inflight.lock().unwrap_or_else(|p| p.into_inner()).clone();
    let files: Vec<(PathBuf, u64)> = usage
        .lock()
        .sizes
        .iter()
        .filter(|(p, _)| p.starts_with(dir))
        .map(|(p, size)| (p.clone(), *size))
        .collect();
    let mut freed = 0;
    for (p, size) in files {
        if stop() {
            break;
        }
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let parent = p.parent();
        let data = parent == Some(dir.join("snaps").as_path())
            || parent.and_then(Path::parent) == Some(dir.join("blocks").as_path());
        // Match the original sweep's scope: echo writes and unknown nested directories are
        // not ours to collect (their pending files need not be registered in `inflight`).
        let pending = data || parent == Some(dir) || parent == Some(dir.join("sessions").as_path());
        let ours = (pending && name.ends_with(".pending"))
            || (data && (name.ends_with(".kv") || name.ends_with(".st")) && !want.contains(&p));
        if ours
            && !busy.contains(&p)
            && fs::symlink_metadata(&p).is_ok_and(|m| m.is_file())
            && rm(kill, usage, &p)
        {
            freed += size;
        }
    }
    freed
}

/// The recency weight of a session in the eviction order: 1 / (1 + hours since its last use).
fn recency(m: &Manifest, now: u64) -> f64 {
    1.0 / (1.0 + now.saturating_sub(m.atime) as f64 / 3600.0)
}

/// One identity's store under the root, for the global budget (v0.1.3).
struct Tier {
    dir: PathBuf,
    id_hash: String,
    manifests: HashMap<String, Manifest>,
}

/// Per-enforcement reference counts and exclusive byte values. Dropping a reference touches
/// only that item's paths and, when a shared path becomes exclusive, its remaining owner.
struct Eviction {
    refs: HashMap<PathBuf, HashSet<String>>,
    own: HashMap<String, u64>,
}

fn manifest_files(dir: &Path, m: &Manifest) -> HashSet<PathBuf> {
    m.blocks
        .iter()
        .map(|(h, c)| block_path_in(dir, h, c))
        .chain(m.snaps.iter().map(|s| snap_path_in(dir, &s.key, &s.c)))
        .collect()
}

impl Eviction {
    fn new(t: &Tier, usage: &DiskUsage) -> Self {
        let mut refs: HashMap<PathBuf, HashSet<String>> = HashMap::new();
        for m in t.manifests.values() {
            for p in manifest_files(&t.dir, m) {
                refs.entry(p).or_default().insert(m.sid.clone());
            }
        }
        let files = usage.lock();
        // Preserve the value policy: all of a session's snapshots plus its exclusive blocks.
        let mut own: HashMap<String, u64> = t
            .manifests
            .values()
            .map(|m| (m.sid.clone(), m.snaps.iter().map(|s| s.bytes).sum()))
            .collect();
        for (p, owners) in &refs {
            if owners.len() == 1 && p.extension().is_some_and(|e| e == "kv") {
                *own.entry(owners.iter().next().unwrap().clone())
                    .or_insert(0) += files.sizes.get(p).copied().unwrap_or(0);
            }
        }
        Self { refs, own }
    }

    fn removed(
        &mut self,
        tier: &Tier,
        old: &Manifest,
        kill: &Kill,
        usage: &DiskUsage,
        inflight: &std::sync::Mutex<HashSet<PathBuf>>,
        stop: &dyn Fn() -> bool,
    ) {
        let new = tier.manifests.get(&old.sid);
        let kept = new
            .map(|m| manifest_files(&tier.dir, m))
            .unwrap_or_default();
        let busy = inflight.lock().unwrap_or_else(|p| p.into_inner());
        for p in manifest_files(&tier.dir, old).difference(&kept) {
            if stop() {
                break;
            }
            let size = usage.lock().sizes.get(p).copied().unwrap_or(0);
            let owners = self.refs.get_mut(p).unwrap();
            let block = p.extension().is_some_and(|e| e == "kv");
            let lost = if block {
                if owners.len() == 1 {
                    size
                } else {
                    0
                }
            } else {
                old.snaps
                    .iter()
                    .filter(|s| snap_path_in(&tier.dir, &s.key, &s.c) == *p)
                    .map(|s| s.bytes)
                    .sum()
            };
            let own = self.own.entry(old.sid.clone()).or_default();
            *own = own.saturating_sub(lost);
            owners.remove(&old.sid);
            match owners.len() {
                0 => {
                    if !busy.contains(p) && fs::symlink_metadata(p).is_ok_and(|m| m.is_file()) {
                        rm(kill, usage, p);
                    }
                }
                1 if block => {
                    *self
                        .own
                        .entry(owners.iter().next().unwrap().clone())
                        .or_default() += size;
                }
                _ => {}
            }
        }
        if new.is_none() {
            self.own.remove(&old.sid);
        }
    }
}

/// The other identities' stores under `root` (identity.json + their readable manifests; nothing is
/// deleted while reading: a foreign file this engine cannot parse is not its to judge).
fn foreign_tiers(root: &Path, own: &Path, stop: &dyn Fn() -> bool) -> Vec<Tier> {
    let mut v = Vec::new();
    let Ok(rd) = read_dir(root) else { return v };
    for de in rd.flatten() {
        if stop() {
            break;
        }
        let p = de.path();
        if p == own || !de.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }
        let idh = fs::read(p.join("identity.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Json>(&b).ok())
            .and_then(|j| {
                j.get("identity_hash")
                    .and_then(Json::as_str)
                    .map(str::to_string)
            });
        let Some(idh) = idh.filter(|h| hex_name(h, 64)) else {
            continue;
        };
        let mut manifests = HashMap::new();
        if let Ok(rd2) = read_dir(p.join("sessions")) {
            for d in rd2.flatten() {
                if stop() {
                    break;
                }
                let mp = d.path();
                if mp.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let small = fs::symlink_metadata(&mp)
                    .is_ok_and(|m| m.is_file() && m.len() <= MAX_MANIFEST_BYTES);
                let m = small
                    .then(|| fs::read(&mp).ok())
                    .flatten()
                    .and_then(|b| serde_json::from_slice::<Json>(&b).ok())
                    .and_then(|j| Manifest::from_json(&j, &idh));
                if let Some(m) =
                    m.filter(|m| mp.file_stem().and_then(|s| s.to_str()) == Some(m.sid.as_str()))
                {
                    manifests.insert(m.sid.clone(), m);
                }
            }
        }
        v.push(Tier {
            dir: p,
            id_hash: idh,
            manifests,
        });
    }
    v
}

/// The snapshot payload at `pos` (the bytes after the file header): u64 KV bytes, the KV rows of the
/// partial block from the live state, then the checkpoint's non-KV state (`snap`, or the live one).
fn snap_payload<E: Engine>(
    eng: &mut E,
    pos: usize,
    snap: Option<&E::Snapshot>,
    w: &mut dyn Write,
) -> Result<(), String> {
    let a = pos / BLOCK * BLOCK;
    let rb = eng.kv_row_bytes();
    w.write_all(&(((pos - a) * rb) as u64).to_le_bytes())
        .map_err(io)?;
    if pos > a {
        eng.kv_rows_save(a, pos, w).map_err(|e| e.0)?;
    }
    match snap {
        Some(s) => eng.snap_save(s, w).map_err(|e| e.0),
        None => eng.snap_save_live(w).map_err(|e| e.0),
    }
}

/// Writer that fails as soon as `cancel` rises (a request arrived) or `deadline` has passed (N2): the
/// write in progress is given up.
struct Cancellable<'a, 'b> {
    w: &'a mut dyn Write,
    cancel: Option<&'b std::sync::atomic::AtomicBool>,
    deadline: Option<std::time::Instant>,
}

impl Write for Cancellable<'_, '_> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        if self
            .cancel
            .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        {
            return Err(std::io::Error::other("cancelled by a request"));
        }
        if self
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            return Err(std::io::Error::other("past the shutdown deadline"));
        }
        self.w.write(b)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

/// v0.1.3 (item 6): a state write prepared under the worker lock ([`StateStore::stage_block`],
/// [`StateStore::stage_snap`]) and put on disk without it ([`Staged::write`]); the worker commits the
/// manifest afterwards (`Worker::commit_staged`). Holds no engine or store reference.
pub struct Staged {
    usage: DiskUsage,
    pub dir: PathBuf,
    pub id_hash: String,
    /// (chain hash, block index, checksum hex, payload; None = that file is on disk already)
    pub blocks: Vec<(String, usize, String, Option<Vec<u8>>)>,
    pub head_key: String,
    pub head_c: String,
    pub head_payload: Vec<u8>,
    /// the manifest to commit (blocks, the consistent snapshots of its lineage, this head)
    pub manifest: Manifest,
    /// ancestors the manifest absorbs (dropped at commit)
    pub absorbed: Vec<String>,
    /// the state written: its length and the cache epoch it was taken at
    pub n: usize,
    pub epoch: u64,
    pub delay_ms: u64,
    inflight: std::sync::Arc<std::sync::Mutex<HashSet<PathBuf>>>,
    paths: Vec<PathBuf>,
    /// this write's own `.pending` names (two off-lock writes of the same block never share one)
    tag: String,
}

static STAGE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Staged {
    /// Register the files of this write in the store's `inflight` set (sweeps keep them until
    /// [`Staged::release`]).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        st: &StateStore,
        blocks: Vec<(String, usize, String, Option<Vec<u8>>)>,
        head_key: String,
        head_c: String,
        head_payload: Vec<u8>,
        manifest: Manifest,
        absorbed: Vec<String>,
        n: usize,
        epoch: u64,
    ) -> Staged {
        let tag = format!(
            "s{}",
            STAGE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let mut paths = Vec::new();
        for (h, _, c, data) in &blocks {
            if data.is_some() {
                let p = block_path_in(&st.dir, h, c);
                paths.push(p.with_file_name(format!("{h}.{tag}.kv.pending")));
                paths.push(p);
            }
        }
        paths.push(snap_path_in(&st.dir, &head_key, &head_c));
        paths.push(
            st.dir
                .join("snaps")
                .join(format!("{head_key}.{tag}.st.pending")),
        );
        st.inflight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .extend(paths.iter().cloned());
        Staged {
            usage: st.usage.clone(),
            dir: st.dir.clone(),
            id_hash: st.ident.hash.clone(),
            blocks,
            head_key,
            head_c,
            head_payload,
            manifest,
            absorbed,
            n,
            epoch,
            delay_ms: st.stage_delay_ms,
            inflight: st.inflight.clone(),
            paths,
            tag,
        }
    }

    /// Write the staged blocks and the head snapshot (atomic: `.pending`, fsync, rename, directory
    /// fsync). No worker lock is needed; returns the bytes written.
    pub fn write(&self) -> Result<u64, String> {
        if self.delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
        }
        let k = Kill::default();
        let mut bytes = 0;
        for (h, i, c, data) in &self.blocks {
            let Some(data) = data else { continue };
            let sub = self.dir.join("blocks").join(&h[..2]);
            if !sub.is_dir() {
                mkdir(&sub)?;
                fsync_dir(&self.dir.join("blocks"));
            }
            let a = (i * BLOCK) as u64;
            let name = format!("{h}.{c}.kv");
            let (_, len, sum) = atomic_stream(
                &k,
                &self.usage,
                &sub,
                &format!("{h}.{}.kv", self.tag),
                HDR,
                &|len, sum| header(MAGIC_BLOCK, &self.id_hash, a, len, sum),
                &|_| name.clone(),
                &mut |w| w.write_all(data).map_err(io),
            )?;
            if format!("{sum:016x}") != *c {
                return Err("staged block changed on its way to disk".into());
            }
            bytes += HDR + len;
        }
        let pos = self.n as u64;
        let name = format!("{}.{}.st", self.head_key, self.head_c);
        let (_, len, sum) = atomic_stream(
            &k,
            &self.usage,
            &self.dir.join("snaps"),
            &format!("{}.{}.st", self.head_key, self.tag),
            HDR,
            &|len, sum| header(MAGIC_SNAP, &self.id_hash, pos, len, sum),
            &|_| name.clone(),
            &mut |w| w.write_all(&self.head_payload).map_err(io),
        )?;
        if format!("{sum:016x}") != self.head_c {
            return Err("staged snapshot changed on its way to disk".into());
        }
        Ok(bytes + HDR + len)
    }

    /// The files are named by a manifest (or given up): sweeps may take what is unnamed again.
    pub fn release(&self) {
        let mut s = self.inflight.lock().unwrap_or_else(|p| p.into_inner());
        for p in &self.paths {
            s.remove(p);
        }
    }
}

/// What identifies one stored session's conversation in the other tiers (RAM, parked, the legacy
/// prefix / session stores) for a session clear: its blocks, its head, and how many leading blocks
/// it shares with other sessions (those prefixes are theirs too and are kept).
#[derive(Clone, Debug)]
pub struct SessionMatch {
    pub id_hash: String,
    pub blocks: Vec<String>,
    pub head: (usize, String),
    pub shared: usize,
    pub hint: Option<String>,
}

impl SessionMatch {
    /// Do these tokens hold this session's conversation: they continue its head, or they are a
    /// prefix of it that reaches past every block shared with another session (a prefix inside the
    /// shared part, the static system prompt, is kept for the others).
    pub fn owns(&self, tokens: &[u32]) -> bool {
        let hs = chain(&self.id_hash, tokens);
        let (hp, hk) = (&self.head.0, &self.head.1);
        let k = hp / BLOCK;
        if tokens.len() >= *hp
            && hs.len() >= k
            && hs[..k] == self.blocks[..k.min(self.blocks.len())]
            && snap_key(&self.id_hash, &hs, tokens, *hp) == *hk
        {
            return true;
        }
        let kt = hs.len();
        kt > self.shared && kt <= self.blocks.len() && hs[..] == self.blocks[..kt]
    }
}

#[cfg(test)]
thread_local! {
    static DIRECTORY_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn read_dir(p: impl AsRef<Path>) -> std::io::Result<fs::ReadDir> {
    #[cfg(test)]
    DIRECTORY_READS.with(|n| n.set(n.get() + 1));
    fs::read_dir(p)
}

/// Bytes of regular files under `p` (symlinks not followed).
pub fn dir_bytes(p: &Path) -> u64 {
    let Ok(rd) = read_dir(p) else { return 0 };
    rd.flatten()
        .map(|d| match d.file_type() {
            Ok(t) if t.is_dir() => dir_bytes(&d.path()),
            Ok(t) if t.is_file() => d.metadata().map(|m| m.len()).unwrap_or(0),
            _ => 0,
        })
        .sum()
}

/// Which snapshot positions of a session to keep (design sec. 2): the anchor and the head always,
/// every position in the last 16K tokens, one per ~8K down to 64K back, then one per ~32K; at
/// most [`MAX_SNAPS`]. See [`thin_max`].
pub fn thin(pos: Vec<(usize, bool)>, n: usize) -> Vec<usize> {
    thin_max(pos, n, MAX_SNAPS)
}

/// [`thin`] with a cap of `max` (`--state-snaps`). `pos` = (position, must keep). Past the cap
/// (v0.1.3) the optional point whose removal leaves the smallest gap goes (ties: the deeper-back
/// one), never the two newest optional points: dense recent points thin before the sparse far
/// ones, so the worst re-prefill after an old edit stays near 32K tokens wherever `max` points can
/// cover the session (the old rule dropped the deepest points first and left the far past bare).
/// Returns the kept positions, deepest first.
pub fn thin_max(mut pos: Vec<(usize, bool)>, n: usize, max: usize) -> Vec<usize> {
    pos.sort_by(|a, b| b.0.cmp(&a.0));
    pos.dedup_by(|a, b| {
        if a.0 == b.0 {
            b.1 |= a.1;
            true
        } else {
            false
        }
    });
    let must: Vec<usize> = pos.iter().filter(|p| p.1).map(|p| p.0).collect();
    let mut out: Vec<usize> = must.clone();
    let mut last_kept = usize::MAX;
    for &(p, m) in &pos {
        if m {
            continue;
        }
        let back = n.saturating_sub(p);
        let step = if back <= 16 << 10 {
            0
        } else if back <= 64 << 10 {
            8 << 10
        } else {
            32 << 10
        };
        if last_kept == usize::MAX || last_kept.saturating_sub(p) >= step {
            out.push(p);
            last_kept = p;
        }
    }
    out.sort_unstable_by(|a, b| b.cmp(a));
    out.dedup();
    let newest: Vec<usize> = out
        .iter()
        .copied()
        .filter(|p| !must.contains(p))
        .take(2)
        .collect();
    while out.len() > max {
        let mut best: Option<(usize, usize)> = None;
        for i in 0..out.len() {
            if must.contains(&out[i]) || newest.contains(&out[i]) {
                continue;
            }
            let above = if i == 0 { n.max(out[0]) } else { out[i - 1] };
            let below = if i + 1 < out.len() { out[i + 1] } else { 0 };
            let gap = above - below;
            if best.is_none_or(|(g, _)| gap <= g) {
                best = Some((gap, i));
            }
        }
        let Some((_, i)) = best else { break };
        out.remove(i);
    }
    out
}

/// Session id of a lineage: 32 hex chars of SHA-256(identity, the tokens up to the end of the
/// first block past the anchor, or all of them while the session is shorter). Stable once the
/// session has a full block past its first user message (a fork after that keeps the id, an edit
/// before it starts a new one); while shorter it moves as it grows, and the worker drops the
/// earlier manifest of the same lineage.
pub fn session_id(id_hash: &str, tokens: &[u32], anchor_pos: usize) -> String {
    let end = ((anchor_pos / BLOCK + 2) * BLOCK).min(tokens.len());
    let mut s = Sha256::default();
    s.update(id_hash.as_bytes());
    s.update(&tok_bytes(&tokens[..end]));
    s.hex()[..32].to_string()
}

/// Error type adapter for the worker.
pub fn eng_err(e: EngineError) -> String {
    e.0
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    use crate::tests_state::{fill, tmp};

    #[test]
    fn enforce_does_not_walk_directories_under_or_over_budget() {
        let root = tmp("budget-no-walk");
        let mut w = fill(&root, 1 << 40, &[1100, 1200, 1300]);
        let st = w.states.as_mut().unwrap();
        st.global_budget = false;
        DIRECTORY_READS.with(|n| n.set(0));
        st.enforce(None);
        assert_eq!(DIRECTORY_READS.with(|n| n.get()), 0, "under budget");

        for m in st.manifests.values_mut() {
            m.atime = now_s() - PIN_SECS - 1;
        }
        st.budget = st.used_bytes() / 2;
        DIRECTORY_READS.with(|n| n.set(0));
        assert!(st.enforce(None) > 0);
        assert!(st.used_bytes() <= st.budget);
        assert_eq!(
            DIRECTORY_READS.with(|n| n.get()),
            0,
            "multi-session eviction"
        );
        assert_eq!(st.used_bytes(), dir_bytes(&root));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unhinted_chat_echo_is_accounted_without_management_resync() {
        use crate::tests_state::{call, files, sys};
        use crate::tests_state_p2b::server_echo;
        use std::sync::atomic::Ordering;

        let root = tmp("budget-unhinted-echo");
        let srv = server_echo(&root, vec![], 1 << 40);
        let (code, result) = call(
            &srv,
            "POST",
            "/v1/chat/completions",
            json!({"messages": [
                {"role": "system", "content": sys()},
                {"role": "user", "content": "Synthetic question"}
            ], "max_tokens": 3, "temperature": 0}),
        )
        .await;
        assert_eq!(code, 200, "{result}");
        let start = std::time::Instant::now();
        while srv.persist_done.load(Ordering::SeqCst) == 0 {
            assert!(start.elapsed().as_secs() < 30, "persist did not finish");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        let w = srv.worker.lock().unwrap();
        let st = w.states.as_ref().unwrap();
        assert!(!st.manifests.is_empty());
        assert!(st.manifests.values().all(|m| m.hint.is_none()));
        assert!(!files(&st.dir.join("echo"), ".jsonl").is_empty());
        assert_eq!(st.used_bytes(), dir_bytes(&root));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn echo_accounting_tracks_deferred_appends_compaction_and_removal() {
        use crate::echo::TurnEnd;
        use crate::tests_state_p2b::server_echo;

        let root = tmp("budget-echo-lifecycle");
        let srv = server_echo(&root, vec![], 1 << 40);
        let check = || {
            let w = srv.worker.lock().unwrap();
            let st = w.states.as_ref().unwrap();
            assert!(st.manifests.is_empty(), "echo accounting needs no manifest");
            assert_eq!(st.used_bytes(), dir_bytes(&root));
            st.used_bytes()
        };
        let baseline = check();
        for hint in [None, Some("synthetic")] {
            srv.echo.lock().unwrap().cfg.compact_bytes = 0;
            for _ in 0..5 {
                let (_, pending) = srv.echo.lock().unwrap().record_deferred(
                    &[1, 2, 3],
                    &[2],
                    99,
                    &chat::ChatMsg {
                        content: "synthetic".into(),
                        ..Default::default()
                    },
                    &[4, 99],
                    TurnEnd::Eog,
                    hint,
                );
                let pending = pending.unwrap();
                check(); // a deferred append has not changed disk usage yet
                assert!(pending.write());
                assert!(check() > baseline);
            }
            let before = check();
            let mut echo = srv.echo.lock().unwrap();
            let group = hint
                .map(|h| format!("s:{h}"))
                .unwrap_or_else(|| format!("k:{}", crate::echo::prefix_key(echo.identity(), &[1])));
            echo.cfg.compact_bytes = 1;
            assert!(echo.maybe_compact(&group));
            assert!(check() < before, "compaction subtracts replaced bytes");
            let h = hint_hash(hint.unwrap_or(&group));
            assert_eq!(echo.drop_hint_hash(&h), 1);
            assert_eq!(check(), baseline);
        }
        // Invalidation refuses an outstanding append, without inventing accounted bytes.
        let (_, pending) = srv.echo.lock().unwrap().record_deferred(
            &[1, 2, 3],
            &[2],
            99,
            &Default::default(),
            &[4, 99],
            TurnEnd::Eog,
            None,
        );
        srv.echo.lock().unwrap().clear();
        assert!(!pending.unwrap().write());
        assert_eq!(check(), baseline);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cancelled_foreign_eviction_reclaims_orphans_before_evicting_live_sessions() {
        use crate::tests_state::{ident, open};

        let root = tmp("budget-cancel-foreign");
        let mut foreign = fill(&root, 1 << 40, &[600, 3000]);
        let other = foreign.states.as_mut().unwrap();
        for mut m in other.manifests.values().cloned().collect::<Vec<_>>() {
            m.atime = now_s() - PIN_SECS - 1;
            m.snaps.retain(|s| s.class == "head");
            other.put_manifest(m).unwrap();
        }
        other.sweep();
        let victim = other.manifests.values().min_by_key(|m| m.n_tokens).unwrap();
        let survivor = other.manifests.values().max_by_key(|m| m.n_tokens).unwrap();
        let kept = manifest_files(&other.dir, survivor);
        let orphans: HashSet<_> = manifest_files(&other.dir, victim)
            .difference(&kept)
            .cloned()
            .collect();
        assert!(!orphans.is_empty());
        let removed_manifest = other.manifest_path(&victim.sid);
        let kept_manifest = other.manifest_path(&survivor.sid);
        let reclaim: u64 = orphans.iter().map(|p| fs::metadata(p).unwrap().len()).sum();
        let mut st = open(&root, ident("other-synthetic", ""), 1 << 40);
        st.global_budget = true;
        st.budget = st.used_bytes() - reclaim - fs::metadata(&removed_manifest).unwrap().len();

        assert!(st.enforce_until(None, &|| !removed_manifest.exists()) > 0);
        assert!(!removed_manifest.exists());
        assert!(
            orphans.iter().all(|p| p.is_file()),
            "cancel before file cleanup"
        );
        assert!(st.used_bytes() > st.budget);
        assert_eq!(st.used_bytes(), dir_bytes(&root));
        st.enforce(None);
        assert!(
            orphans.iter().all(|p| !p.exists()),
            "resume foreign cleanup"
        );
        assert!(kept_manifest.is_file(), "do not evict another live session");
        assert!(kept.iter().all(|p| p.is_file()), "keep shared files");
        assert!(st.used_bytes() <= st.budget);
        assert_eq!(st.used_bytes(), dir_bytes(&root));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn accounting_tracks_rewrites_shared_files_and_failed_writes() {
        use crate::engine::StubEngine;
        use crate::tests_state::{ident, open};

        for kill_at in [None, Some(0), Some(1), Some(2)] {
            let root = tmp(&format!("budget-writes-{kill_at:?}"));
            let mut st = open(&root, ident("synthetic", ""), 1 << 40);
            let mut eng = StubEngine::new(4096, 300, 0, vec![]);
            eng.prefill(&vec![1; BLOCK + 1]).unwrap();
            let h = "a".repeat(64);
            let key = "b".repeat(64);
            let c = st.write_block(&mut eng, &h, 0, &mut Vec::new()).unwrap();
            let before = st.used_bytes();
            st.write_block(&mut eng, &h, 0, &mut Vec::new()).unwrap();
            assert_eq!(st.used_bytes(), before, "shared block counted once");
            let (snap_c, _) = st.write_snap(&mut eng, &key, BLOCK + 1, None).unwrap();
            let before = st.used_bytes();
            st.write_snap(&mut eng, &key, BLOCK + 1, None).unwrap();
            assert_eq!(st.used_bytes(), before, "snapshot overwrite counted once");
            assert!(st.has_block(&h, &c) && st.has_snap(&key, &snap_c));
            assert_eq!(st.used_bytes(), dir_bytes(&root));

            st.kill.at = kill_at.map(|n| st.kill.taken() + n);
            let result = st.write_snap(&mut eng, &"c".repeat(64), BLOCK + 1, None);
            assert_eq!(result.is_err(), kill_at.is_some());
            assert_eq!(
                st.used_bytes(),
                dir_bytes(&root),
                "including a torn pending or renamed file"
            );
            st.kill.at = None;
            st.sweep();
            assert_eq!(st.used_bytes(), dir_bytes(&root));
            assert!(!st.has_block(&h, &c), "unreferenced block swept");
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn staged_writes_share_accounting_before_commit_and_after_abandonment() {
        use crate::generate::GenParams;
        use crate::tests_state_p2b::{gen, toks, worker};

        let root = tmp("budget-staged");
        let mut w = worker(&root);
        w.persist_every = 0;
        gen(
            &mut w,
            &toks(1300, 4),
            &GenParams {
                n_predict: 3,
                ..Default::default()
            },
        );
        let staged = w.stage_persist(Some("synthetic")).unwrap();
        staged.write().unwrap();
        let st = w.states.as_mut().unwrap();
        assert_eq!(st.used_bytes(), dir_bytes(&root));
        let before = st.used_bytes();
        staged.write().unwrap();
        assert_eq!(st.used_bytes(), before, "same staged paths counted once");
        st.sweep();
        assert_eq!(st.used_bytes(), before, "inflight files kept");
        staged.release();
        assert!(st.sweep() > 0, "abandoned staged files reclaimed");
        assert_eq!(st.used_bytes(), dir_bytes(&root));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn clearing_a_foreign_model_resyncs_the_returned_usage() {
        use crate::tests_state::{call, ident, open, server};
        let root = tmp("budget-clear-foreign");
        let foreign_id = ident("foreign-synthetic", "");
        let foreign = open(&root, foreign_id.clone(), 1 << 40);
        let srv = server(&root, vec![], ident("synthetic", ""), 1 << 40);
        let (code, result) = call(
            &srv,
            "POST",
            "/state-cache/clear",
            json!({"scope": "model", "model": foreign_id.model_key()}),
        )
        .await;
        assert_eq!(code, 200, "{result}");
        assert!(!foreign.dir.exists());
        assert!(result["freed_bytes"].as_u64().unwrap() > 0);
        assert_eq!(result["used_bytes"], dir_bytes(&root));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn indexed_sweeps_keep_the_original_directory_scope() {
        use crate::tests_state::{ident, open};
        let root = tmp("budget-sweep-scope");
        let mut st = open(&root, ident("synthetic", ""), 1 << 40);
        let echo_pending = st.dir.join("echo/record.pending");
        let unknown = st.dir.join("snaps/nested");
        mkdir(&unknown).unwrap();
        let unknown = unknown.join("unknown.st");
        atomic_write(&st.kill, &st.usage, &echo_pending, b"in progress").unwrap();
        atomic_write(&st.kill, &st.usage, &unknown, b"not ours").unwrap();
        assert_eq!(st.sweep(), 0);
        assert!(echo_pending.is_file() && unknown.is_file());
        assert_eq!(st.used_bytes(), dir_bytes(&root));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn management_resyncs_external_changes_without_following_symlinks() {
        use crate::tests_state::{ident, open};
        let root = tmp("budget-resync");
        let st = open(&root, ident("synthetic", ""), 1 << 40);
        let before = st.used_bytes();
        let outside = tmp("budget-outside");
        fs::write(outside.join("data"), b"not cache data").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
        fs::create_dir(root.join("foreign")).unwrap();
        fs::write(root.join("foreign/data"), [0; 123]).unwrap();
        assert_eq!(st.used_bytes(), before, "normal reads use the counter");
        assert_eq!(st.usage()["used_bytes"], before + 123);
        assert_eq!(st.used_bytes(), dir_bytes(&root));
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn cancellation_between_evictions_keeps_remaining_manifests_and_can_resume() {
        let root = tmp("budget-cancel");
        let mut w = fill(&root, 1 << 40, &[1100, 1200, 1300]);
        let st = w.states.as_mut().unwrap();
        st.global_budget = false;
        let mut manifests: Vec<Manifest> = st.manifests.values().cloned().collect();
        for m in &mut manifests {
            m.atime = now_s() - PIN_SECS - 1;
            m.snaps.retain(|s| s.class == "head");
            st.put_manifest(m.clone()).unwrap();
        }
        st.sweep();
        let paths: Vec<PathBuf> = manifests.iter().map(|m| st.manifest_path(&m.sid)).collect();
        st.budget = 0;
        let cancelled = std::sync::atomic::AtomicBool::new(true);
        assert_eq!(st.enforce_cancellable(None, Some(&cancelled)), 0);
        assert_eq!(st.manifests.len(), 3);
        // Deterministic arrival of a request just after the first session eviction.
        assert!(st.enforce_until(None, &|| paths.iter().any(|p| !p.exists())) > 0);
        assert_eq!(st.manifests.len(), 2);
        assert_eq!(st.used_bytes(), dir_bytes(&root));
        for m in st.manifests.values() {
            assert!(m.blocks.iter().all(|(h, c)| st.has_block(h, c)));
            assert!(m.snaps.iter().all(|s| st.has_snap(&s.key, &s.c)));
        }
        st.enforce(None);
        assert!(st.manifests.is_empty());
        assert_eq!(st.used_bytes(), dir_bytes(&root));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn progress_defers_maintenance_until_the_cached_budget_is_exceeded() {
        use crate::engine::StubEngine;
        use crate::generate::PersistMode;
        let root = tmp("budget-progress");
        let mut w = fill(&root, 1 << 40, &[1100]);
        let key = "d".repeat(64);
        let st = w.states.as_mut().unwrap();
        let mut other = StubEngine::new(4096, 300, 0, vec![]);
        let (c, _) = st.write_snap(&mut other, &key, 0, None).unwrap();
        let more = vec![5; BLOCK];
        w.engine.prefill(&more).unwrap();
        w.cache.appended(&more);
        DIRECTORY_READS.with(|n| n.set(0));
        assert!(w.persist(None, PersistMode::Progress, None).head.is_some());
        let st = w.states.as_mut().unwrap();
        assert!(st.has_snap(&key, &c), "under budget: no maintenance pass");
        assert_eq!(DIRECTORY_READS.with(|n| n.get()), 0);
        assert_eq!(st.used_bytes(), dir_bytes(&root));
        st.budget = st.used_bytes() - 1;
        w.engine.prefill(&more).unwrap();
        w.cache.appended(&more);
        assert!(w.persist(None, PersistMode::Progress, None).head.is_some());
        let st = w.states.as_ref().unwrap();
        assert!(!st.has_snap(&key, &c), "over budget: orphan reclaimed");
        assert_eq!(st.used_bytes(), dir_bytes(&root));
        fs::remove_dir_all(root).unwrap();
    }
}
