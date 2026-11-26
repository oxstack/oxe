//! `oxe models` and `oxe import-oxa`: the user model catalog in `models.toml`.
//!
//! The catalog is the only place `oxe serve` looks for models. Adds and imports never load
//! weights: `oxe admission` reads the GGUF header and prints the launch arguments and
//! allocation bounds, which are stored verbatim.
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const CONFIG_VERSION: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Budget {
    pub kv_bytes_per_position: u64,
    pub scratch_bytes: u64,
    pub mtp_state_bytes: u64,
    pub margin_bytes: u64,
}

impl Budget {
    pub fn to_json(self) -> Value {
        json!({
            "kv_bytes_per_position": self.kv_bytes_per_position,
            "scratch_bytes": self.scratch_bytes,
            "mtp_state_bytes": self.mtp_state_bytes,
            "margin_bytes": self.margin_bytes,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    pub model: PathBuf,
    pub arguments: Vec<String>,
    pub context_window: u32,
    pub max_output: u32,
    pub allocation_budget: Budget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub path: PathBuf,
    pub binary: Option<PathBuf>,
    pub listen: String,
    pub allow_remote: bool,
    pub idle_unload_seconds: u64,
    pub startup_timeout_seconds: u64,
    pub max_waiters: usize,
    pub environment: BTreeMap<String, String>,
    pub models: BTreeMap<String, ModelConfig>,
}

impl Config {
    pub fn empty(path: PathBuf) -> Config {
        Config {
            path,
            binary: None,
            listen: "127.0.0.1:8090".into(),
            allow_remote: false,
            idle_unload_seconds: 600,
            startup_timeout_seconds: 180,
            max_waiters: 8,
            environment: BTreeMap::new(),
            models: BTreeMap::new(),
        }
    }

    pub fn load(path: &Path) -> Result<Config, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Config::parse(path, &text)
    }

    pub fn parse(path: &Path, text: &str) -> Result<Config, String> {
        let doc: toml_edit::DocumentMut = text
            .parse()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if let Some(version) = doc.get("version").and_then(|v| v.as_integer()) {
            if version != CONFIG_VERSION as i64 {
                return Err(format!(
                    "{}: config version {version} is not supported (expected {CONFIG_VERSION})",
                    path.display()
                ));
            }
        }
        let mut cfg = Config::empty(path.to_path_buf());
        cfg.binary = doc
            .get("binary")
            .and_then(|v| v.as_str())
            .map(PathBuf::from);
        cfg.listen = doc
            .get("listen")
            .and_then(|v| v.as_str())
            .unwrap_or(&cfg.listen)
            .to_string();
        cfg.allow_remote = doc
            .get("allow_remote")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        cfg.idle_unload_seconds = positive(&doc, "idle_unload_seconds", cfg.idle_unload_seconds)?;
        cfg.startup_timeout_seconds =
            positive(&doc, "startup_timeout_seconds", cfg.startup_timeout_seconds)?;
        // Zero is meaningful: one active request and no queued waiters, so an extra request
        // gets the documented 503 immediately.
        cfg.max_waiters = match doc.get("max_waiters") {
            None => cfg.max_waiters,
            Some(value) => usize::try_from(
                value
                    .as_integer()
                    .filter(|v| *v >= 0)
                    .ok_or("max_waiters must be a non-negative integer")?,
            )
            .map_err(|_| "max_waiters is out of range".to_string())?,
        };
        if let Some(env) = doc.get("environment") {
            let table = env
                .as_table_like()
                .ok_or_else(|| format!("{}: environment must be a table", path.display()))?;
            for (key, value) in table.iter() {
                let value = value.as_str().ok_or_else(|| {
                    format!("{}: environment.{key} must be a string", path.display())
                })?;
                cfg.environment.insert(key.to_string(), value.to_string());
            }
        }
        let Some(models) = doc.get("models") else {
            return Ok(cfg);
        };
        let models = models
            .as_table_like()
            .ok_or_else(|| format!("{}: models must be a table", path.display()))?;
        for (id, entry) in models.iter() {
            if id.is_empty() || id.len() > 128 || id.chars().any(char::is_control) {
                return Err(format!("{}: invalid model id {id:?}", path.display()));
            }
            let table = entry
                .as_table_like()
                .ok_or_else(|| format!("{}: models.{id} must be a table", path.display()))?;
            let model = table
                .get("model")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("{}: models.{id}.model is required", path.display()))?;
            let arguments = match table.get("arguments") {
                None => Vec::new(),
                Some(value) => value
                    .as_array()
                    .ok_or_else(|| {
                        format!("{}: models.{id}.arguments must be an array", path.display())
                    })?
                    .iter()
                    .map(|v| {
                        v.as_str().map(str::to_owned).ok_or_else(|| {
                            format!("{}: models.{id}.arguments must be strings", path.display())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            };
            let context_window = u32_arg(table.get("context_window"), id, "context_window", path)?;
            let max_output = u32_arg(table.get("max_output"), id, "max_output", path)?;
            if max_output > context_window {
                return Err(format!(
                    "{}: models.{id}.max_output ({max_output}) exceeds context_window ({context_window})",
                    path.display()
                ));
            }
            let budget_value = table.get("allocation_budget").ok_or_else(|| {
                format!(
                    "{}: models.{id}.allocation_budget is required",
                    path.display()
                )
            })?;
            let allocation_budget = budget_from_toml(budget_value, id, path)?;
            cfg.models.insert(
                id.to_string(),
                ModelConfig {
                    model: PathBuf::from(model),
                    arguments,
                    context_window,
                    max_output,
                    allocation_budget,
                },
            );
        }
        Ok(cfg)
    }

    /// Canonical TOML for this catalog. Comments and unknown keys are not preserved.
    pub fn to_toml(&self) -> String {
        let mut doc = toml_edit::DocumentMut::new();
        doc["version"] = toml_edit::value(CONFIG_VERSION as i64);
        if let Some(binary) = &self.binary {
            doc["binary"] = toml_edit::value(binary.display().to_string());
        }
        doc["listen"] = toml_edit::value(self.listen.as_str());
        doc["allow_remote"] = toml_edit::value(self.allow_remote);
        doc["idle_unload_seconds"] = toml_edit::value(self.idle_unload_seconds as i64);
        doc["startup_timeout_seconds"] = toml_edit::value(self.startup_timeout_seconds as i64);
        doc["max_waiters"] = toml_edit::value(self.max_waiters as i64);
        if !self.environment.is_empty() {
            let mut env = toml_edit::Table::new();
            for (key, value) in &self.environment {
                env[key] = toml_edit::value(value.as_str());
            }
            doc["environment"] = toml_edit::Item::Table(env);
        }
        let mut models = toml_edit::Table::new();
        for (id, model) in &self.models {
            let mut table = toml_edit::Table::new();
            table["model"] = toml_edit::value(model.model.display().to_string());
            let mut arguments = toml_edit::Array::new();
            for argument in &model.arguments {
                arguments.push(argument.as_str());
            }
            table["arguments"] = toml_edit::value(arguments);
            table["context_window"] = toml_edit::value(i64::from(model.context_window));
            table["max_output"] = toml_edit::value(i64::from(model.max_output));
            let mut budget = toml_edit::Table::new();
            budget["kv_bytes_per_position"] =
                toml_edit::value(model.allocation_budget.kv_bytes_per_position as i64);
            budget["scratch_bytes"] =
                toml_edit::value(model.allocation_budget.scratch_bytes as i64);
            budget["mtp_state_bytes"] =
                toml_edit::value(model.allocation_budget.mtp_state_bytes as i64);
            budget["margin_bytes"] = toml_edit::value(model.allocation_budget.margin_bytes as i64);
            table["allocation_budget"] = toml_edit::Item::Table(budget);
            models[id] = toml_edit::Item::Table(table);
        }
        doc["models"] = toml_edit::Item::Table(models);
        doc.to_string()
    }

    pub fn save(&self) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        std::fs::write(&self.path, self.to_toml())
            .map_err(|e| format!("{}: {e}", self.path.display()))
    }
}

fn positive(doc: &toml_edit::DocumentMut, key: &str, default: u64) -> Result<u64, String> {
    match doc.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_integer()
            .filter(|v| *v > 0)
            .map(|v| v as u64)
            .ok_or_else(|| format!("{key} must be a positive integer")),
    }
}

fn u32_arg(
    value: Option<&toml_edit::Item>,
    id: &str,
    key: &str,
    path: &Path,
) -> Result<u32, String> {
    value
        .and_then(|v| v.as_integer())
        .and_then(|v| u32::try_from(v).ok())
        .filter(|v| *v > 0)
        .ok_or_else(|| {
            format!(
                "{}: models.{id}.{key} must be a positive integer",
                path.display()
            )
        })
}

fn budget_from_toml(value: &toml_edit::Item, id: &str, path: &Path) -> Result<Budget, String> {
    let table = value.as_table_like().ok_or_else(|| {
        format!(
            "{}: models.{id}.allocation_budget must be a table",
            path.display()
        )
    })?;
    let bytes = |key: &str| -> Result<u64, String> {
        table
            .get(key)
            .and_then(|v| v.as_integer())
            .and_then(|v| u64::try_from(v).ok())
            .ok_or_else(|| {
                format!(
                    "{}: models.{id}.allocation_budget.{key} must be a non-negative integer",
                    path.display()
                )
            })
    };
    let budget = Budget {
        kv_bytes_per_position: bytes("kv_bytes_per_position")?,
        scratch_bytes: bytes("scratch_bytes")?,
        mtp_state_bytes: bytes("mtp_state_bytes")?,
        margin_bytes: bytes("margin_bytes")?,
    };
    if budget.margin_bytes == 0 {
        return Err(format!(
            "{}: models.{id}.allocation_budget.margin_bytes must be positive",
            path.display()
        ));
    }
    Ok(budget)
}

pub fn default_config_path() -> PathBuf {
    config_home().join("oxe/models.toml")
}

pub fn config_home() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .unwrap_or_else(|| PathBuf::from(".config"))
}

fn load_or_empty(path: &Path) -> Result<Config, String> {
    if path.exists() {
        Config::load(path)
    } else {
        Ok(Config::empty(path.to_path_buf()))
    }
}

fn absolute(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .map_err(|e| format!("cannot resolve {}: {e}", path.display()))
}

pub fn run_models(args: &[String]) -> ! {
    match args.split_first() {
        Some((verb, rest)) if verb == "list" => {
            let path = match rest {
                [] => default_config_path(),
                [flag, path] if flag == "--config" => PathBuf::from(path),
                _ => usage("usage: oxe models list [--config PATH]"),
            };
            list(path)
        }
        Some((verb, rest)) if verb == "add" => add(rest),
        Some((verb, rest)) if verb == "rm" => rm(rest),
        _ => usage("usage: oxe models list | add <gguf> [--id ID] [--ctx N] | rm ID"),
    }
}

fn rest_config(flag: Option<&Path>) -> PathBuf {
    flag.map(Path::to_path_buf)
        .unwrap_or_else(default_config_path)
}

fn list(path: PathBuf) -> ! {
    match load_or_empty(&path) {
        Ok(cfg) if cfg.models.is_empty() => {
            println!("no models configured in {}", path.display());
            std::process::exit(0)
        }
        Ok(cfg) => {
            for (id, model) in &cfg.models {
                println!(
                    "{id}\t{}\tctx={}\tmax_output={}\t{}",
                    model.model.display(),
                    model.context_window,
                    model.max_output,
                    "allocation budget set"
                );
            }
            std::process::exit(0)
        }
        Err(error) => fail(&error),
    }
}

fn add(args: &[String]) -> ! {
    let mut gguf = None;
    let mut id = None;
    let mut ctx = None;
    let mut config = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--id" => {
                i += 1;
                id = Some(value(args, i, "--id").to_string());
            }
            "--ctx" => {
                i += 1;
                ctx = Some(
                    value(args, i, "--ctx")
                        .parse::<u32>()
                        .unwrap_or_else(|_| usage("--ctx takes a positive integer")),
                );
            }
            "--config" => {
                i += 1;
                config = Some(PathBuf::from(value(args, i, "--config")));
            }
            flag if flag.starts_with('-') => usage(&format!("unknown flag {flag}")),
            path if gguf.is_none() => gguf = Some(PathBuf::from(path)),
            extra => usage(&format!("unexpected argument {extra}")),
        }
        i += 1;
    }
    let Some(gguf) = gguf else {
        usage("models add needs a GGUF path");
    };
    let gguf = absolute(&gguf).unwrap_or_else(|e| fail(&e));
    let id = id.unwrap_or_else(|| {
        gguf.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".into())
    });
    let path = rest_config(config.as_deref());
    let mut cfg = load_or_empty(&path).unwrap_or_else(|e| fail(&e));
    let report = admission_report(&gguf, ctx).unwrap_or_else(|e| fail(&e));
    let model = model_from_admission(&gguf, &report).unwrap_or_else(|e| fail(&e));
    if cfg.models.contains_key(&id) {
        eprintln!("oxe models: replacing existing model {id}");
    }
    println!(
        "oxe models: {id}: ctx={} arguments={} budget={}",
        model.context_window,
        model.arguments.len(),
        serde_json::to_string(&model.allocation_budget.to_json()).unwrap_or_default()
    );
    cfg.models.insert(id, model);
    cfg.save().unwrap_or_else(|e| fail(&e));
    println!("oxe models: wrote {}", path.display());
    std::process::exit(0)
}

fn rm(args: &[String]) -> ! {
    let mut id = None;
    let mut config = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                config = Some(PathBuf::from(value(args, i, "--config")));
            }
            flag if flag.starts_with('-') => usage(&format!("unknown flag {flag}")),
            model if id.is_none() => id = Some(model.to_string()),
            extra => usage(&format!("unexpected argument {extra}")),
        }
        i += 1;
    }
    let Some(id) = id else {
        usage("models rm needs a model id");
    };
    let path = rest_config(config.as_deref());
    let mut cfg = Config::load(&path).unwrap_or_else(|e| fail(&e));
    if cfg.models.remove(&id).is_none() {
        fail(&format!(
            "model {id} is not configured in {}",
            path.display()
        ));
    }
    cfg.save().unwrap_or_else(|e| fail(&e));
    println!("oxe models: removed {id} from {}", path.display());
    std::process::exit(0)
}

/// Run the engine's own allocation report (`oxe admission --json`). It reads the GGUF header
/// only: no weights are mapped and no inference runs.
pub fn admission_report(gguf: &Path, ctx: Option<u32>) -> Result<Value, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current executable: {e}"))?;
    let mut command = Command::new(exe);
    command.arg("admission").arg("--model").arg(gguf);
    if let Some(ctx) = ctx {
        command.args(["--ctx", &ctx.to_string()]);
    }
    command.arg("--json");
    let output = command
        .output()
        .map_err(|e| format!("run oxe admission: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "oxe admission failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("oxe admission printed invalid JSON: {e}"))
}

pub fn model_from_admission(gguf: &Path, report: &Value) -> Result<ModelConfig, String> {
    let arguments = report
        .get("arguments")
        .and_then(Value::as_array)
        .ok_or("the admission report has no arguments array")?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "the admission arguments must be strings".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let limits = report
        .get("limits")
        .ok_or("the admission report has no limits")?;
    let context_window = u32_value(limits.get("context_window"), "context_window")?;
    let max_output = u32_value(limits.get("max_output"), "max_output")?;
    if max_output > context_window {
        return Err(format!(
            "the admission report has max_output {max_output} above context_window {context_window}; narrow it before adding"
        ));
    }
    let budget = limits
        .get("allocation_budget")
        .ok_or("the admission report has no allocation_budget")?;
    let bytes = |key: &str| {
        budget
            .get(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("the admission report has no allocation_budget.{key}"))
    };
    Ok(ModelConfig {
        model: absolute(gguf)?,
        arguments,
        context_window,
        max_output,
        allocation_budget: Budget {
            kv_bytes_per_position: bytes("kv_bytes_per_position")?,
            scratch_bytes: bytes("scratch_bytes")?,
            mtp_state_bytes: bytes("mtp_state_bytes")?,
            margin_bytes: bytes("margin_bytes")?,
        },
    })
}

fn u32_value(value: Option<&Value>, key: &str) -> Result<u32, String> {
    value
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .filter(|v| *v > 0)
        .ok_or_else(|| format!("the admission report has no positive {key}"))
}

pub fn run_import_oxa(args: &[String]) -> ! {
    let mut from = None;
    let mut dry_run = false;
    let mut force = false;
    let mut config = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--from" => {
                i += 1;
                from = Some(PathBuf::from(value(args, i, "--from")));
            }
            "--config" => {
                i += 1;
                config = Some(PathBuf::from(value(args, i, "--config")));
            }
            "--dry-run" => dry_run = true,
            "--force" => force = true,
            "-h" | "--help" => {
                usage("usage: oxe import-oxa [--from PATH] [--config PATH] [--dry-run] [--force]")
            }
            flag => usage(&format!("unknown argument {flag}")),
        }
        i += 1;
    }
    let from = from.unwrap_or_else(|| {
        std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(".oxa/agent/settings.json"))
            .unwrap_or_else(|| PathBuf::from(".oxa/agent/settings.json"))
    });
    let text = std::fs::read_to_string(&from)
        .unwrap_or_else(|e| fail(&format!("{}: {e}", from.display())));
    let value: Value =
        serde_json::from_str(&text).unwrap_or_else(|e| fail(&format!("{}: {e}", from.display())));
    let imported = import_value(&value).unwrap_or_else(|e| fail(&e));
    let path = rest_config(config.as_deref());
    let mut cfg = load_or_empty(&path).unwrap_or_else(|e| fail(&e));
    let collisions: Vec<_> = imported
        .models
        .keys()
        .filter(|id| cfg.models.contains_key(*id))
        .cloned()
        .collect();
    if !collisions.is_empty() && !force {
        fail(&format!(
            "{} already configures {}; pass --force to overwrite",
            path.display(),
            collisions.join(", ")
        ));
    }
    cfg.binary = imported.binary.or(cfg.binary);
    cfg.idle_unload_seconds = imported.idle_unload_seconds;
    cfg.startup_timeout_seconds = imported.startup_timeout_seconds;
    cfg.environment = imported.environment;
    for (id, model) in imported.models {
        cfg.models.insert(id, model);
    }
    if dry_run {
        print!("{}", cfg.to_toml());
        eprintln!(
            "oxe import-oxa: dry run; {} was not modified",
            path.display()
        );
        std::process::exit(0)
    }
    cfg.save().unwrap_or_else(|e| fail(&e));
    println!(
        "oxe import-oxa: imported {} model(s) from {} into {}",
        cfg.models.len(),
        from.display(),
        path.display()
    );
    println!("oxe import-oxa: {} was not modified", from.display());
    std::process::exit(0)
}

/// Convert an oxa settings document. Only `oxa.local` is read.
fn import_value(value: &Value) -> Result<Config, String> {
    let root = value.get("oxa").unwrap_or(value);
    let local = root
        .get("local")
        .ok_or("settings have no oxa.local section")?;
    let models = local
        .get("models")
        .and_then(Value::as_object)
        .filter(|m| !m.is_empty())
        .ok_or("oxa.local.models is missing or empty")?;
    let limits = local
        .get("limits")
        .and_then(Value::as_object)
        .ok_or("oxa.local.limits is missing")?;
    let mut cfg = Config::empty(default_config_path());
    cfg.binary = local
        .get("binary")
        .and_then(Value::as_str)
        .map(PathBuf::from);
    cfg.idle_unload_seconds = local
        .get("idle_unload_seconds")
        .and_then(Value::as_u64)
        .unwrap_or(cfg.idle_unload_seconds);
    cfg.startup_timeout_seconds = local
        .get("startup_timeout_seconds")
        .and_then(Value::as_u64)
        .unwrap_or(cfg.startup_timeout_seconds);
    if let Some(env) = local.get("environment") {
        let env = env
            .as_object()
            .ok_or("oxa.local.environment must be an object")?;
        for (key, value) in env {
            let value = value
                .as_str()
                .ok_or_else(|| format!("oxa.local.environment.{key} must be a string"))?;
            cfg.environment.insert(key.clone(), value.to_string());
        }
    }
    for (id, entry) in models {
        let model = entry
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("oxa.local.models.{id}.model is missing"))?;
        let arguments = match entry.get("arguments") {
            None => Vec::new(),
            Some(value) => value
                .as_array()
                .ok_or_else(|| format!("oxa.local.models.{id}.arguments must be an array"))?
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| format!("oxa.local.models.{id}.arguments must be strings"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        let limit = limits
            .get(id)
            .ok_or_else(|| format!("oxa.local.limits.{id} is missing"))?;
        let positive = |key: &str| -> Result<u32, String> {
            limit
                .get(key)
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
                .filter(|v| *v > 0)
                .ok_or_else(|| format!("oxa.local.limits.{id}.{key} must be a positive integer"))
        };
        let budget = limit.get("allocation_budget").ok_or_else(|| {
            format!("oxa.local.limits.{id}.allocation_budget is missing; add the four allocation components")
        })?;
        let bytes = |key: &str| -> Result<u64, String> {
            budget.get(key).and_then(Value::as_u64).ok_or_else(|| {
                format!(
                    "oxa.local.limits.{id}.allocation_budget.{key} must be a non-negative integer"
                )
            })
        };
        let context_window = positive("context_window")?;
        let max_output = positive("max_output")?;
        if max_output > context_window {
            return Err(format!(
                "oxa.local.limits.{id}.max_output ({max_output}) exceeds context_window ({context_window})"
            ));
        }
        cfg.models.insert(
            id.clone(),
            ModelConfig {
                model: PathBuf::from(model),
                arguments,
                context_window,
                max_output,
                allocation_budget: Budget {
                    kv_bytes_per_position: bytes("kv_bytes_per_position")?,
                    scratch_bytes: bytes("scratch_bytes")?,
                    mtp_state_bytes: bytes("mtp_state_bytes")?,
                    margin_bytes: bytes("margin_bytes")?,
                },
            },
        );
    }
    Ok(cfg)
}

fn value<'a>(args: &'a [String], i: usize, flag: &str) -> &'a str {
    args.get(i)
        .map(String::as_str)
        .unwrap_or_else(|| usage(&format!("missing value for {flag}")))
}

fn usage(message: &str) -> ! {
    eprintln!("{message}\n  oxe models list | add <gguf> [--id ID] [--ctx N] | rm ID\n  oxe import-oxa [--from PATH] [--dry-run] [--force]");
    std::process::exit(2)
}

fn fail(message: &str) -> ! {
    eprintln!("oxe: {message}");
    std::process::exit(1)
}
