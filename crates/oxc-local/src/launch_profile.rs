//! Pure launch description for the local engine (Milestone A).
//!
//! The caller supplies the *user* settings document and its installed layout. Never pass
//! merged project settings here: project files must not select an executable or environment.
//! This module reads no files/environment, probes no binaries and starts no processes.
//! Executable identity, memory policy, authentication and supervision remain caller-owned.
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt;
use std::path::{Component, Path, PathBuf};

/// An installed model's explicit paths and literal base arguments.
#[derive(Clone)]
pub struct InstalledModel {
    pub model: PathBuf,
    pub arguments: Vec<String>,
}

/// Resolved by the installer/caller, not guessed from the current directory or PATH.
#[derive(Clone)]
pub struct InstalledLayout {
    pub binary: PathBuf,
    pub models: BTreeMap<String, InstalledModel>,
    /// Explicit base environment. No process environment is implicitly inherited here.
    pub environment: BTreeMap<String, String>,
}

/// Pass these literal words to the native supervisor, with its environment cleared first.
#[derive(Clone, PartialEq, Eq)]
pub struct LaunchProfile {
    pub binary: PathBuf,
    pub model: PathBuf,
    /// Does not include the executable. Binding/model/alias words precede extra arguments.
    pub arguments: Vec<OsString>,
    pub environment: BTreeMap<String, String>,
}
impl fmt::Debug for LaunchProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LaunchProfile")
            .field("binary", &"[configured]")
            .field("model", &"[configured]")
            .field("argument_count", &self.arguments.len())
            .field("environment_count", &self.environment.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Settings,
    UnknownModel,
    Binary,
    Model,
    Alias,
    Port,
    Arguments,
    OwnedArgument,
    Environment,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Settings => "invalid local launch settings",
            Self::UnknownModel => "local model is absent from the installed layout",
            Self::Binary => "local engine needs an absolute native binary path",
            Self::Model => "local model needs an absolute model path",
            Self::Alias => "invalid local model alias",
            Self::Port => "local engine port must be nonzero",
            Self::Arguments => "invalid local engine arguments",
            Self::OwnedArgument => "local engine arguments override a supervisor-owned option",
            Self::Environment => "invalid local engine environment",
        })
    }
}
impl std::error::Error for Error {}

/// Resolve user-only per-model paths, literal arguments, environment and MTP controls.
///
/// Missing values use the supplied layout. Arguments replace, rather than append to, the
/// installed argument list; environment entries override the explicit base. As in the old
/// native launch path, inherited GGML_/LLAMA_ tuning is removed before explicit overrides.
/// The two shipped MoE aliases get the owner-selected sampled-MTP/cutoff profile.
/// Other engine tuning, cache directories and interpreter/shell trampolines are not inferred.
pub fn from_user_settings(
    user: &Value,
    layout: &InstalledLayout,
    alias: &str,
    port: u16,
) -> Result<LaunchProfile, Error> {
    if alias.is_empty() || alias.chars().any(char::is_control) {
        return Err(Error::Alias);
    }
    if port == 0 {
        return Err(Error::Port);
    }
    let installed = layout.models.get(alias).ok_or(Error::UnknownModel)?;
    let root = user.as_object().ok_or(Error::Settings)?;
    let oxa = object(root.get("oxa"))?;
    let local = object(oxa.and_then(|v| v.get("local")))?;
    let models = object(local.and_then(|v| v.get("models")))?;
    let selected = object(models.and_then(|v| v.get(alias)))?;
    let empty = Map::new();
    let selected = selected.unwrap_or(&empty);
    if selected.keys().any(|k| {
        !matches!(
            k.as_str(),
            "binary" | "model" | "arguments" | "environment" | "mtp_max_context" | "sampled_mtp"
        )
    }) {
        return Err(Error::Settings);
    }
    let binary = path(selected.get("binary"), &layout.binary, Error::Binary)?;
    if binary.extension().is_some_and(|v| v == "py")
        || binary
            .file_name()
            .is_some_and(|v| matches!(v.to_str(), Some("python" | "python3" | "sh" | "bash")))
    {
        return Err(Error::Binary);
    }
    let model = path(selected.get("model"), &installed.model, Error::Model)?;
    let extra = match selected.get("arguments") {
        None => installed.arguments.clone(),
        Some(Value::Array(values)) => values
            .iter()
            .map(|v| v.as_str().map(str::to_owned).ok_or(Error::Arguments))
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(Error::Arguments),
    };
    // Preserve the published per-model crossover; the owner enables sampled MTP
    // for these two shipped MoE aliases in the Rust-only release (#763).
    let default_cutoff = match alias {
        "ornith-35b-a3b-local" => Some(32768),
        "qwen3.6-35b-a3b-local" => Some(16384),
        _ => None,
    };
    let cutoff = match selected.get("mtp_max_context") {
        None => default_cutoff,
        Some(value) => Some(
            value
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .filter(|v| *v > 0)
                .ok_or(Error::Settings)?,
        ),
    };
    let sampled = match selected.get("sampled_mtp") {
        None => default_cutoff.is_some(),
        Some(value) => value.as_bool().ok_or(Error::Settings)?,
    };
    // The native parser accepts these controls only as separate argv words.
    if extra
        .iter()
        .any(|word| word.starts_with("--mtp-max-ctx=") || word.starts_with("--mtp-sampled="))
    {
        return Err(Error::Arguments);
    }
    let has_flag = |flag: &str| extra.iter().any(|word| word.split('=').next() == Some(flag));
    let raw_cutoff = has_flag("--mtp-max-ctx");
    let raw_sampled = has_flag("--mtp-sampled");
    if raw_cutoff && selected.contains_key("mtp_max_context")
        || raw_sampled && selected.contains_key("sampled_mtp")
    {
        return Err(Error::Arguments);
    }
    let mut arguments: Vec<OsString> = vec![
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string().into(),
        "--model".into(),
        model.as_os_str().into(),
        "--alias".into(),
        alias.into(),
    ];
    for word in extra {
        if word.chars().any(char::is_control) {
            return Err(Error::Arguments);
        }
        let flag = word.split('=').next().unwrap_or("");
        if matches!(
            flag,
            "--host" | "--port" | "--listen" | "--bind" | "--model" | "--alias" | "-a" | "-m"
        ) || (word.starts_with("-m") && !word.starts_with("--"))
        {
            return Err(Error::OwnedArgument);
        }
        arguments.push(word.into());
    }
    if let Some(cutoff) = cutoff.filter(|_| !raw_cutoff) {
        arguments.extend(["--mtp-max-ctx".into(), cutoff.to_string().into()]);
    }
    if sampled && !raw_sampled {
        arguments.push("--mtp-sampled".into());
    }
    let mut environment: BTreeMap<String, String> = layout
        .environment
        .iter()
        .filter(|(key, _)| !key.starts_with("GGML_") && !key.starts_with("LLAMA_"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if let Some(value) = selected.get("environment") {
        for (key, value) in value.as_object().ok_or(Error::Environment)? {
            environment.insert(key.clone(), value.as_str().ok_or(Error::Environment)?.to_owned());
        }
    }
    if environment.iter().any(|(key, value)| {
        key.is_empty() || key.contains('=') || key.chars().any(char::is_control) || value.contains('\0')
    }) {
        return Err(Error::Environment);
    }
    Ok(LaunchProfile {
        binary,
        model,
        arguments,
        environment,
    })
}

fn object(value: Option<&Value>) -> Result<Option<&Map<String, Value>>, Error> {
    value.map(|v| v.as_object().ok_or(Error::Settings)).transpose()
}
fn path(value: Option<&Value>, fallback: &Path, error: Error) -> Result<PathBuf, Error> {
    let path = match value {
        None => fallback.to_path_buf(),
        Some(Value::String(value)) => PathBuf::from(value),
        _ => return Err(error),
    };
    if !path.is_absolute()
        || path.file_name().is_none()
        || path.components().any(|c| matches!(c, Component::ParentDir))
        || path.as_os_str().as_encoded_bytes().contains(&0)
    {
        return Err(error);
    }
    Ok(path)
}
