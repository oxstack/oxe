//! User-only local factory. The caller has already selected the default-off setting.
use crate::launch_profile::{from_user_settings, InstalledLayout, InstalledModel};
use crate::lifecycle::Policy;
use crate::native::{LocalEngine, LocalModel};
use crate::Config;
use oxa_engine::StreamOpts;
use oxa_proto::{codes, Engine, OxaError};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

pub const LOCAL_SETTING: &str = "oxa.engine.inProcessLocal";

/// Only pass the user settings' oxa.local subtree and private agent directory.
/// Merged project settings must never select a binary, arguments or environment.
/// This function performs no file, credential, model, process or network I/O.
pub fn configured_engine(
    local: Option<&Value>,
    private_dir: PathBuf,
    opts: StreamOpts,
) -> Result<Arc<dyn Engine>, OxaError> {
    let local = local.and_then(Value::as_object).ok_or_else(invalid)?;
    if local.keys().any(|k| {
        !matches!(
            k.as_str(),
            "binary"
                | "models"
                | "limits"
                | "environment"
                | "idle_unload_seconds"
                | "startup_timeout_seconds"
        )
    }) {
        return Err(invalid());
    }
    let binary = PathBuf::from(local.get("binary").and_then(Value::as_str).ok_or_else(invalid)?);
    let entries = local
        .get("models")
        .and_then(Value::as_object)
        .filter(|v| !v.is_empty())
        .ok_or_else(invalid)?;
    let limits = local
        .get("limits")
        .and_then(Value::as_object)
        .ok_or_else(invalid)?;
    if limits.len() != entries.len() || limits.keys().any(|k| !entries.contains_key(k)) {
        return Err(invalid());
    }
    let environment = match local.get("environment") {
        None => BTreeMap::new(),
        Some(value) => value
            .as_object()
            .ok_or_else(invalid)?
            .iter()
            .map(|(key, value)| Ok((key.clone(), value.as_str().ok_or_else(invalid)?.to_owned())))
            .collect::<Result<_, OxaError>>()?,
    };
    let mut installed = BTreeMap::new();
    for (alias, entry) in entries {
        let model = PathBuf::from(entry.get("model").and_then(Value::as_str).ok_or_else(invalid)?);
        let arguments = match entry.get("arguments") {
            None => Vec::new(),
            Some(value) => value
                .as_array()
                .ok_or_else(invalid)?
                .iter()
                .map(|v| v.as_str().map(str::to_owned).ok_or_else(invalid))
                .collect::<Result<_, _>>()?,
        };
        installed.insert(alias.clone(), InstalledModel { model, arguments });
    }
    let layout = Arc::new(InstalledLayout {
        binary,
        models: installed,
        environment,
    });
    let user = Arc::new(serde_json::json!({"oxa":{"local":local}}));
    let startup = duration(local.get("startup_timeout_seconds"), Duration::from_secs(180))?;
    let policy = Policy {
        idle_timeout: match local.get("idle_unload_seconds") {
            Some(Value::Null) => None,
            value => Some(duration(value, Duration::from_secs(7200))?),
        },
    };
    let mut models = Vec::new();
    for alias in entries.keys() {
        let value = &limits[alias];
        let fields = value.as_object().ok_or_else(invalid)?;
        if fields.keys().any(|k| {
            !matches!(
                k.as_str(),
                "context_window" | "max_output" | "required_available_bytes" | "thinking" | "images"
            )
        }) {
            return Err(invalid());
        }
        let profile = from_user_settings(&user, &layout, alias, 1).map_err(|_| invalid())?;
        if profile.arguments.iter().any(|word| {
            word.to_str().is_some_and(|word| {
                matches!(
                    word.split('=').next().unwrap_or(""),
                    "--api-key" | "--api-key-file"
                )
            })
        }) || profile
            .environment
            .keys()
            .any(|key| key.starts_with("OXE_API_KEY"))
        {
            return Err(invalid());
        }
        let alias_owned = alias.clone();
        let layout = layout.clone();
        let user = user.clone();
        models.push(LocalModel {
            alias: alias.clone(),
            context_window: limit(value, "context_window")?,
            max_output: limit(value, "max_output")?,
            required_available_bytes: value["required_available_bytes"]
                .as_u64()
                .filter(|v| *v > 0)
                .ok_or_else(invalid)?,
            thinking: optional_bool(value, "thinking")?,
            images: optional_bool(value, "images")?,
            launch: Arc::new(move |port| {
                let profile =
                    from_user_settings(&user, &layout, &alias_owned, port).map_err(|_| invalid())?;
                Ok(Config {
                    executable: profile.binary,
                    arguments: profile.arguments,
                    environment: profile
                        .environment
                        .into_iter()
                        .map(|(k, v)| (k.into(), v.into()))
                        .collect(),
                    startup_timeout: startup,
                    health_interval: Duration::from_millis(250),
                    unhealthy_limit: 2,
                    term_grace: Duration::from_secs(3),
                    restart_backoff: Duration::from_secs(1),
                    max_restarts: 1,
                })
            }),
        });
    }
    Ok(Arc::new(LocalEngine::new(models, private_dir, opts, policy)?))
}
fn invalid() -> OxaError {
    OxaError::new(codes::INVALID, "Invalid user local configuration in oxa.local.")
}
fn limit(value: &Value, key: &str) -> Result<u32, OxaError> {
    value[key]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .filter(|v| *v > 0)
        .ok_or_else(invalid)
}
fn optional_bool(value: &Value, key: &str) -> Result<Option<bool>, OxaError> {
    value
        .get(key)
        .map(|v| v.as_bool().ok_or_else(invalid))
        .transpose()
}
fn duration(value: Option<&Value>, default: Duration) -> Result<Duration, OxaError> {
    match value {
        None => Ok(default),
        Some(value) => {
            let v = value.as_f64().filter(|v| *v > 0.0).ok_or_else(invalid)?;
            Duration::try_from_secs_f64(v).map_err(|_| invalid())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxa_fake::TestEnv;
    fn settings() -> Value {
        serde_json::json!({"binary":"/synthetic/oxe-server", "models":{"synthetic":{"model":"/synthetic/model.gguf","arguments":["--engine","stub"]}},"limits":{"synthetic":{"context_window":4096,"max_output":64,"required_available_bytes":1048576}}})
    }
    #[test]
    fn metadata_factory_never_opens_launch_paths_or_private_storage() {
        let env = TestEnv::new();
        let private = env.work().join("not-created");
        let engine = configured_engine(Some(&settings()), private.clone(), StreamOpts::default()).unwrap();
        let info = engine.info(Some("synthetic")).unwrap();
        assert_eq!(info.n_ctx, 4096);
        assert_eq!(info.max_output, 64);
        assert!(!info.caps.remote);
        assert!(!private.exists());
    }
    #[test]
    fn malformed_limits_owned_auth_and_interpreter_launches_refuse() {
        let env = TestEnv::new();
        for kind in 0..5 {
            let mut settings = settings();
            match kind {
                0 => settings["limits"]["synthetic"]["context_window"] = serde_json::json!(0),
                1 => settings["limits"]["synthetic"]["max_output"] = serde_json::json!(8192),
                2 => {
                    settings["models"]["synthetic"]["arguments"] =
                        serde_json::json!(["--api-key-file", "/synthetic/forbidden"])
                }
                3 => settings["binary"] = serde_json::json!("/usr/bin/python3"),
                _ => settings["environment"] = serde_json::json!({"OXE_API_KEY_FILE":"/synthetic/forbidden"}),
            }
            assert!(
                configured_engine(Some(&settings), env.work().join("private"), StreamOpts::default())
                    .is_err()
            );
        }
        assert!(!env.work().join("private").exists());
    }
}
