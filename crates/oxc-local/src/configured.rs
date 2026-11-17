//! User-only local factory. The caller has already selected the default-off setting.
use crate::launch_profile::{from_user_settings, InstalledLayout, InstalledModel};
use crate::lifecycle::{Model, Policy};
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
/// Discovery does not inspect binaries/weights, credentials or sockets or start a process.
/// With no binary override, only the OS-reported oxa executable path is queried.
/// The embedding executable must call [`crate::platform::child_entry`] before its
/// own argument parsing so a scope-required child can wait safely before engine exec.
pub fn configured_engine(
    local: Option<&Value>,
    private_dir: PathBuf,
    opts: StreamOpts,
) -> Result<Arc<dyn Engine>, OxaError> {
    configured_local_engine(local, private_dir, opts).map(|engine| engine as Arc<dyn Engine>)
}

/// Build the same user-only native engine while retaining its guest entry point.
pub fn configured_local_engine(
    local: Option<&Value>,
    private_dir: PathBuf,
    opts: StreamOpts,
) -> Result<Arc<LocalEngine>, OxaError> {
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
    let binary = match local.get("binary") {
        Some(value) => PathBuf::from(value.as_str().ok_or_else(invalid)?),
        None => sibling_binary(&std::env::current_exe().map_err(|_| invalid())?)?,
    };
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
        if fields.contains_key("required_available_bytes") {
            return Err(crate::admission::invalid());
        }
        if fields.keys().any(|k| {
            !matches!(
                k.as_str(),
                "context_window" | "max_output" | "allocation_budget" | "thinking" | "images"
            )
        }) {
            return Err(invalid());
        }
        let context = limit(value, "context_window")?;
        let budget = value["allocation_budget"].clone();
        crate::admission::required(&budget, context, 0)?;
        let mut profile = from_user_settings(&user, &layout, alias, 1).map_err(|_| invalid())?;
        crate::admission::bind_context(&mut profile.arguments, context)?;
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
            context_window: context,
            max_output: limit(value, "max_output")?,
            thinking: optional_bool(value, "thinking")?,
            images: optional_bool(value, "images")?,
            launch: Arc::new(move |port| {
                let mut profile =
                    from_user_settings(&user, &layout, &alias_owned, port).map_err(|_| invalid())?;
                crate::admission::bind_context(&mut profile.arguments, context)?;
                // Check only at cold load/switch, before creating child authentication.
                // Discovery never inspects weights or runs the engine.
                let required = required_for_weights(&profile.model, &budget, context)?;
                use std::os::unix::fs::PermissionsExt;
                if !std::fs::metadata(&profile.binary)
                    .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
                {
                    return Err(OxaError::new(codes::NOT_FOUND,
                        "Local engine binary is missing or not executable; install oxe-server beside oxa or set oxa.local.binary."));
                }
                if profile.arguments.windows(2).any(|pair| pair[0] == "--engine" && pair[1] == "flash")
                    || profile.arguments.iter().any(|word| word == "--engine=flash")
                {
                    return Err(OxaError::unsupported("Flash streaming-budget admission in the resident local factory"));
                }
                Ok(Model { name: alias_owned.clone(), required_available_bytes: required, config: Config {
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
                }})
            }),
        });
    }
    Ok(Arc::new(LocalEngine::new(models, private_dir, opts, policy)?))
}
fn required_for_weights(model: &std::path::Path, budget: &Value, context: u32) -> Result<u64, OxaError> {
    let weights = std::fs::metadata(model).ok().filter(|m| m.is_file()).ok_or_else(||
        OxaError::new(codes::NOT_FOUND, "Local weights are missing; set oxa.local.models.<alias>.model to an existing GGUF path. No download is attempted."))?;
    crate::admission::required(budget, context, weights.len())
}
fn sibling_binary(executable: &std::path::Path) -> Result<PathBuf, OxaError> {
    executable
        .parent()
        .filter(|p| p.is_absolute())
        .map(|p| p.join("oxe-server"))
        .ok_or_else(invalid)
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
        serde_json::json!({"binary":"/synthetic/oxe-server", "models":{"synthetic":{"model":"/synthetic/model.gguf","arguments":["--engine","stub"]}},"limits":{"synthetic":{"context_window":4096,"max_output":64,"allocation_budget":{"kv_bytes_per_position":1,"scratch_bytes":0,"mtp_state_bytes":0,"margin_bytes":1044480}}}})
    }
    #[test]
    fn opaque_total_without_allocation_components_refuses_clearly() {
        let env = TestEnv::new();
        let mut old = settings();
        old["limits"]["synthetic"]
            .as_object_mut()
            .unwrap()
            .remove("allocation_budget");
        old["limits"]["synthetic"]["required_available_bytes"] = serde_json::json!(1048576);
        let error = match configured_engine(Some(&old), env.work().join("private"), StreamOpts::default()) {
            Ok(_) => panic!("opaque total accepted without allocation components"),
            Err(error) => error,
        };
        assert!(error.message.contains("allocation_budget"));
    }
    #[test]
    fn cold_weights_requirement_uses_current_full_file_length_without_reading_contents() {
        let env = TestEnv::new();
        let path = env.work().join("synthetic-sparse.gguf");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(22_700_000_000).unwrap();
        let budget = serde_json::json!({"kv_bytes_per_position":22528,"scratch_bytes":1048576,"mtp_state_bytes":1048576,"margin_bytes":1048576});
        let first = required_for_weights(&path, &budget, 16384).unwrap();
        assert_eq!(first, 22_700_000_000 + 369_098_752 + 3 * 1048576);
        file.set_len(22_800_000_000).unwrap();
        assert_eq!(
            required_for_weights(&path, &budget, 16384).unwrap(),
            first + 100_000_000
        );
        assert!(required_for_weights(&env.work(), &budget, 16384).is_err());
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
    fn absent_binary_uses_installed_sibling_without_launching() {
        let env = TestEnv::new();
        let mut local = settings();
        local.as_object_mut().unwrap().remove("binary");
        let private = env.work().join("private");
        let engine = configured_engine(Some(&local), private.clone(), StreamOpts::default()).unwrap();
        assert_eq!(engine.info(None).unwrap().model, "synthetic");
        assert!(!private.exists());
    }
    #[test]
    fn missing_weights_are_reported_before_child_authentication_or_launch() {
        use oxa_proto::{CancelToken, ChatRequest, FrozenPrefix};
        let env = TestEnv::new();
        let private = env.work().join("private");
        let engine = configured_engine(Some(&settings()), private.clone(), StreamOpts::default()).unwrap();
        let prefix = FrozenPrefix {
            sections: vec![],
            tools_json: "[]".into(),
            sha256: [0; 32],
        };
        let error = engine
            .chat(
                &ChatRequest::new(&prefix, &[], "synthetic"),
                &CancelToken::new(),
                &mut |_| panic!("missing weights produced output"),
            )
            .unwrap_err();
        assert_eq!(error.code, codes::NOT_FOUND);
        assert!(error.message.contains("weights"));
        assert!(!private.exists());
    }
    #[test]
    fn configured_context_forms_refuse_before_launch_or_private_storage() {
        let env = TestEnv::new();
        let private = env.work().join("private");
        for arguments in [
            serde_json::json!(["-c16384"]),
            serde_json::json!(["-c4096"]),
            serde_json::json!(["-c=4096"]),
            serde_json::json!(["--ctx-size=4096"]),
            serde_json::json!(["-c", "16384"]),
            serde_json::json!(["--ctx-size", "16384"]),
            serde_json::json!(["-c", "4096", "--ctx-size", "4096"]),
            serde_json::json!(["--ctx-size", "4096", "-c", "16384"]),
            serde_json::json!(["--ctx-size", "4096", "-c4096"]),
        ] {
            let mut local = settings();
            local["models"]["synthetic"]["arguments"] = arguments.clone();
            let error = match configured_engine(Some(&local), private.clone(), StreamOpts::default()) {
                Ok(_) => panic!("unsupported context accepted: {arguments}"),
                Err(error) => error,
            };
            assert_eq!(error.code, codes::INVALID);
            assert!(error.message.contains("context must match one"));
            assert!(!private.exists());
        }
        for arguments in [
            serde_json::json!([]),
            serde_json::json!(["-c", "4096"]),
            serde_json::json!(["--ctx-size", "4096"]),
        ] {
            let mut local = settings();
            local["models"]["synthetic"]["arguments"] = arguments;
            let engine = configured_engine(Some(&local), private.clone(), StreamOpts::default()).unwrap();
            assert_eq!(engine.info(Some("synthetic")).unwrap().n_ctx, 4096);
            assert!(!private.exists());
        }
    }
    #[test]
    fn flash_spellings_refuse_before_child_authentication_or_spawn() {
        use std::os::unix::fs::PermissionsExt;
        let env = TestEnv::new();
        env.write("work/synthetic.gguf", "Synthetic placeholder; never loaded");
        env.write("work/oxe-server", "Synthetic placeholder; must never execute");
        let binary = env.work().join("oxe-server");
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
        let prefix = oxa_proto::FrozenPrefix {
            sections: vec![],
            tools_json: "[]".into(),
            sha256: [0; 32],
        };
        for arguments in [
            serde_json::json!(["--engine", "flash"]),
            serde_json::json!(["--engine=flash"]),
        ] {
            let mut local = settings();
            local["binary"] = serde_json::json!(binary);
            local["models"]["synthetic"]["model"] = serde_json::json!(env.work().join("synthetic.gguf"));
            local["models"]["synthetic"]["arguments"] = arguments.clone();
            let private = env.work().join("private");
            let engine = configured_engine(Some(&local), private.clone(), StreamOpts::default()).unwrap();
            assert!(!private.exists());
            let error = engine
                .chat(
                    &oxa_proto::ChatRequest::new(&prefix, &[], "synthetic"),
                    &oxa_proto::CancelToken::new(),
                    &mut |_| panic!("Flash refusal generated output"),
                )
                .unwrap_err();
            assert_eq!(error.code, codes::UNSUPPORTED_IN_OXA, "{arguments}");
            assert_eq!(
                error.message,
                "Flash streaming-budget admission in the resident local factory is not supported in oxa"
            );
            assert!(!private.exists(), "{arguments} reached child authentication");
        }
    }
    #[test]
    fn binary_location_is_a_pure_sibling_not_a_path_or_cwd_search() {
        assert_eq!(
            sibling_binary(std::path::Path::new("/opt/example/bin/oxa")).unwrap(),
            PathBuf::from("/opt/example/bin/oxe-server")
        );
        assert!(sibling_binary(std::path::Path::new("relative/oxa")).is_err());
    }
    #[test]
    fn missing_or_nonexecutable_binary_has_install_guidance_before_launch() {
        let env = TestEnv::new();
        env.write(
            "work/synthetic.gguf",
            "Synthetic metadata placeholder; never loaded",
        );
        env.write("work/not-executable", "Synthetic placeholder; never executed");
        for binary in [env.work().join("missing-oxe"), env.work().join("not-executable")] {
            let mut local = settings();
            local["binary"] = serde_json::json!(binary);
            local["models"]["synthetic"]["model"] = serde_json::json!(env.work().join("synthetic.gguf"));
            let private = env.work().join("private");
            let engine = configured_engine(Some(&local), private.clone(), StreamOpts::default()).unwrap();
            let prefix = oxa_proto::FrozenPrefix {
                sections: vec![],
                tools_json: "[]".into(),
                sha256: [0; 32],
            };
            let error = engine
                .chat(
                    &oxa_proto::ChatRequest::new(&prefix, &[], "synthetic"),
                    &oxa_proto::CancelToken::new(),
                    &mut |_| panic!("missing engine generated output"),
                )
                .unwrap_err();
            assert_eq!(error.code, codes::NOT_FOUND);
            assert!(error.message.contains("install oxe-server beside oxa"));
            assert!(!private.exists());
        }
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
