//! Checked full-resident weights plus explicit anonymous allocation accounting.
use oxa_proto::{codes, OxaError};
use serde_json::Value;
use std::ffi::OsString;

pub(crate) fn invalid() -> OxaError {
    OxaError::new(codes::INVALID,
        "Local admission requires oxa.local.limits.<alias>.allocation_budget: kv_bytes_per_position, scratch_bytes, mtp_state_bytes and positive margin_bytes. Supply qualified allocation bounds for the configured context/arguments, not the weights file size.")
}

pub(crate) fn required(value: &Value, context: u32, weights_bytes: u64) -> Result<u64, OxaError> {
    let object = value.as_object().ok_or_else(invalid)?;
    let fields = [
        "kv_bytes_per_position",
        "scratch_bytes",
        "mtp_state_bytes",
        "margin_bytes",
    ];
    if object.len() != fields.len()
        || object.keys().any(|key| !fields.contains(&key.as_str()))
        || context == 0
    {
        return Err(invalid());
    }
    let bytes = |key: &str| value[key].as_u64().ok_or_else(invalid);
    if bytes("margin_bytes")? == 0 {
        return Err(invalid());
    }
    // The engine pads all context-sized allocations to 256 positions.
    let capacity = u64::from(context).div_ceil(256) * 256;
    let kv = capacity
        .checked_mul(bytes("kv_bytes_per_position")?)
        .ok_or_else(invalid)?;
    ["scratch_bytes", "mtp_state_bytes", "margin_bytes"]
        .into_iter()
        .try_fold(
            kv.checked_add(weights_bytes).ok_or_else(invalid)?,
            |total, key| total.checked_add(bytes(key)?).ok_or_else(invalid),
        )
}

/// Bind the allocation calculation to the actual native context, including -c.
/// No duplicate/last-wins or unsupported equals-form authority is accepted.
pub(crate) fn bind_context(arguments: &mut Vec<OsString>, context: u32) -> Result<(), OxaError> {
    let mut found = false;
    for (index, word) in arguments.iter().enumerate() {
        let Some(word) = word.to_str() else { continue };
        if word.starts_with("--ctx-size=") || word.starts_with("-c=") {
            return Err(context_error());
        }
        if matches!(word, "--ctx-size" | "-c") {
            if found
                || arguments
                    .get(index + 1)
                    .and_then(|v| v.to_str())
                    .and_then(|v| v.parse::<u32>().ok())
                    != Some(context)
            {
                return Err(context_error());
            }
            found = true;
        }
    }
    if !found {
        arguments.extend(["--ctx-size".into(), context.to_string().into()]);
    }
    Ok(())
}
fn context_error() -> OxaError {
    OxaError::new(codes::INVALID, "Local admission context must match one --ctx-size N or -c N argument and oxa.local.limits.<alias>.context_window; remove duplicate or conflicting context arguments.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn budget() -> Value {
        // Synthetic non-KV bounds, not qualification of the laptop/model.
        json!({"kv_bytes_per_position":22528, "scratch_bytes":1048576,
            "mtp_state_bytes":1048576,"margin_bytes":1048576})
    }
    #[test]
    fn qwen_kv_depends_on_context_and_includes_mtp_layer_once() {
        let fixed = 3 * 1048576;
        assert_eq!(required(&budget(), 131072, 0).unwrap() - fixed, 2952790016); // 2.75 GiB
        assert_eq!(required(&budget(), 16384, 0).unwrap() - fixed, 369098752); // 0.34375 GiB
        assert_eq!(required(&budget(), 257, 0).unwrap() - fixed, 512 * 22528);
    }
    #[test]
    fn qwen_full_weights_are_resident_and_not_an_optional_working_set() {
        let value = budget();
        let anonymous = required(&value, 16384, 0).unwrap();
        assert_eq!(
            required(&value, 16384, 22_700_000_000).unwrap(),
            anonymous + 22_700_000_000
        );
        assert!(required(&value, 16384, u64::MAX).is_err());
    }
    #[test]
    fn unknown_missing_invalid_or_overflowing_components_fail_closed() {
        for key in [
            "kv_bytes_per_position",
            "scratch_bytes",
            "mtp_state_bytes",
            "margin_bytes",
        ] {
            let mut value = budget();
            value.as_object_mut().unwrap().remove(key);
            assert!(required(&value, 16384, 0).is_err());
            for bad in [Value::Null, json!(-1), json!(1.5), json!("0")] {
                let mut value = budget();
                value[key] = bad;
                assert!(required(&value, 16384, 0).is_err());
            }
        }
        for (key, bad) in [
            ("margin_bytes", 0),
            ("kv_bytes_per_position", u64::MAX),
            ("scratch_bytes", u64::MAX),
        ] {
            let mut value = budget();
            value[key] = json!(bad);
            assert!(required(&value, 16384, 0).is_err());
        }
        let mut value = budget();
        value["weights_file_size"] = json!(22700000000_u64);
        assert!(required(&value, 16384, 0).is_err());
    }
    #[test]
    fn launched_context_is_explicit_and_cannot_diverge_from_budget() {
        let mut words = vec!["--engine".into(), "stub".into()];
        bind_context(&mut words, 16384).unwrap();
        assert_eq!(
            &words[2..],
            [OsString::from("--ctx-size"), OsString::from("16384")]
        );
        for flag in ["--ctx-size", "-c"] {
            let mut words = vec![flag.into(), "16384".into()];
            bind_context(&mut words, 16384).unwrap();
            assert_eq!(words.len(), 2);
        }
        for words in [
            vec!["--ctx-size", "131072"],
            vec!["-c", "131072"],
            vec!["--ctx-size=16384"],
            vec!["--ctx-size", "16384", "-c", "16384"],
            vec!["--ctx-size"],
        ] {
            assert!(bind_context(&mut words.into_iter().map(OsString::from).collect(), 16384).is_err());
        }
    }
}
