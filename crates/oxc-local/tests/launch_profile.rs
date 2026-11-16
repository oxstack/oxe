// No engine process or filesystem fixture is needed.
use oxc_local::launch_profile::{from_user_settings, Error, InstalledLayout, InstalledModel};
use serde_json::json;
use std::collections::BTreeMap;
use std::ffi::OsString;

fn layout() -> InstalledLayout {
    InstalledLayout {
        binary: "/opt/example/bin/oxe-server".into(),
        models: BTreeMap::from([(
            "local-model".into(),
            InstalledModel {
                model: "/opt/example/models/model.gguf".into(),
                arguments: vec!["--ctx-size".into(), "8192".into()],
            },
        )]),
        environment: BTreeMap::from([
            ("LANG".into(), "C.UTF-8".into()),
            ("GGML_OLD_TUNING".into(), "discard".into()),
            ("LLAMA_OLD_TUNING".into(), "discard".into()),
        ]),
    }
}

#[test]
fn installed_layout_needs_no_profile_or_filesystem_probe() {
    let p = from_user_settings(&json!({}), &layout(), "local-model", 8080).unwrap();
    assert_eq!(p.binary.to_str(), Some("/opt/example/bin/oxe-server"));
    assert_eq!(
        p.arguments,
        [
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--model",
            "/opt/example/models/model.gguf",
            "--alias",
            "local-model",
            "--ctx-size",
            "8192"
        ]
        .map(OsString::from)
    );
    assert_eq!(p.environment, BTreeMap::from([("LANG".into(), "C.UTF-8".into())]));
}

#[test]
fn user_paths_and_arguments_are_literal_and_replace_installed_tuning() {
    let settings = json!({"oxa":{"local":{"models":{"local-model":{
        "binary":"/other path/oxe-server", "model":"/other path/$(touch marker).gguf",
        "arguments":["--ctx-size","4096","--label","literal; $(false)"],
        "environment":{"LANG":"C","GGML_EXPLICIT":"yes","LITERAL":"$(false)"}
    }}}}});
    let p = from_user_settings(&settings, &layout(), "local-model", 9123).unwrap();
    assert_eq!(p.model.to_str(), Some("/other path/$(touch marker).gguf"));
    assert_eq!(
        &p.arguments[8..],
        ["--ctx-size", "4096", "--label", "literal; $(false)"].map(OsString::from)
    );
    assert_eq!(p.environment["LANG"], "C");
    assert_eq!(p.environment["GGML_EXPLICIT"], "yes");
    assert_eq!(p.environment["LITERAL"], "$(false)");
    let debug = format!("{p:?}");
    assert!(!debug.contains("other path"));
    assert!(!debug.contains("false"));
}

#[test]
fn malformed_settings_never_fall_back_to_a_different_profile() {
    for settings in [
        json!(null),
        json!({"oxa":false}),
        json!({"oxa":{"local":[]}}),
        json!({"oxa":{"local":{"models":{"local-model":null}}}}),
        json!({"oxa":{"local":{"models":{"local-model":{"argumnts":[]}}}}}),
    ] {
        assert_eq!(
            from_user_settings(&settings, &layout(), "local-model", 8080),
            Err(Error::Settings)
        );
    }
    assert_eq!(
        from_user_settings(&json!({}), &layout(), "unknown", 8080),
        Err(Error::UnknownModel)
    );
    assert_eq!(
        from_user_settings(&json!({}), &layout(), "local-model", 0),
        Err(Error::Port)
    );
}

#[test]
fn invalid_paths_and_interpreters_are_refused_without_opening_them() {
    for binary in [
        "relative",
        "/opt/../bin/oxe-server",
        "/usr/bin/python3",
        "/usr/bin/sh",
        "/opt/engine.py",
    ] {
        let settings = json!({"oxa":{"local":{"models":{"local-model":{"binary":binary}}}}});
        assert_eq!(
            from_user_settings(&settings, &layout(), "local-model", 8080),
            Err(Error::Binary)
        );
    }
    let mut installed = layout();
    installed.models.get_mut("local-model").unwrap().model = "relative.gguf".into();
    assert_eq!(
        from_user_settings(&json!({}), &installed, "local-model", 8080),
        Err(Error::Model)
    );
}

#[test]
fn profile_cannot_override_binding_model_or_alias_through_extra_arguments() {
    for flag in [
        "--host",
        "--host=0.0.0.0",
        "--port",
        "--port=42",
        "--listen",
        "--bind",
        "--model",
        "--model=other",
        "--alias",
        "-m",
        "-mother",
    ] {
        let settings = json!({"oxa":{"local":{"models":{"local-model":{"arguments":[flag]}}}}});
        assert_eq!(
            from_user_settings(&settings, &layout(), "local-model", 8080),
            Err(Error::OwnedArgument)
        );
    }
}

#[test]
fn argument_and_environment_types_are_checked() {
    for profile in [
        json!({"arguments":false}),
        json!({"arguments":[1]}),
        json!({"arguments":["a\u{0000}b"]}),
    ] {
        let settings = json!({"oxa":{"local":{"models":{"local-model":profile}}}});
        assert_eq!(
            from_user_settings(&settings, &layout(), "local-model", 8080),
            Err(Error::Arguments)
        );
    }
    for profile in [
        json!({"environment":[]}),
        json!({"environment":{"KEY":1}}),
        json!({"environment":{"":"value"}}),
        json!({"environment":{"A=B":"value"}}),
        json!({"environment":{"KEY":"a\u{0000}b"}}),
    ] {
        let settings = json!({"oxa":{"local":{"models":{"local-model":profile}}}});
        assert_eq!(
            from_user_settings(&settings, &layout(), "local-model", 8080),
            Err(Error::Environment)
        );
    }
}

#[test]
fn resolving_one_model_does_not_mutate_or_reuse_another_models_overrides() {
    let mut installed = layout();
    installed.models.insert(
        "other".into(),
        InstalledModel {
            model: "/opt/example/models/other.gguf".into(),
            arguments: vec![],
        },
    );
    let settings =
        json!({"oxa":{"local":{"models":{"local-model":{"arguments":[],"environment":{"ONE":"1"}}}}}});
    let before = settings.clone();
    let one = from_user_settings(&settings, &installed, "local-model", 8080).unwrap();
    let other = from_user_settings(&settings, &installed, "other", 8081).unwrap();
    assert_eq!(one.arguments.len(), 8);
    assert!(!other.environment.contains_key("ONE"));
    assert_eq!(other.model.to_str(), Some("/opt/example/models/other.gguf"));
    assert_eq!(settings, before);
}

#[test]
fn invalid_layout_values_fail_before_any_supervision_work() {
    let mut installed = layout();
    installed.binary = "/".into();
    assert_eq!(
        from_user_settings(&json!({}), &installed, "local-model", 8080),
        Err(Error::Binary)
    );
    installed = layout();
    installed.environment.insert("BAD=NAME".into(), "value".into());
    assert_eq!(
        from_user_settings(&json!({}), &installed, "local-model", 8080),
        Err(Error::Environment)
    );
    assert_eq!(
        from_user_settings(&json!({}), &layout(), "bad\nmodel", 8080),
        Err(Error::Alias)
    );
}

#[test]
fn short_alias_cannot_replace_selected_identity_from_either_argument_source() {
    let mut installed = layout();
    installed.models.get_mut("local-model").unwrap().arguments = vec!["-a".into(), "other-model".into()];
    assert_eq!(
        from_user_settings(&json!({}), &installed, "local-model", 8080),
        Err(Error::OwnedArgument)
    );
    let settings = json!({"oxa":{"local":{"models":{"local-model":{
        "arguments":["-a","other-model"]
    }}}}});
    assert_eq!(
        from_user_settings(&settings, &layout(), "local-model", 8080),
        Err(Error::OwnedArgument)
    );
}

#[test]
fn shipped_moe_aliases_default_to_sampled_mtp_with_their_existing_cutoffs() {
    for (alias, cutoff) in [
        ("ornith-35b-a3b-local", "32768"),
        ("qwen3.6-35b-a3b-local", "16384"),
    ] {
        let mut installed = layout();
        installed
            .models
            .insert(alias.into(), installed.models["local-model"].clone());
        let profile = from_user_settings(&json!({}), &installed, alias, 8080).unwrap();
        assert_eq!(
            &profile.arguments[8..],
            ["--ctx-size", "8192", "--mtp-max-ctx", cutoff, "--mtp-sampled"].map(OsString::from)
        );
    }
}

#[test]
fn model_mtp_settings_override_defaults_without_changing_other_models() {
    let alias = "ornith-35b-a3b-local";
    let mut installed = layout();
    installed
        .models
        .insert(alias.into(), installed.models["local-model"].clone());
    let user = json!({"oxa":{"local":{"models":{(alias):{"mtp_max_context":4096,"sampled_mtp":false}}}}});
    let profile = from_user_settings(&user, &installed, alias, 8080).unwrap();
    assert_eq!(
        &profile.arguments[8..],
        ["--ctx-size", "8192", "--mtp-max-ctx", "4096"].map(OsString::from)
    );
    let unrelated = from_user_settings(&user, &installed, "local-model", 8080).unwrap();
    assert_eq!(
        &unrelated.arguments[8..],
        ["--ctx-size", "8192"].map(OsString::from)
    );
}

#[test]
fn explicit_literal_mtp_flags_remain_the_callers_choice() {
    let alias = "qwen3.6-35b-a3b-local";
    let mut installed = layout();
    installed.models.insert(
        alias.into(),
        InstalledModel {
            model: "/models/synthetic.gguf".into(),
            arguments: vec!["--mtp-max-ctx".into(), "8192".into(), "--mtp-sampled".into()],
        },
    );
    let profile = from_user_settings(&json!({}), &installed, alias, 8080).unwrap();
    assert_eq!(
        &profile.arguments[8..],
        ["--mtp-max-ctx", "8192", "--mtp-sampled"].map(OsString::from)
    );
}

#[test]
fn malformed_or_ambiguous_mtp_settings_are_refused() {
    for settings in [
        json!({"sampled_mtp":"true"}),
        json!({"sampled_mtp":null}),
        json!({"mtp_max_context":0}),
        json!({"mtp_max_context":-1}),
        json!({"mtp_max_context":1.5}),
        json!({"mtp_max_context":4294967296_u64}),
        json!({"sampled_mtp":false,"arguments":["--mtp-sampled"]}),
        json!({"mtp_max_context":4096,"arguments":["--mtp-max-ctx=8192"]}),
    ] {
        let user = json!({"oxa":{"local":{"models":{"local-model":settings}}}});
        assert!(from_user_settings(&user, &layout(), "local-model", 8080).is_err());
    }
}

#[test]
fn equals_form_mtp_flags_are_rejected_before_launch() {
    for arguments in [vec!["--mtp-max-ctx=8192"], vec!["--mtp-sampled=true"]] {
        let user = json!({"oxa":{"local":{"models":{"local-model":{"arguments":arguments}}}}});
        assert_eq!(
            from_user_settings(&user, &layout(), "local-model", 8080),
            Err(Error::Arguments)
        );
    }
}
