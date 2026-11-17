//! Actual native factory with stub-only synthetic model input. Use fake kernel
//! metadata in the test mount namespace; this does not qualify real hardware.
use oxa_engine::StreamOpts;
use oxa_proto::{CancelToken, ChatRequest, FrozenPrefix};
use std::{io, path::PathBuf};

fn footprint(label: &str) -> io::Result<()> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    for line in status
        .lines()
        .filter(|line| line.starts_with("VmRSS:") || line.starts_with("Threads:"))
    {
        println!("{label} {line}");
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let binary = PathBuf::from(args.next().expect("absolute stub engine"));
    let model = PathBuf::from(args.next().expect("synthetic GGUF"));
    let private = PathBuf::from(args.next().expect("throwaway private directory"));
    assert!(args.next().is_none());
    let local = serde_json::json!({"binary":binary,"models":{"synthetic":{"model":model,"arguments":["--engine","stub","--ctx-size","4096"]}},
        "limits":{"synthetic":{"context_window":4096,"max_output":64,"allocation_budget":{"kv_bytes_per_position":1,"scratch_bytes":0,"mtp_state_bytes":0,"margin_bytes":1044480}}},"idle_unload_seconds":0.1});
    let engine = oxc_local::configured_engine(Some(&local), private.clone(), StreamOpts::default())
        .map_err(io::Error::other)?;
    assert_eq!(engine.info(None).map_err(io::Error::other)?.n_ctx, 4096);
    assert!(!private.exists());
    footprint("cold")?;
    let prefix = FrozenPrefix {
        sections: vec![("system".into(), "Synthetic native task".into())],
        tools_json: "[]".into(),
        sha256: [0; 32],
    };
    let request = ChatRequest {
        max_tokens: 4,
        ..ChatRequest::new(&prefix, &[], "synthetic")
    };
    let mut increments = 0;
    let reply = engine
        .chat(&request, &CancelToken::new(), &mut |_| {
            increments += 1;
        })
        .map_err(io::Error::other)?;
    assert!(reply.usage.completion > 0 && increments > 0);
    footprint("resident")?;
    drop(engine);
    footprint("stopped")?;
    assert_eq!(std::fs::read_dir(private)?.count(), 0);
    println!("PASS: native factory, typed discovery, authenticated stub chat/stream and child/auth cleanup; no controller");
    Ok(())
}
