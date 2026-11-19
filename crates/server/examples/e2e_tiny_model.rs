//! Writes the tiny synthetic model the end-to-end job (`.github/workflows/e2e.yml`) serves:
//! the 4-layer qwen35moe from `model::testkit` plus a byte-level tokenizer and the Qwen3.x chat
//! template in the GGUF header, so `oxe-server --engine oxe --model FILE` loads it alone. The
//! weights are seeded filler, so the file is byte-identical on every run (its sha256 is pinned in
//! `tools/ci/e2e-tiny.toml`).
//!
//! usage: cargo run --release -p server --example e2e_tiny_model -- OUT.gguf

use model::testkit::{mini_model_with, GgufWriter};

const TEMPLATE: &str = include_str!("../../chat/src/fixtures/qwen38-flash-next.jinja");
const SPECIALS: &[(&str, i32)] = &[
    ("<|endoftext|>", 3),
    ("<|im_start|>", 3),
    ("<|im_end|>", 3),
    ("<think>", 4),
    ("</think>", 4),
    ("<tool_call>", 4),
    ("</tool_call>", 4),
    ("<tool_response>", 4),
    ("</tool_response>", 4),
];

fn str_array(w: &mut GgufWriter, k: &str, v: &[String]) {
    w.key(k, 9);
    w.kv.extend(8u32.to_le_bytes());
    w.kv.extend((v.len() as u64).to_le_bytes());
    for s in v {
        w.kv.extend((s.len() as u64).to_le_bytes());
        w.kv.extend(s.as_bytes());
    }
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: e2e_tiny_model OUT.gguf");
    let table = tokenizer::unicode::byte_to_cpt_table();
    let mut toks: Vec<String> = table.iter().map(|c| c.to_string()).collect();
    let mut types = vec![1i32; 256];
    toks.push("ab".into());
    types.push(1);
    for (s, t) in SPECIALS {
        toks.push((*s).into());
        types.push(*t);
    }
    // the model's vocabulary must be a multiple of 8: pad with plain tokens
    let pad = toks.len().next_multiple_of(8) - toks.len();
    for i in 0..pad {
        toks.push(format!("zq{i}"));
        types.push(1);
    }
    let id = |s: &str| 257 + SPECIALS.iter().position(|(x, _)| *x == s).unwrap() as u32;
    let vocab = toks.len() as u64;
    mini_model_with(std::path::Path::new(&out), false, 65_536, 2, vocab, &|w| {
        w.str("tokenizer.ggml.model", "gpt2");
        w.str("tokenizer.ggml.pre", "qwen2");
        str_array(w, "tokenizer.ggml.tokens", &toks);
        w.arr_i32("tokenizer.ggml.token_type", &types);
        str_array(w, "tokenizer.ggml.merges", &["a b".to_string()]);
        w.u32("tokenizer.ggml.eos_token_id", id("<|im_end|>"));
        w.u32("tokenizer.ggml.padding_token_id", id("<|endoftext|>"));
        w.str("tokenizer.chat_template", TEMPLATE);
    });
}
