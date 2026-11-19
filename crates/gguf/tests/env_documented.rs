//! Every `OXE_*` / `OXF_*` / `OXI_*` name the engine reads is a row of `docs/ENVIRONMENT.md`, and
//! every row has a reader. The test only reads files (gguf is the smallest crate; no new dependency).
//!
//! Reader = a string literal matching `^OX[EFI]_[A-Z0-9_]+$` in non-test source of
//! `crates/*/src/**/*.rs` or `crates/*/build.rs`. Not scanned: files named `tests*.rs`, files under
//! a `tests/` directory, line comments, and everything from an inline `#[cfg(test)] mod x {`.
//! Two registries are not readers and are skipped: `server/src/startup_check.rs` (the list of
//! known names it validates) and the `ARITH_KNOBS` array of `server/src/state_store.rs` (names
//! that go into the state identity). Names built at run time would need a prefix entry here; the
//! engine has none today.
//!
//! Rows of class `test` are read by tests, fixtures or benches only, so they are exempt from the
//! "has a non-test reader" direction; they must still appear in some `.rs` file of the workspace.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

fn engine_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn is_env_name(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() > 4
        && b[0] == b'O'
        && b[1] == b'X'
        && matches!(b[2], b'E' | b'F' | b'I')
        && b[3] == b'_'
        && b[4..]
            .iter()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == b'_')
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

fn crate_dirs() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(engine_root().join("crates"))
        .expect("engine/crates")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    v.sort();
    v
}

/// Non-test text of a source file: no line comments, no registry array, nothing from an inline
/// `#[cfg(test)] mod x {`.
fn non_test_text(raw: &str) -> String {
    let lines: Vec<&str> = raw.lines().collect();
    let mut out = String::new();
    let mut i = 0;
    let mut in_arith = false;
    while i < lines.len() {
        let t = lines[i].trim();
        if in_arith {
            if t == "];" {
                in_arith = false;
            }
            i += 1;
            continue;
        }
        if t.contains("const ARITH_KNOBS") {
            in_arith = true;
            i += 1;
            continue;
        }
        if t.starts_with("#[cfg(") && t.contains("test") && !t.contains("not(") {
            // the item the attribute belongs to: skip further attributes
            let mut j = i + 1;
            while j < lines.len() && lines[j].trim().starts_with("#[") {
                j += 1;
            }
            let item = lines.get(j).map_or("", |l| l.trim());
            let is_mod = item.starts_with("mod ")
                || item.starts_with("pub mod ")
                || item.starts_with("pub(crate) mod ");
            if is_mod && item.ends_with('{') {
                break;
            }
        }
        if !t.starts_with("//") {
            out.push_str(lines[i]);
            out.push('\n');
        }
        i += 1;
    }
    out
}

/// Every `"OXE_..."`-shaped string literal of `text`.
fn literals(text: &str) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let mut rest = text;
    while let Some(at) = rest.find("\"OX") {
        let after = &rest[at + 1..];
        match after.find('"') {
            Some(end) => {
                if is_env_name(&after[..end]) {
                    found.insert(after[..end].to_string());
                }
                rest = &after[end + 1..];
            }
            None => break,
        }
    }
    found
}

/// name -> files (relative to crates/) that read it, from non-test source.
fn names_read() -> BTreeMap<String, Vec<String>> {
    let mut files = Vec::new();
    for c in crate_dirs() {
        rs_files(&c.join("src"), &mut files);
        let b = c.join("build.rs");
        if b.is_file() {
            files.push(b);
        }
    }
    let crates = engine_root().join("crates");
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for f in files {
        let rel = f
            .strip_prefix(&crates)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        let in_tests_dir = f
            .strip_prefix(&crates)
            .unwrap()
            .components()
            .any(|c| c.as_os_str() == "tests");
        if name.starts_with("tests") || in_tests_dir || name == "startup_check.rs" {
            continue;
        }
        let raw = fs::read_to_string(&f).unwrap_or_else(|e| panic!("{rel}: {e}"));
        for n in literals(&non_test_text(&raw)) {
            map.entry(n).or_default().push(rel.clone());
        }
    }
    map
}

/// name -> class, from the first and last column of the tables of ENVIRONMENT.md.
fn documented() -> BTreeMap<String, String> {
    let path = engine_root().join("docs/ENVIRONMENT.md");
    let doc = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut rows = BTreeMap::new();
    for line in doc.lines().filter(|l| l.starts_with('|')) {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if cells.len() < 4 {
            continue;
        }
        let name = cells[1].trim_matches('`');
        if !is_env_name(name) {
            continue;
        }
        let class = cells[cells.len() - 2].trim_matches('`').to_string();
        assert!(
            ["option", "tuning", "experiment", "test", "build"].contains(&class.as_str()),
            "ENVIRONMENT.md: {name} has class {class:?}"
        );
        assert!(
            rows.insert(name.to_string(), class).is_none(),
            "ENVIRONMENT.md: {name} has two rows"
        );
    }
    rows
}

/// Every name that occurs as a literal in any `.rs` file of any crate (tests, benches included).
fn names_anywhere() -> BTreeSet<String> {
    let mut files = Vec::new();
    for c in crate_dirs() {
        rs_files(&c, &mut files);
    }
    let mut all = BTreeSet::new();
    for f in files {
        if f.components().any(|c| c.as_os_str() == "target") {
            continue;
        }
        if let Ok(raw) = fs::read_to_string(&f) {
            all.extend(literals(&raw));
        }
    }
    all
}

#[test]
fn every_name_read_has_a_row() {
    let doc = documented();
    let missing: Vec<String> = names_read()
        .into_iter()
        .filter(|(n, _)| !doc.contains_key(n))
        .map(|(n, f)| format!("{n} (read in {})", f.join(", ")))
        .collect();
    assert!(
        missing.is_empty(),
        "environment names read without a row in engine/docs/ENVIRONMENT.md:\n  {}",
        missing.join("\n  ")
    );
}

#[test]
fn every_row_has_a_reader() {
    let read = names_read();
    let anywhere = names_anywhere();
    let orphans: Vec<String> = documented()
        .into_iter()
        .filter(|(n, class)| {
            if class == "test" {
                !anywhere.contains(n)
            } else {
                !read.contains_key(n)
            }
        })
        .map(|(n, class)| format!("{n} (class {class})"))
        .collect();
    assert!(
        orphans.is_empty(),
        "rows of engine/docs/ENVIRONMENT.md with no reader (remove the row or fix its class):\n  {}",
        orphans.join("\n  ")
    );
}
