use crate::{err, Result};
use regex::Regex;
use std::collections::BTreeMap;

pub const AUTO_PREFIX: &str = "Local AI memlat (";
pub const AUTO_COMMENT: &str = "Auto-rebuilt Yoga MEMLAT; stock fallback Omarchy/linux-omarchy";

pub fn regex(pattern: &str) -> Regex {
    Regex::new(pattern).expect("constant regular expression")
}
#[derive(Debug)]
pub struct Entry {
    pub index: usize,
    pub path: Option<String>,
    pub efi: bool,
    pub cmdline: Option<String>,
    pub top: bool,
    pub comment: Option<String>,
}
pub type Menu = BTreeMap<String, Entry>;
pub fn headers() -> Regex {
    regex(r"(?m)^\s*(/+)([^/\r\n][^\r\n]*)$")
}
fn one(values: Vec<String>) -> Option<String> {
    if values.len() == 1 {
        values.into_iter().next()
    } else {
        None
    }
}
fn fields(body: &str, field: &str, token: bool) -> Vec<String> {
    let pattern = if token {
        format!(r"(?mi)^\s*{field}\s*:\s*(\S+)\s*$")
    } else {
        format!(r"(?mi)^[ \t]*{field}[ \t]*:[ \t]*(.*?)[ \t]*$")
    };
    regex(&pattern)
        .captures_iter(body)
        .map(|c| c[1].to_string())
        .collect()
}
pub fn entries(text: &str) -> Result<Menu> {
    if regex(r"(?mi)^\s*(include|config_path)\s*:").is_match(text) {
        return err("included menus need manual review");
    }
    let re = headers();
    let hs: Vec<_> = re.captures_iter(text).collect();
    let mut stack = Vec::new();
    let mut result = Menu::new();
    for (i, h) in hs.iter().enumerate() {
        let depth = h[1].len();
        let name = h[2].trim_start_matches('+');
        if name.contains(['\\', '/', '#']) || depth > stack.len() + 1 {
            return err("unsupported menu name or nesting");
        }
        stack.truncate(depth - 1);
        stack.push(name.to_string());
        let key = stack.join("/");
        if result.contains_key(&key) {
            return err("duplicate menu path");
        }
        let end = hs.get(i + 1).map_or(text.len(), |x| x.get(0).unwrap().start());
        let body = &text[h.get(0).unwrap().end()..end];
        let cmdlines = fields(body, "cmdline", false);
        if cmdlines.len() > 1 {
            return err("duplicate entry command lines need manual review");
        }
        result.insert(
            key,
            Entry {
                index: i + 1,
                path: one(fields(body, "path", true)),
                efi: fields(body, "protocol", true) == ["efi"],
                cmdline: one(cmdlines),
                top: depth == 1,
                comment: one(fields(body, "comment", false)),
            },
        );
    }
    Ok(result)
}
pub fn selected(text: &str, menu: &Menu) -> Result<String> {
    let re = regex(r"(?mi)^\s*default_entry\s*:\s*(\S.*?)\s*$");
    let defaults: Vec<_> = re.captures_iter(text).map(|c| c[1].to_string()).collect();
    if defaults.len() != 1 || regex(r"(?mi)^\s*remember_last_entry\s*:\s*yes").is_match(text) {
        return err("require one explicit default with remember_last_entry disabled");
    }
    let value = &defaults[0];
    if value.chars().all(|c| c.is_ascii_digit()) {
        let n = value.parse::<usize>().ok();
        return menu
            .iter()
            .find(|(_, e)| Some(e.index) == n)
            .map(|(k, _)| k.clone())
            .ok_or_else(|| "unresolved numeric default".into());
    }
    Ok(value.clone())
}
pub fn replace_default(text: &str, target: &str) -> String {
    regex(r"(?mi)^(\s*default_entry\s*:)\s*.*$")
        .replacen(text, 1, |c: &regex::Captures<'_>| format!("{} {target}", &c[1]))
        .into_owned()
}
pub fn remove_block(text: &str, name: &str, missing: bool) -> Result<String> {
    let re = headers();
    let hs: Vec<_> = re.captures_iter(text).collect();
    let matches: Vec<_> = hs
        .iter()
        .enumerate()
        .filter(|(_, h)| &h[1] == "/" && &h[2] == name)
        .collect();
    if matches.is_empty() && missing {
        return Ok(text.into());
    }
    if matches.len() != 1 {
        return err("automatic menu entry missing or duplicated");
    }
    let (i, h) = matches[0];
    let end = hs.get(i + 1).map_or(text.len(), |h| h.get(0).unwrap().start());
    Ok(format!("{}{}", &text[..h.get(0).unwrap().start()], &text[end..]))
}
pub fn semantic_fields(block: &str) -> Result<Vec<(String, String)>> {
    let mut result = Vec::new();
    for line in block.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('/') {
            result.push(("header".into(), line.into()));
            continue;
        }
        let (key, value) = line.split_once(':').ok_or("unsupported automatic menu field")?;
        result.push((key.trim().to_lowercase(), value.trim().into()));
    }
    result.sort();
    Ok(result)
}
