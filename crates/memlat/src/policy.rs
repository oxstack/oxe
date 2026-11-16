use crate::{err, menu, replace, sha256, Result};
use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
pub fn string<'a>(p: &'a Value, k: &str) -> Result<&'a str> {
    p.get(k)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("'{k}'").into())
}
pub fn path(p: &Value, k: &str) -> Result<PathBuf> {
    Ok(string(p, k)?.into())
}
pub fn optional<'a>(p: &'a Value, k: &str, default: &'a str) -> &'a str {
    p.get(k).and_then(Value::as_str).unwrap_or(default)
}
pub fn blake2b(bytes: &[u8]) -> Result<String> {
    let mut child = Command::new("b2sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let mut input = child.stdin.take().ok_or("missing checksum input")?;
    // Drain stdout concurrently with writing only a single fixed-size hash response.
    let writing = input.write_all(bytes);
    drop(input);
    let output = child.wait_with_output()?;
    writing?;
    if !output.status.success() {
        return err("BLAKE2 checksum failed");
    }
    let text = String::from_utf8(output.stdout)?;
    let hash = text.split_whitespace().next().ok_or("missing BLAKE2 checksum")?;
    if hash.len() != 128 || !hash.bytes().all(|c| c.is_ascii_hexdigit()) {
        return err("invalid BLAKE2 checksum");
    }
    Ok(hash.into())
}
pub fn verify_entry(menu: &menu::Menu, name: &str, image: &Path, expected: Option<&str>) -> Result<()> {
    verify_at(menu, name, image, expected, Path::new("/boot"))
}
fn verify_at(menu: &menu::Menu, name: &str, image: &Path, expected: Option<&str>, boot: &Path) -> Result<()> {
    let e = menu
        .get(name)
        .filter(|e| e.efi && e.path.is_some())
        .ok_or_else(|| format!("no unambiguous EFI leaf: {name}"))?;
    let (location, checksum) = e
        .path
        .as_ref()
        .unwrap()
        .split_once('#')
        .map_or((e.path.as_deref().unwrap(), None), |(a, b)| (a, Some(b)));
    let relative = image.strip_prefix(boot)?;
    if location != format!("boot():/{}", relative.display()) || !image.is_file() {
        return err(&format!("EFI image/path mismatch: {name}"));
    }
    if let Some(checksum) = checksum {
        if checksum.len() != 128
            || !checksum.bytes().all(|c| c.is_ascii_hexdigit())
            || blake2b(&fs::read(image)?)? != checksum.to_lowercase()
        {
            return err(&format!("Limine EFI hash mismatch: {name}"));
        }
    }
    if let Some(expected) = expected.filter(|s| !s.is_empty()) {
        if sha256(image)? != expected {
            return err(&format!("EFI image changed since qualification: {name}"));
        }
    }
    Ok(())
}
pub fn auto_entries(menu: &menu::Menu, p: &Value) -> Result<Vec<String>> {
    let directory = Path::new(optional(p, "image_directory", "/boot/EFI/LocalAI"));
    let relative = directory.strip_prefix(optional(p, "boot_root", "/boot"))?;
    let relative = relative.to_string_lossy();
    let prefix = format!(
        "boot():/{}omarchy-memlat-",
        if relative.is_empty() {
            String::new()
        } else {
            format!("{relative}/")
        }
    );
    Ok(menu
        .iter()
        .filter(|(name, e)| {
            e.top
                && name.starts_with(menu::AUTO_PREFIX)
                && name.ends_with(')')
                && e.comment.as_deref() == Some(menu::AUTO_COMMENT)
                && e.efi
                && e.path
                    .as_ref()
                    .is_some_and(|path| path.strip_prefix(&prefix).is_some_and(|tail| !tail.contains('/')))
        })
        .map(|(n, _)| n.clone())
        .collect())
}
pub trait Host {
    fn verify(&self, menu: &menu::Menu, name: &str, image: &Path, expected: Option<&str>) -> Result<()> {
        verify_entry(menu, name, image, expected)
    }
    fn build(&self, p: &Value, output: &Path) -> Result<Value>;
    fn notify(&self, p: &Value, error: &str);
}
pub fn apply(p: &Value, action: &str, host: &impl Host) -> Result<Value> {
    let config = path(p, "config")?;
    let before = crate::read_text(&config)?;
    let menu = menu::entries(&before)?;
    let current = menu::selected(&before, &menu)?;
    let own = string(p, "memlat_entry")?;
    let stock = string(p, "stock_entry")?;
    let adopted = current != own && auto_entries(&menu, p)?.contains(&current);
    if action != "enable" && current != own && !adopted {
        return Ok(json!({"action":action,"changed":false,"default":current}));
    }
    host.verify(&menu, stock, &path(p, "stock_image")?, None)?;
    let target = if action == "enable" {
        if !path(p, "modules")?.is_dir() {
            return err("qualified kernel modules missing");
        }
        host.verify(
            &menu,
            own,
            &path(p, "memlat_image")?,
            Some(string(p, "memlat_sha256")?),
        )?;
        host.verify(
            &menu,
            stock,
            &path(p, "stock_image")?,
            Some(string(p, "stock_sha256")?),
        )?;
        own
    } else if action == "check" {
        let valid = !adopted
            && path(p, "modules")?.is_dir()
            && sha256(&path(p, "stock_image")?)? == string(p, "stock_sha256")?;
        let image_valid = host
            .verify(
                &menu,
                own,
                &path(p, "memlat_image")?,
                Some(string(p, "memlat_sha256")?),
            )
            .is_ok();
        if valid && image_valid {
            own
        } else {
            stock
        }
    } else {
        stock
    };
    let after = menu::replace_default(&before, target);
    if after != before {
        replace(&config, &before, &after)?;
    }
    // Pre-update failure must never leave a qualified MEMLAT default selected.
    // Commit verified STOCK before the fallible pending-marker write.
    if action == "prepare-update" {
        let pending = path(p, "pending")?;
        fs::create_dir_all(pending.parent().ok_or("missing pending parent")?)?;
        fs::write(pending, "Rebuild approved MEMLAT default after stock update.\n")?;
    }
    Ok(json!({"action":action,"changed":after!=before,"default":target}))
}
pub fn remove_owned_block(text: &str, p: &Value) -> Result<String> {
    let expected = optional(p, "automatic_entry_block", "");
    let mut text = text.to_string();
    if !expected.is_empty() {
        if text.matches(expected).count() == 1 {
            text = text.replacen(expected, "", 1);
        } else {
            let name = string(p, "memlat_entry")?;
            let re = menu::headers();
            let hs: Vec<_> = re.captures_iter(&text).collect();
            let matches: Vec<_> = hs
                .iter()
                .enumerate()
                .filter(|(_, h)| &h[1] == "/" && &h[2] == name)
                .collect();
            if matches.len() > 1 {
                return err("owned automatic menu entry duplicated");
            }
            if let Some((i, h)) = matches.first() {
                let end = hs.get(i + 1).map_or(text.len(), |h| h.get(0).unwrap().start());
                if menu::semantic_fields(expected)?
                    != menu::semantic_fields(&text[h.get(0).unwrap().start()..end])?
                {
                    return err("owned automatic menu block changed semantically");
                }
                text = menu::remove_block(&text, name, false)?;
            }
        }
    }
    for name in auto_entries(&menu::entries(&text)?, p)? {
        text = menu::remove_block(&text, &name, false)?;
    }
    Ok(text)
}

pub fn rebuild(policy_path: &Path, p: &Value, dry_run: bool, force: bool, host: &impl Host) -> Result<Value> {
    let pending = path(p, "pending")?;
    if !(pending.exists() || dry_run || force) {
        return Ok(json!({"action":"rebuild","changed":false,"reason":"no update requested"}));
    }
    let result = (|| -> Result<Value> {
        if !dry_run {
            apply(p, "fallback", host)?;
        }
        let directory = crate::Temporary::directory("memlat-rebuild-")?;
        let image = directory.path.join("candidate.efi");
        let evidence = host.build(p, &image)?;
        if dry_run {
            let mut result = json!({"action":"rebuild-dry-run","changed":false,"passed":true});
            merge(&mut result, &evidence)?;
            return Ok(result);
        }
        let config = path(p, "config")?;
        let before = crate::read_text(&config)?;
        let menu = menu::entries(&before)?;
        if menu::selected(&before, &menu)? != string(p, "stock_entry")? {
            return err("default changed during rebuild; refusing promotion");
        }
        host.verify(
            &menu,
            string(p, "stock_entry")?,
            &path(p, "stock_image")?,
            Some(string(&evidence, "stock_sha256")?),
        )?;
        let hash = string(&evidence, "memlat_sha256")?;
        let short = hash.get(..12).ok_or("invalid image checksum")?;
        let destination = Path::new(optional(p, "image_directory", "/boot/EFI/LocalAI")).join(format!(
            "omarchy-memlat-{}-{short}.efi",
            string(&evidence, "kernel")?
        ));
        if destination.exists() {
            if sha256(&destination)? != hash {
                return err("existing rebuilt image hash mismatch");
            }
        } else {
            let temporary = destination.with_extension("staging");
            if temporary.exists() {
                return err("stale image staging file; inspect before retry");
            }
            let installed = (|| -> Result<()> {
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&temporary)?;
                std::io::copy(&mut fs::File::open(&image)?, &mut file)?;
                file.flush()?;
                file.sync_all()?;
                fs::rename(&temporary, &destination)?;
                Ok(())
            })();
            let _ = fs::remove_file(&temporary);
            installed?;
        }
        let name = format!("{}{})", menu::AUTO_PREFIX, string(&evidence, "kernel")?);
        let body = remove_owned_block(&before, p)?;
        if menu::entries(&body)?.contains_key(&name) {
            return err("new automatic entry name already exists");
        }
        let mut entry = format!(
            "\n/{name}\ncomment: {}\nprotocol: efi\npath: boot():/{}#{}\n",
            menu::AUTO_COMMENT,
            destination
                .strip_prefix(optional(p, "boot_root", "/boot"))?
                .display(),
            blake2b(&fs::read(&image)?)?
        );
        if let Some(cmdline) = &menu
            .get(string(p, "stock_entry")?)
            .ok_or("missing stock entry")?
            .cmdline
        {
            entry += &format!("cmdline: {cmdline}\n");
        }
        let after = format!("{}\n{entry}", body.trim_end_matches('\n'));
        let mut updated = p.clone();
        merge(&mut updated, &evidence)?;
        updated["memlat_entry"] = json!(name);
        updated["memlat_image"] = json!(destination);
        updated["automatic_entry_block"] = json!(entry);
        let record = json_text(&updated, true)? + "\n";
        replace(policy_path, &crate::read_text(policy_path)?, &record)?;
        replace(&config, &before, &after)?;
        let mut result = apply(&updated, "enable", host)?;
        match fs::remove_file(&pending) {
            Ok(()) => (),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(e) => return Err(e.into()),
        }
        merge(&mut result, &evidence)?;
        result["image"] = json!(destination);
        Ok(result)
    })();
    if let Err(error) = &result {
        if !dry_run {
            let latest: Value = serde_json::from_str(&crate::read_text(policy_path)?)?;
            let fallback = apply(&latest, "fallback", host);
            host.notify(&latest, &error.to_string());
            fallback?;
        }
    }
    result
}
fn merge(destination: &mut Value, source: &Value) -> Result<()> {
    for (key, value) in source.as_object().ok_or("invalid build evidence")? {
        destination[key] = value.clone();
    }
    Ok(())
}
// Python json.dumps(indent=2) escapes non-ASCII; preserve policy bytes too.
pub fn json_text(value: &Value, pretty: bool) -> Result<String> {
    let raw = if pretty {
        serde_json::to_string_pretty(value)?
    } else {
        serde_json::to_string(value)?
    };
    let mut out = String::new();
    let mut quoted = false;
    let mut escaped = false;
    for c in raw.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut buf = [0; 2];
            for u in c.encode_utf16(&mut buf) {
                out += &format!("\\u{u:04x}");
            }
        }
        if escaped {
            escaped = false;
            continue;
        }
        if quoted && c == '\\' {
            escaped = true;
            continue;
        }
        if c == '"' {
            quoted = !quoted;
        }
        if !pretty && !quoted && (c == ':' || c == ',') {
            out.push(' ');
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_hash_verification_on_private_fixture() {
        let tmp = crate::Temporary::directory("memlat-hash-").unwrap();
        let image = tmp.path.join("stock.efi");
        fs::write(&image, b"payload").unwrap();
        let text = format!(
            "/Stock\nprotocol: efi\npath: boot():/stock.efi#{}\n",
            "0".repeat(128)
        );
        let menu = menu::entries(&text).unwrap();
        assert_eq!(
            verify_at(&menu, "Stock", &image, None, &tmp.path)
                .unwrap_err()
                .to_string(),
            "Limine EFI hash mismatch: Stock"
        );
        let text = text.replace(&"0".repeat(128), &blake2b(b"payload").unwrap());
        verify_at(
            &menu::entries(&text).unwrap(),
            "Stock",
            &image,
            Some(&sha256(&image).unwrap()),
            &tmp.path,
        )
        .unwrap();
    }
}
