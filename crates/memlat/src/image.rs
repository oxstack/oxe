//! The unsigned PE32+ subset used by the original MEMLAT builder.
use crate::policy::{path, string};
use crate::{err, sha256, Result, Temporary};
use serde_json::{json, Value};
use std::io::Write;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
#[derive(Clone, Debug)]
pub struct Section {
    pub header: usize,
    pub name: Vec<u8>,
    pub virtual_size: usize,
    pub virtual_addr: usize,
    pub raw_size: usize,
    pub raw_offset: usize,
}
fn slice(data: &[u8], offset: usize, len: usize) -> Result<&[u8]> {
    data.get(offset..offset.checked_add(len).ok_or("truncated PE image")?)
        .ok_or_else(|| "truncated PE image".into())
}
fn u16le(data: &[u8], offset: usize) -> Result<usize> {
    Ok(u16::from_le_bytes(slice(data, offset, 2)?.try_into()?) as usize)
}
fn u32le(data: &[u8], offset: usize) -> Result<usize> {
    Ok(u32::from_le_bytes(slice(data, offset, 4)?.try_into()?) as usize)
}
pub fn sections(data: &[u8]) -> Result<Vec<Section>> {
    if !data.starts_with(b"MZ") {
        return err("not a PE image");
    }
    let pe = u32le(data, 0x3c)?;
    if data.get(pe..pe + 4) != Some(b"PE\0\0") {
        return err("bad PE signature");
    }
    let count = u16le(data, pe + 6)?;
    let size = u16le(data, pe + 20)?;
    let opt = pe + 24;
    if size < 152 || u16le(data, opt)? != 0x20b {
        return err("PE32+ optional header required");
    }
    if u32le(data, opt + 64)? != 0 || u32le(data, opt + 144)? != 0 || u32le(data, opt + 148)? != 0 {
        return err("signed/checksummed UKI unsupported; signing is not implemented");
    }
    let mut result = Vec::new();
    for i in 0..count {
        let h = opt + size + i * 40;
        slice(data, h, 40)?;
        let vs = u32le(data, h + 8)?;
        let va = u32le(data, h + 12)?;
        let rs = u32le(data, h + 16)?;
        let rp = u32le(data, h + 20)?;
        if rp.checked_add(rs).is_none_or(|end| end > data.len()) || vs > rs {
            return err("invalid PE section bounds");
        }
        let mut name = slice(data, h, 8)?.to_vec();
        while name.last() == Some(&0) {
            name.pop();
        }
        result.push(Section {
            header: h,
            name,
            virtual_size: vs,
            virtual_addr: va,
            raw_size: rs,
            raw_offset: rp,
        });
    }
    let table_end = opt + size + count * 40;
    let mut spans: Vec<_> = result
        .iter()
        .filter(|s| s.raw_size != 0)
        .map(|s| (s.raw_offset, s.raw_offset + s.raw_size))
        .collect();
    spans.sort();
    if spans.iter().any(|(a, _)| *a < table_end) || spans.windows(2).any(|s| s[0].1 > s[1].0) {
        return err("overlapping PE sections/headers");
    }
    Ok(result)
}
pub fn payload<'a>(data: &'a [u8], section: &Section) -> &'a [u8] {
    &data[section.raw_offset..section.raw_offset + section.virtual_size]
}
pub fn patch_uki(data: &[u8], stock: &[u8], changed: &[u8]) -> Result<Vec<u8>> {
    let sections = sections(data)?;
    let selected: Vec<_> = sections
        .iter()
        .filter(|s| s.name == b".dtbauto" && payload(data, s) == stock)
        .collect();
    if selected.len() != 1 {
        return err("expected exactly one .dtbauto matching the complete stock DTB");
    }
    let s = selected[0];
    if !stock.starts_with(b"\xd0\x0d\xfe\xed") || changed.get(..4) != stock.get(..4) {
        return err("DTB magic required");
    }
    let next = sections
        .iter()
        .filter(|q| q.virtual_addr > s.virtual_addr)
        .map(|q| q.virtual_addr)
        .min();
    if changed.len() > s.raw_size || next.is_some_and(|n| s.virtual_addr + changed.len() > n) {
        return err("changed DTB does not fit the existing section");
    }
    let mut out = data.to_vec();
    out[s.header + 8..s.header + 12].copy_from_slice(&u32::try_from(changed.len())?.to_le_bytes());
    out[s.raw_offset..s.raw_offset + s.raw_size].fill(0);
    out[s.raw_offset..s.raw_offset + changed.len()].copy_from_slice(changed);
    Ok(out)
}
fn secure_boot(root: &Path) -> String {
    let Ok(read) = fs::read_dir(root) else {
        return "unknown".into();
    };
    let files: Vec<_> = read
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("SecureBoot-"))
        .collect();
    if files.len() != 1 {
        return "unknown".into();
    }
    match fs::read(files[0].path()).ok().and_then(|v| v.get(4).copied()) {
        Some(0) => "disabled",
        Some(1) => "enabled",
        _ => "unknown",
    }
    .into()
}
fn tool(name: &str, kernel: &Path) -> PathBuf {
    for parent in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let path = parent.join(name);
        if path.is_file() {
            use std::os::unix::fs::PermissionsExt;
            if path.metadata().is_ok_and(|m| m.permissions().mode() & 0o111 != 0) {
                return path;
            }
        }
    }
    kernel.join("scripts/dtc").join(name)
}
fn run(command: &mut Command) -> Result<()> {
    let status = command.status()?;
    if !status.success() {
        return err(&format!(
            "Command {:?} returned non-zero exit status {}.",
            command,
            status.code().unwrap_or(-1)
        ));
    }
    Ok(())
}
pub fn build(p: &Value, output: &Path) -> Result<Value> {
    if secure_boot(Path::new("/sys/firmware/efi/efivars")) != "disabled" {
        return err("unsigned MEMLAT rebuild requires Secure Boot disabled");
    }
    let stock = path(p, "stock_image")?;
    let original = fs::read(&stock)?;
    let old = sections(&original)?;
    let uname: Vec<_> = old.iter().filter(|s| s.name == b".uname").collect();
    if uname.len() != 1 {
        return err("expected one kernel version in stock UKI");
    }
    let bytes = payload(&original, uname[0]);
    let bytes = &bytes[..bytes.iter().rposition(|b| *b != 0).map_or(0, |i| i + 1)];
    let version = std::str::from_utf8(bytes)?;
    if version.is_empty()
        || !version
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._+-".contains(&c))
    {
        return err("invalid kernel version");
    }
    let modules = Path::new("/usr/lib/modules").join(version);
    let kernel = modules.join("build");
    if !modules.is_dir()
        || !fs::read_to_string(kernel.join(".config"))?.contains("CONFIG_SCMI_QCOM_MEMLAT_DEVFREQ=y")
    {
        return err("matching kernel modules/headers or built-in MEMLAT driver missing");
    }
    let tmp = Temporary::directory("memlat-")?;
    let stock_dtb = tmp.path.join("stock.dtb");
    fs::copy(path(p, "stock_dtb")?, &stock_dtb)?;
    let compiled = tmp.path.join("overlay.dtbo");
    let changed = tmp.path.join("changed.dtb");
    run(Command::new(tool("dtc", &kernel))
        .args(["-@", "-I", "dts", "-O", "dtb", "-o"])
        .arg(&compiled)
        .arg(string(p, "overlay")?))?;
    run(Command::new(tool("fdtoverlay", &kernel))
        .arg("-i")
        .arg(&stock_dtb)
        .arg("-o")
        .arg(&changed)
        .arg(&compiled))?;
    let patched = patch_uki(&original, &fs::read(&stock_dtb)?, &fs::read(&changed)?)?;
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(output)?;
    file.write_all(&patched)?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"output":output,"source_sha256":sha256(&stock)?,"output_sha256":sha256(output)?,"stock_dtb_sha256":sha256(&stock_dtb)?,"changed_dtb_sha256":sha256(&changed)?})
        )?
    );
    let new = sections(&patched)?;
    let different: Vec<_> = old
        .iter()
        .zip(&new)
        .filter(|(a, b)| payload(&original, a) != payload(&patched, b))
        .collect();
    if different.len() != 1 || different[0].0.name != b".dtbauto" {
        return err("expected exactly the Yoga DTB section to change");
    }
    Ok(
        json!({"kernel":version,"modules":modules,"stock_sha256":sha256(&stock)?,"memlat_sha256":sha256(output)?,"stock_dtb_sha256":sha256(&path(p,"stock_dtb")?)?}),
    )
}
