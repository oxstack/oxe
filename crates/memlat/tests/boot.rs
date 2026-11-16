//! Ports of all 19 test_memlat_boot.py cases; synthetic filesystem only.
use oxc_memlat::{
    menu,
    policy::{self, Host},
    sha256, Result, Temporary,
};
use serde_json::{json, Value};
use std::{
    cell::Cell,
    fs,
    path::{Path, PathBuf},
};
const MENU:&str="default_entry: 2\n/+Omarchy\n  //linux-omarchy\n  protocol: efi\n  path: boot():/EFI/Linux/stock.efi\n  //Snapshots\n    ///Old\n    protocol: efi\n    path: boot():/old.efi\n/Local AI memlat test (7.3.0-40)\nprotocol: efi\npath: boot():/EFI/LocalAI/test.efi\n";
const OWN: &str = "Local AI memlat test (7.3.0-40)";
const STALE:&str="\n/Local AI memlat (newer-kernel)\ncomment: Auto-rebuilt Yoga MEMLAT; stock fallback Omarchy/linux-omarchy\nprotocol: efi\npath: boot():/omarchy-memlat-newer-kernel-abc.efi\n";
#[derive(Default)]
struct Fake {
    bad_stock: Cell<bool>,
    bad_image: Cell<bool>,
    fail_build: Cell<bool>,
    builds: Cell<usize>,
    verifications: Cell<usize>,
    notifications: Cell<usize>,
}
impl Host for Fake {
    fn verify(&self, _: &menu::Menu, name: &str, _: &Path, expected: Option<&str>) -> Result<()> {
        self.verifications.set(self.verifications.get() + 1);
        if self.bad_image.get() && name == OWN {
            return oxc_memlat::err("bad image");
        }
        if self.bad_stock.get() && name == "Omarchy/linux-omarchy" && expected.is_some() {
            return oxc_memlat::err("changed stock");
        }
        Ok(())
    }
    fn build(&self, p: &Value, out: &Path) -> Result<Value> {
        self.builds.set(self.builds.get() + 1);
        if self.fail_build.get() {
            return oxc_memlat::err("DTB mismatch");
        }
        fs::write(out, b"patched kernel fixture")?;
        Ok(
            json!({"kernel":"new-kernel","modules":p["modules"],"stock_sha256":p["stock_sha256"],"memlat_sha256":sha256(out)?,"stock_dtb_sha256":"fixture"}),
        )
    }
    fn notify(&self, _: &Value, _: &str) {
        self.notifications.set(self.notifications.get() + 1);
    }
}
struct Fixture {
    _tmp: Temporary,
    p: Value,
    policy: PathBuf,
    host: Fake,
}
impl Fixture {
    fn new() -> Self {
        let tmp = Temporary::directory("memlat-test-").unwrap();
        let root = &tmp.path;
        fs::write(root.join("limine.conf"), MENU).unwrap();
        fs::write(root.join("stock.efi"), b"qualified stock").unwrap();
        fs::create_dir(root.join("modules")).unwrap();
        let p = json!({"config":root.join("limine.conf"),"stock_entry":"Omarchy/linux-omarchy","memlat_entry":OWN,"stock_image":root.join("stock.efi"),"stock_sha256":sha256(&root.join("stock.efi")).unwrap(),"modules":root.join("modules"),"memlat_image":"/boot/EFI/LocalAI/test.efi","memlat_sha256":"known","pending":root.join("pending"),"image_directory":root,"boot_root":root});
        let policy = root.join("policy");
        fs::write(&policy, p.to_string()).unwrap();
        Self {
            _tmp: tmp,
            p,
            policy,
            host: Fake::default(),
        }
    }
    fn apply(&self, a: &str) -> Value {
        policy::apply(&self.p, a, &self.host).unwrap()
    }
    fn text(&self) -> String {
        fs::read_to_string(policy::path(&self.p, "config").unwrap()).unwrap()
    }
    fn write(&self, s: &str) {
        fs::write(policy::path(&self.p, "config").unwrap(), s).unwrap();
    }
    fn save(&self) {
        fs::write(&self.policy, self.p.to_string()).unwrap();
    }
    fn stale(&self) {
        self.write(
            &(MENU.replace(
                "default_entry: 2",
                "default_entry: Local AI memlat (newer-kernel)",
            ) + STALE),
        );
    }
}
#[test]
fn numeric_and_named_selection_survive_snapshot_renumbering() {
    assert_eq!(
        menu::selected(MENU, &menu::entries(MENU).unwrap()).unwrap(),
        "Omarchy/linux-omarchy"
    );
    let text = MENU
        .replace("default_entry: 2", &format!("default_entry: {OWN}"))
        .replace(
            "/Local AI memlat",
            "/Another entry\nprotocol: efi\npath: boot():/other\n/Local AI memlat",
        );
    assert_eq!(
        menu::selected(&text, &menu::entries(&text).unwrap()).unwrap(),
        OWN
    );
}
#[test]
fn enable_fallback_idempotence_and_preservation() {
    let f = Fixture::new();
    assert_eq!(f.apply("enable")["changed"], true);
    assert_eq!(f.apply("enable")["changed"], false);
    assert_eq!(f.apply("fallback")["default"], "Omarchy/linux-omarchy");
    assert_eq!(
        f.text(),
        MENU.replace("default_entry: 2", "default_entry: Omarchy/linux-omarchy")
    );
    assert_eq!(f.apply("fallback")["changed"], false);
}
#[test]
fn missing_modules_and_changed_stock_reject_enable() {
    for failure in ["modules", "hash"] {
        let f = Fixture::new();
        if failure == "modules" {
            fs::remove_dir(policy::path(&f.p, "modules").unwrap()).unwrap();
        } else {
            f.host.bad_stock.set(true);
        }
        assert!(policy::apply(&f.p, "enable", &f.host).is_err());
        assert_eq!(f.text(), MENU);
    }
}
#[test]
fn check_falls_back_if_stock_changed_or_modules_removed() {
    for failure in ["stock", "modules", "image"] {
        let f = Fixture::new();
        f.apply("enable");
        match failure {
            "stock" => fs::write(policy::path(&f.p, "stock_image").unwrap(), b"new kernel").unwrap(),
            "modules" => fs::remove_dir(policy::path(&f.p, "modules").unwrap()).unwrap(),
            _ => f.host.bad_image.set(true),
        }
        assert_eq!(f.apply("check")["default"], "Omarchy/linux-omarchy");
    }
}
#[test]
fn other_user_default_is_not_overridden_by_update_hooks() {
    let f = Fixture::new();
    f.host.bad_stock.set(true);
    f.host.bad_image.set(true);
    for a in ["check", "fallback"] {
        assert_eq!(f.apply(a)["changed"], false);
    }
    assert_eq!(f.text(), MENU);
    assert_eq!(
        f.host.verifications.get(),
        0,
        "unrelated defaults must not trigger EFI verification"
    );
}
#[test]
fn ambiguity_and_remember_last_rejected() {
    for text in [
        format!("{MENU}\ndefault_entry: 1\n"),
        format!("{MENU}\nremember_last_entry: yes\n"),
        MENU.replace("/+Omarchy", "/+Omarchy\ninclude: another"),
    ] {
        assert!(menu::entries(&text)
            .and_then(|m| menu::selected(&text, &m))
            .is_err());
    }
}
#[test]
fn unbootable_fallback_and_bad_hash_rejected() {
    let mut m = menu::entries(MENU).unwrap();
    for name in ["Omarchy", "missing"] {
        assert!(policy::verify_entry(&m, name, Path::new("/boot/no"), None).is_err());
    }
    m.get_mut("Omarchy/linux-omarchy").unwrap().path =
        Some(format!("boot():/EFI/Linux/stock.efi#{}", "0".repeat(128)));
    assert!(policy::verify_entry(
        &m,
        "Omarchy/linux-omarchy",
        Path::new("/tmp/memlat-non-boot-fixture"),
        None
    )
    .is_err());
    assert_ne!(policy::blake2b(b"payload").unwrap(), "0".repeat(128));
}
#[test]
fn compare_and_replace_refuses_intervening_changes() {
    let tmp = Temporary::directory("memlat-test-").unwrap();
    let path = tmp.path.join("config");
    fs::write(&path, "user edit").unwrap();
    assert!(oxc_memlat::replace(&path, "old", "new").is_err());
    assert_eq!(fs::read_to_string(path).unwrap(), "user edit");
}
#[test]
fn update_prepares_marker_before_fallback_and_repeated_calls_keep_it() {
    let f = Fixture::new();
    f.apply("enable");
    assert_eq!(f.apply("prepare-update")["default"], "Omarchy/linux-omarchy");
    let pending = policy::path(&f.p, "pending").unwrap();
    assert!(pending.exists());
    assert_eq!(f.apply("prepare-update")["changed"], false);
    assert!(pending.exists());
}
#[test]
fn dry_run_never_changes_menu_policy_or_pending() {
    let f = Fixture::new();
    let before = fs::read(&f.policy).unwrap();
    let result = policy::rebuild(&f.policy, &f.p, true, false, &f.host).unwrap();
    assert_eq!(result["passed"], true);
    assert_eq!(result["changed"], false);
    assert_eq!(f.text(), MENU);
    assert_eq!(fs::read(&f.policy).unwrap(), before);
    assert!(!policy::path(&f.p, "pending").unwrap().exists());
}
#[test]
fn update_rebuild_is_repeatable_and_preserves_original_test_entry() {
    let mut f = Fixture::new();
    f.apply("enable");
    for _ in 0..2 {
        f.apply("prepare-update");
        let result = policy::rebuild(&f.policy, &f.p, false, false, &f.host).unwrap();
        assert_eq!(result["default"], "Local AI memlat (new-kernel)");
        assert!(!policy::path(&f.p, "pending").unwrap().exists());
        f.p = serde_json::from_slice(&fs::read(&f.policy).unwrap()).unwrap();
        let text = f.text();
        let m = menu::entries(&text).unwrap();
        assert!(m.contains_key(OWN));
        assert!(m.contains_key(f.p["memlat_entry"].as_str().unwrap()));
        assert_eq!(menu::selected(&text, &m).unwrap(), f.p["memlat_entry"]);
    }
}
#[test]
fn build_failure_keeps_stock_and_notifies() {
    let f = Fixture::new();
    f.apply("enable");
    f.apply("prepare-update");
    f.host.fail_build.set(true);
    assert!(policy::rebuild(&f.policy, &f.p, false, false, &f.host).is_err());
    assert_eq!(
        menu::selected(&f.text(), &menu::entries(&f.text()).unwrap()).unwrap(),
        "Omarchy/linux-omarchy"
    );
    assert!(policy::path(&f.p, "pending").unwrap().exists());
    assert_eq!(f.host.notifications.get(), 1);
}
#[test]
fn no_pending_means_no_build_or_default_change() {
    let f = Fixture::new();
    let result = policy::rebuild(
        &f.policy.with_file_name("no-policy-needed"),
        &f.p,
        false,
        false,
        &f.host,
    )
    .unwrap();
    assert_eq!(result["changed"], false);
    assert_eq!(f.host.builds.get(), 0);
    assert_eq!(f.text(), MENU);
}
#[test]
fn rebuild_preserves_stock_outer_command_line() {
    let f = Fixture::new();
    f.write(&MENU.replace(
        "path: boot():/EFI/Linux/stock.efi",
        "path: boot():/EFI/Linux/stock.efi\n  cmdline: root=fixture quiet extra=outer",
    ));
    policy::rebuild(&f.policy, &f.p, false, true, &f.host).unwrap();
    let m = menu::entries(&f.text()).unwrap();
    assert_eq!(
        m["Local AI memlat (new-kernel)"].cmdline.as_deref(),
        Some("root=fixture quiet extra=outer")
    );
}
#[test]
fn generator_formatting_is_accepted_but_semantic_edits_are_not() {
    let block = "\n/Automatic\ncomment: owned\nprotocol: efi\npath: boot():/owned\n";
    let normalized = "\n/Automatic\n  path: boot():/owned\n  protocol: efi\n  comment: owned\n";
    let p = json!({"memlat_entry":"Automatic","automatic_entry_block":block});
    let text = format!("{MENU}{normalized}/Unrelated\nprotocol: efi\npath: boot():/other\n");
    assert!(!policy::remove_owned_block(&text, &p)
        .unwrap()
        .contains("/Automatic"));
    for changed in [
        text.replace("/owned", "/different"),
        text.replace("  comment: owned", "  cmdline: unexpected"),
    ] {
        assert!(policy::remove_owned_block(&changed, &p).is_err());
    }
}
#[test]
fn update_falls_back_from_auto_entry_unknown_to_restored_policy() {
    for a in ["prepare-update", "fallback", "check"] {
        let f = Fixture::new();
        f.stale();
        assert_eq!(f.apply(a)["default"], "Omarchy/linux-omarchy");
        assert_eq!(
            menu::selected(&f.text(), &menu::entries(&f.text()).unwrap()).unwrap(),
            "Omarchy/linux-omarchy"
        );
    }
}
#[test]
fn lookalike_user_entry_is_not_adopted() {
    for block in [
        STALE.replace("comment: Auto-rebuilt", "comment: Hand-made"),
        STALE.replace("boot():/omarchy-memlat-", "boot():/elsewhere/omarchy-memlat-"),
    ] {
        let f = Fixture::new();
        f.write(
            &(MENU.replace(
                "default_entry: 2",
                "default_entry: Local AI memlat (newer-kernel)",
            ) + &block),
        );
        assert_eq!(f.apply("prepare-update")["changed"], false);
    }
}
#[test]
fn rebuild_replaces_auto_entry_unknown_to_restored_policy() {
    let mut f = Fixture::new();
    f.stale();
    f.p["automatic_entry_block"]=json!("\n/Local AI memlat (old-kernel)\ncomment: Auto-rebuilt Yoga MEMLAT; stock fallback Omarchy/linux-omarchy\nprotocol: efi\npath: boot():/omarchy-memlat-old-kernel-def.efi\n");
    f.p["memlat_entry"] = json!("Local AI memlat (old-kernel)");
    f.save();
    f.apply("prepare-update");
    let result = policy::rebuild(&f.policy, &f.p, false, false, &f.host).unwrap();
    assert_eq!(result["default"], "Local AI memlat (new-kernel)");
    let m = menu::entries(&f.text()).unwrap();
    assert!(!m.contains_key("Local AI memlat (newer-kernel)"));
    assert!(m.contains_key(OWN));
}
#[test]
fn empty_command_line_does_not_consume_the_next_field() {
    let text = MENU.replacen("  protocol: efi", "  cmdline:\n  protocol: efi", 1);
    assert_eq!(
        menu::entries(&text).unwrap()["Omarchy/linux-omarchy"]
            .cmdline
            .as_deref(),
        Some("")
    );
}

#[test]
fn pending_marker_failure_still_selects_verified_stock() {
    let mut f = Fixture::new();
    f.apply("enable");
    let blocked = f.policy.with_file_name("not-a-directory");
    fs::write(&blocked, b"fixture blocker").unwrap();
    f.p["pending"] = json!(blocked.join("pending"));
    assert!(policy::apply(&f.p, "prepare-update", &f.host).is_err());
    let text = f.text();
    assert_eq!(
        menu::selected(&text, &menu::entries(&text).unwrap()).unwrap(),
        "Omarchy/linux-omarchy"
    );
}
