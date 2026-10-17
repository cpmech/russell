//! Every path in the workspace must be checkable out on Windows.
//!
//! Git refuses to materialize a path Windows cannot represent, and it does so at
//! checkout time -- before anything is compiled. A file called `aux.rs` is a
//! reserved DOS device name, so it failed the Windows job with
//! `error: invalid path 'russell_lab/src/native/lapack/aux.rs'` while every
//! other platform stayed green. The module is `auxiliary.rs` now, and this test
//! keeps the next such name from reaching CI.
//!
//! Rules checked, all of which Windows enforces: reserved device names (with or
//! without an extension), the characters `< > : " | ? *`, control characters,
//! and a name ending in a space or a period.
//!
//! Scope: every file and directory under the workspace root, excluding build
//! output and `.git`.

use std::fs;
use std::path::Path;

/// DOS device names. Windows resolves them with any extension, so `con.rs`,
/// `aux.md` and a bare `nul` are all unusable.
const RESERVED: &[&str] = &[
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9", "lpt1", "lpt2",
    "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// Characters Windows rejects in a name; `/` and `\` are separators, not names.
const ILLEGAL: &[char] = &['<', '>', ':', '"', '|', '?', '*'];

/// Returns why `name` cannot exist on Windows, or `None` when it can.
fn problem(name: &str) -> Option<String> {
    if let Some(c) = name.chars().find(|c| ILLEGAL.contains(c) || c.is_control()) {
        return Some(format!("illegal character {:?}", c));
    }
    if name.ends_with(' ') || name.ends_with('.') {
        return Some("ends with a space or a period".to_string());
    }
    // The device test applies to the part before the first dot.
    let base = name.split('.').next().unwrap_or("").to_lowercase();
    if RESERVED.contains(&base.as_str()) {
        return Some(format!("reserved device name {:?}", base));
    }
    None
}

fn scan(dir: &Path, offenders: &mut Vec<String>) {
    for entry in fs::read_dir(dir).expect("read_dir") {
        let path = entry.expect("entry").path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name == "target" || name == ".git" || name == "node_modules" {
            continue;
        }
        if let Some(why) = problem(name) {
            offenders.push(format!("{}: {}", path.display(), why));
        }
        if path.is_dir() {
            scan(&path, offenders);
        }
    }
}

#[test]
fn every_path_can_be_checked_out_on_windows() {
    // The workspace root is one level above this crate.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("workspace root");
    let mut offenders = Vec::new();
    scan(root, &mut offenders);
    assert!(
        offenders.is_empty(),
        "{} path(s) that Windows cannot check out -- CI runs on Windows, so git fails there \
         before compiling:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}
