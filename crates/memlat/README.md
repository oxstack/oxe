# MEMLAT boot policy tool

`local-ai-memlat-boot` replaces `oxc/tools/memlat_boot.py` for #785. It accepts
`enable`, `fallback`, `check`, `prepare-update`, and `rebuild`, with the existing
`--policy`, `--no-lock`, `--dry-run`, and `--force` options. The default policy is
`/etc/local-ai/memlat-boot.json`; the boot mutex remains
`/run/lock/boot-partition.lock`. No command reboots the machine.

The release stage installs the native executable at
`/usr/bin/local-ai-memlat-boot`; the hook templates use the same path.
It does not install a policy,
change an ESP, enable MEMLAT, or install new hooks. Existing approved policy
files continue to work; the legacy `builder` field is retained but no Python
module is imported. Rebuild uses native unsigned PE32+ validation/patching and
the same external `dtc` and `fdtoverlay` tools. BLAKE2b checksums use coreutils
`b2sum`; SHA-256 uses the workspace's existing `sha2` crate. Missing tools refuse
rebuild while retaining STOCK; no runtime Python dependency is introduced.

The 19 named Rust behavior tests correspond to the 19 legacy Python tests.
The differential suite runs the actual Rust command over 28 scenarios and
52 steps and compares its exit status, selected output and ESP file digests
with the captured [boot oracle](tests/boot-oracle.json). The test tooling uses
Python to prepare synthetic fixtures and compares results in mandatory
bubblewrap with private `/boot`, `/sys`, `/usr/lib/modules`, `/run/lock`, and HOME,
with no network. DTC commands use synthetic executable fixtures. The installed
Rust helper requires no Python.

Boot-default tooling requires the orchestrator's source review before merge.
Laptop installation and hardware validation require the owner's explicit approval.

One deliberate failure-path correction follows #785's fail-safe requirement:
`prepare-update` commits the verified STOCK default before writing its pending
marker. Python writes the marker first, which can leave MEMLAT selected when
that write fails. A regression test fails with that ordering and passes with
the Rust ordering. Ambiguous menus or an unverifiable STOCK image still refuse
without guessing; the existing pre-transaction hook's `AbortOnFail` prevents
the kernel update. This difference requires explicit orchestrator review.

Existing manually installed hooks that still reference the old
`/usr/local/libexec` path must be updated to the supplied templates during the
owner-approved laptop installation. The package never overwrites a manually
installed file under `/usr/local` and never changes the ESP during installation.
