#!/usr/bin/env python3
"""Fail on Hexagon/QAIRT SDK material or obvious secrets in the tracked tree.

Port of the ox-llm monorepo's `scans` job (checks.yml) with the paths of this repository: the
source is at the root (`crates/`, `npu/`), not under `engine/` and `oxa/`.

Two independent checks over `git ls-files`:
  * secret scan: well-known credential shapes (GitHub, Hugging Face, OpenAI, AWS, Slack, private
    keys) in tracked text files;
  * SDK-material check: headers, IDL, stubs, skeletons, shared objects, ELF objects and
    Hexagon/QAIRT directory or file names.

The allow-lists below are exact paths with a reason. Nothing is added "because CI was red": a
match here means either a leak or a fixture whose fake value must be reviewed in the diff.

Usage: python3 -I tools/ci/scan_tree.py   (from the repository root; exit 1 on any finding)
"""
import re
import subprocess
import sys
from pathlib import Path

SECRET = re.compile(
    r'(ghp_[A-Za-z0-9]{30,}'
    r'|gho_[A-Za-z0-9]{30,}'
    r'|github_pat_[A-Za-z0-9_]{40,}'
    r'|hf_[A-Za-z0-9]{30,}'
    r'|sk-[A-Za-z0-9_-]{32,}'
    r'|AKIA[0-9A-Z]{16}'
    r'|xox[baprs]-[A-Za-z0-9-]{10,}'
    r'|-----BEGIN [A-Z ]*PRIVATE KEY-----)'
)

# Exact tracked paths allowed to contain a secret-shaped string, with the reason. A realistic
# credential is never acceptable here; only a fixture whose value is obviously synthetic and
# whose test asserts on it.
SECRET_ALLOWLIST: dict[str, str] = {}

# Files that are allowed to match the SDK file-name pattern below. Documentation about the ban
# is prose, not a file with one of these names, so no entry is needed today.
SDK_ALLOWLIST: dict[str, str] = {}

SDK_NAME = re.compile(
    r'^(HAP_.*\.h'
    r'|qurt.*\.h'
    r'|dspqueue\.h'
    r'|remote(?:64)?\.h'
    r'|AEEStd.*\.h'
    r'|AEEatomic\.h'
    r'|rpcmem\.h'
    r'|adsprpc.*\.h'
    r'|hexagon_.*\.h'
    r'|hvx_hexagon_protos\.h'
    r'|Qnn[A-Za-z0-9_]*\.h'
    r'|qnn_[A-Za-z0-9_]*\.h'
    r'|libQnn.*\.(?:a|so)(?:\..*)?'
    r'|.*\.idl'
    r'|.*skel.*\.so.*'
    r'|.*\.so(?:\..*)?'
    r'|.*_stub\.c'
    r'|.*_skel\.c)$',
    re.I,
)

# Directory components that only a vendored SDK creates.
SDK_DIRS = {'ggml-hexagon', 'hexagon-sdk', 'hexagon_sdk', 'qairt'}

TEXT_SNIFF = 8000  # bytes read from a candidate before the text checks


def tracked_files(root: Path) -> list[Path]:
    """Tracked paths when this is a git checkout, otherwise every file in the tree."""
    if (root / '.git').exists():
        out = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root)
        return [root / name for name in out.decode().split('\0') if name]
    files = []
    for path in root.rglob('*'):
        rel = path.relative_to(root)
        if path.is_file() and rel.parts[0] not in ('.git', 'target'):
            files.append(path)
    return files


def is_binary(data: bytes) -> bool:
    return b'\0' in data


def secret_hits(root: Path, files: list[Path]) -> list[str]:
    hits = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        if rel in SECRET_ALLOWLIST:
            continue
        try:
            data = path.read_bytes()[:TEXT_SNIFF]
        except OSError as error:
            hits.append(f'{rel}: unreadable ({error})')
            continue
        if is_binary(data):
            continue
        text = data.decode('utf-8', 'replace')
        for match in SECRET.finditer(text):
            line = text.count('\n', 0, match.start()) + 1
            hits.append(f'{rel}:{line}: secret-like {match.group(0)[:12]}...')
    return hits


def sdk_hits(root: Path, files: list[Path]) -> list[str]:
    hits = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        if rel in SDK_ALLOWLIST:
            continue
        parts = rel.split('/')
        for part in parts[:-1]:
            if part.lower() in SDK_DIRS:
                hits.append(f'{rel}: inside SDK directory {part}/')
                break
        if SDK_NAME.match(path.name):
            hits.append(f'{rel}: SDK-like file name')
            continue
        try:
            head = path.read_bytes()[:4]
        except OSError as error:
            hits.append(f'{rel}: unreadable ({error})')
            continue
        if head == b'\x7fELF':
            hits.append(f'{rel}: ELF object')
    return hits


def main() -> int:
    root = Path.cwd().resolve()
    files = tracked_files(root)
    hits = sorted({*secret_hits(root, files), *sdk_hits(root, files)})
    if hits:
        print('\n'.join(hits))
        print(f'\n{len(hits)} finding(s) in {len(files)} tracked files; '
              'remove them or add a reviewed allow-list entry', file=sys.stderr)
        return 1
    print(f'no secrets or SDK material in {len(files)} tracked files')
    return 0


if __name__ == '__main__':
    sys.exit(main())
