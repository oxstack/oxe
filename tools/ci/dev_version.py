"""Dev build version: a pre-release of the NEXT version after the latest stable release tag.

`v0.2.0` released -> dev builds are `v0.2.1-dev.<sha7>` (semver sorts that after v0.2.0).
Without any stable tag the workspace version is used as the base.
"""
import re
import subprocess
import sys
from pathlib import Path

STABLE = re.compile(r'^v(\d+)\.(\d+)\.(\d+)$')
WORKSPACE = re.compile(r'^\d+\.\d+\.\d+')


def latest_stable(tags):
    """Highest vX.Y.Z tag by semver; pre-release and unrelated tags are ignored."""
    found = [tuple(int(part) for part in match.groups())
             for match in map(STABLE.match, tags) if match]
    return max(found) if found else None


def bump(tags, workspace_version, sha7):
    latest = latest_stable(tags)
    if latest:
        major, minor, patch = latest
        return f'v{major}.{minor}.{patch + 1}-dev.{sha7}'
    return f'v{workspace_version}-dev.{sha7}'


def workspace_version(root):
    text = (Path(root) / 'oxa' / 'Cargo.toml').read_text()
    section = text.split('[workspace.package]', 1)[1].split('\n[', 1)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', section, re.M)
    if not match or not WORKSPACE.match(match.group(1)):
        raise SystemExit('workspace version not found')
    return match.group(1)


def main(argv):
    sha7 = argv[1][:7]
    if not re.fullmatch(r'[0-9a-f]{7}', sha7):
        raise SystemExit('usage: dev_version.py <commit-sha>')
    root = Path(__file__).resolve().parents[2]
    tags = subprocess.run(['git', '-C', str(root), 'tag', '--list', 'v[0-9]*.[0-9]*.[0-9]*'],
                          check=True, capture_output=True, text=True).stdout.split()
    print(bump(tags, workspace_version(root), sha7))


if __name__ == '__main__':
    main(sys.argv)
