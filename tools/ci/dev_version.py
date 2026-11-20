"""Dev build version: a pre-release of the NEXT version after the latest stable release tag.

`v0.2.0` released -> dev builds are `v0.2.1-dev.<N>+<sha7>`, N = commits since the stable tag.
Semver compares the numeric pre-release N numerically (dev.9 < dev.10) and ignores the +sha
build metadata. Without any stable tag the workspace version is the base and N counts from the root.
Archive file names carry the same version with '+' written as '.' (see file_name_version).
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


def bump(tags, workspace_version, sha7, count):
    latest = latest_stable(tags)
    if latest:
        major, minor, patch = latest
        return f'v{major}.{minor}.{patch + 1}-dev.{count}+{sha7}'
    return f'v{workspace_version}-dev.{count}+{sha7}'


def file_name_version(version):
    """Version as used inside file names: '+' is not safe in release asset names, so it becomes '.'."""
    return version.replace('+', '.')


def commit_count(root, tags):
    """Commits since the highest stable tag; from the root when there is none."""
    latest = latest_stable(tags)
    spec = 'HEAD' if latest is None else 'v%d.%d.%d..HEAD' % latest
    out = subprocess.run(['git', '-C', str(root), 'rev-list', '--count', spec],
                         check=True, capture_output=True, text=True).stdout.strip()
    return int(out)


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
        raise SystemExit('usage: dev_version.py <commit-sha> [--file-name]')
    root = Path(__file__).resolve().parents[2]
    tags = subprocess.run(['git', '-C', str(root), 'tag', '--list', 'v[0-9]*.[0-9]*.[0-9]*'],
                          check=True, capture_output=True, text=True).stdout.split()
    version = bump(tags, workspace_version(root), sha7, commit_count(root, tags))
    if '--file-name' in argv[2:]:
        version = file_name_version(version)
    print(version)


if __name__ == '__main__':
    main(sys.argv)
