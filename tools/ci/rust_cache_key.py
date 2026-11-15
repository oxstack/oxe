#!/usr/bin/env python3
"""Content/compiler cache keys using only GitHub's permitted actions/cache backend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def keys(root, files, variant, compiler, flags):
    def digest(values):
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    dependencies = []
    source = []
    for name in sorted(files):
        path = root / name
        if not path.is_file():
            continue
        entry = (name, hashlib.sha256(path.read_bytes()).hexdigest())
        source.append(entry)
        if (path.name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain', 'rust-toolchain.toml')
                or '.cargo' in path.parts):
            dependencies.append(entry)
    prefix = f'rust-v1-{variant}-{digest([compiler, flags, dependencies])}-'
    return prefix + digest(source), prefix


def source_files(root, workspaces):
    if (root / '.git').exists():
        return subprocess.check_output(['git', 'ls-files', '-z', '--', *workspaces,
            '.cargo', 'rust-toolchain', 'rust-toolchain.toml'], cwd=root).decode().split('\0')[:-1]
    # Minimal Fedora images have no git at checkout time: checkout uses a source archive.
    # That archive contains tracked source only; build outputs must not enter the key.
    files = []
    for workspace in [*workspaces, '.cargo']:
        base = root / workspace
        for directory, dirs, names in os.walk(base):
            dirs[:] = [name for name in dirs if name not in ('.git', '__pycache__')
                       and Path(directory) / name != base / 'target']
            files.extend(str((Path(directory) / name).relative_to(root)) for name in names)
    files.extend(name for name in ('rust-toolchain', 'rust-toolchain.toml') if (root / name).is_file())
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True)
    parser.add_argument('workspaces', nargs='+')
    args = parser.parse_args()
    if not re.fullmatch('[a-zA-Z0-9_-]+', args.variant):
        parser.error('variant must contain only letters, digits, underscore or hyphen')
    files = source_files(Path.cwd(), args.workspaces)
    compiler = {'platform': Path('/etc/os-release').read_text()}
    for command in (['rustc', '--version', '--verbose'], ['cc', '--version'], ['c++', '--version']):
        compiler[command[0]] = subprocess.check_output(command, text=True).strip()
    names = ('RUSTFLAGS', 'RUSTDOCFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_TARGET',
             'CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'RUSTC', 'RUSTDOC',
             'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER')
    flags = {name: value for name, value in os.environ.items()
             if name in names or name.startswith(('CARGO_PROFILE_', 'CARGO_TARGET_'))}
    key, prefix = keys(Path.cwd(), files, args.variant, compiler, flags)
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'key={key}\nprefix={prefix}\n')


if __name__ == '__main__':
    main()
