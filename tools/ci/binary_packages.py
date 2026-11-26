#!/usr/bin/env python3
"""Print the cargo packages that own the given workspace binary targets.

usage: binary_packages.py BIN [BIN...]

Reads `cargo metadata --no-deps --locked` for the workspace in the current directory. Each binary
name must belong to exactly one workspace package; anything else is an error, so a packaging
script cannot ship a binary without the licence texts of the crates linked into it. Packages are
printed once each, in the order their binaries were asked for.
"""
import json
import subprocess
import sys


def owners(packages, bins):
    found = {}
    for package in packages:
        for target in package.get('targets', []):
            if 'bin' in target.get('kind', []) and target['name'] in bins:
                found.setdefault(target['name'], []).append(package['name'])
    return found


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    wanted = argv[1:]
    metadata = json.loads(subprocess.run(
        ['cargo', 'metadata', '--no-deps', '--locked', '--format-version', '1'],
        check=True, capture_output=True, text=True).stdout)
    found = owners(metadata['packages'], set(wanted))
    result, bad = [], []
    for name in wanted:
        packages = found.get(name, [])
        if len(packages) != 1:
            bad.append(f'{name}: {len(packages)} owning packages {sorted(packages)}')
        elif packages[0] not in result:
            result.append(packages[0])
    if bad:
        raise SystemExit('binary_packages: ' + '; '.join(bad))
    print('\n'.join(result))


if __name__ == '__main__':
    main(sys.argv)
