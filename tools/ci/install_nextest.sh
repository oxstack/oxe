#!/usr/bin/env bash
# Official nextest release; checksums reviewed with this version, never fetched at runtime.
set -euo pipefail
version=0.9.152
case "$(uname -m)" in
  x86_64) target=x86_64-unknown-linux-gnu; digest=d039b43c3e0958441b6ad0a39c8884d018b3d2e8fb9928f209074b1100b5e17a ;;
  aarch64) target=aarch64-unknown-linux-gnu; digest=ca1f81a5b86533eb45ab97e0064aa88e925e53aab9a698ca6331a7cad1f85f07 ;;
  *) echo 'unsupported nextest host' >&2; exit 1 ;;
esac
destination=${1:?usage: install_nextest.sh DESTINATION}
mkdir -p "$destination"
archive=$(mktemp)
trap 'rm -f "$archive"' EXIT
curl --proto '=https' --tlsv1.2 -fsSL -o "$archive" \
  "https://github.com/nextest-rs/nextest/releases/download/cargo-nextest-$version/cargo-nextest-$version-$target.tar.gz"
echo "$digest  $archive" | sha256sum -c -
tar -xzf "$archive" -C "$destination" cargo-nextest
"$destination/cargo-nextest" nextest --version
if [ -n "${GITHUB_PATH:-}" ]; then echo "$destination" >> "$GITHUB_PATH"; fi
