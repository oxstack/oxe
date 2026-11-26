#!/usr/bin/env bash
# Package release binaries into one tarball with a usr/bin/ tree, the licences, the notices and a
# README.txt.
#   tools/ci/package.sh NAME VERSION DIST_DIR TARGET_DIR BIN [BIN...]
# Writes DIST_DIR/NAME-VERSION-linux-aarch64.tar.gz and its .sha256.
# The licence scan covers the workspace package that owns each binary (binary_packages.py).
# Used by .github/workflows/release-build.yml.
set -euo pipefail
NAME=$1 VERSION=$2 DIST=$3 TARGET=$4
shift 4
[ $# -ge 1 ] || { echo "no binaries given" >&2; exit 2; }
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STEM="$NAME-$VERSION-linux-aarch64"
STAGE=$(mktemp -d)/$STEM
mkdir -p "$STAGE/usr/bin" "$STAGE/licenses" "$DIST"

for b in "$@"; do
  [ -f "$TARGET/release/$b" ] || { echo "missing binary: $TARGET/release/$b" >&2; exit 1; }
  install -m 0755 "$TARGET/release/$b" "$STAGE/usr/bin/$b"
done
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$ROOT/THIRD_PARTY_NOTICES" "$STAGE/"
cp "$ROOT/vendor/ggml-subset/LICENSE" "$STAGE/licenses/ggml-subset-LICENSE"
packages=$("$ROOT/tools/ci/binary_packages.py" "$@")
for package in $packages; do
  (cd "$ROOT" && tools/ci/crate_licenses.sh "$package" "$STAGE/licenses/crates/$package")
done

# Highest glibc symbol version any binary needs.
glibc=$(for b in "$STAGE"/usr/bin/*; do objdump -T "$b"; done | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -n1)
glibc=${glibc#GLIBC_}
sums=$(cd "$STAGE/usr/bin" && sha256sum -- *)
repo=${GITHUB_REPOSITORY:-oxstack/oxe}

cat > "$STAGE/README.txt" <<EOF
$NAME $VERSION for Linux on ARM64 (aarch64)
$(printf '%*s' $((${#NAME} + ${#VERSION} + 26)) '' | tr ' ' '=')

Built by GitHub Actions from the tagged source of https://github.com/$repo
  source commit: ${GITHUB_SHA:-unknown}
  ref:           ${GITHUB_REF_NAME:-unknown}
  workflow run:  ${GITHUB_SERVER_URL:-https://github.com}/$repo/actions/runs/${GITHUB_RUN_ID:-unknown}
  toolchain:     $(rustc --version), $(cc --version | head -n1)
Nobody builds these binaries by hand; the release workflows are
$repo/.github/workflows/release.yml (tag) and dev-build.yml (rolling dev).

Binaries (usr/bin/):
$sums

CPU: ARMv8.6-A with dotprod, i8mm, fp16 and bf16, no SVE needed. Tuned for and measured
on the Snapdragon X Elite (X1E80100) only; the X Plus has not been tested. The Rust code is
compiled with target-cpu=oryon-1 and the vendored ggml kernels with
-march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme. A core without i8mm or bf16 is expected
to stop with "Illegal instruction"; this has not been tested.
OS:  Linux aarch64 with glibc $glibc or newer.

The Flash engine (oxe-server --engine flash) is not part of these binaries: it links a
production ggml archive that is not in the source repository (issue #7).

Usage: README.md of https://github.com/$repo. No model weights are included.

Verify this download (from the same release):
  sha256sum -c SHA256SUMS
  gh attestation verify $STEM.tar.gz --repo $repo \\
    --signer-workflow $repo/.github/workflows/release.yml   (tag release)
  gh attestation verify $STEM.tar.gz --repo $repo \\
    --signer-workflow $repo/.github/workflows/dev-build.yml (rolling dev prerelease)

Licences: LICENSE (MIT, this project), NOTICE, THIRD_PARTY_NOTICES, licenses/ (the vendored
ggml subset, and every Rust crate in licenses/crates/).
EOF

epoch=${SOURCE_DATE_EPOCH:-$(git -C "$ROOT" log -1 --format=%ct 2>/dev/null || date +%s)}
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$epoch" \
    -C "$(dirname "$STAGE")" -czf "$DIST/$STEM.tar.gz" "$STEM"
(cd "$DIST" && sha256sum "$STEM.tar.gz" > "$STEM.tar.gz.sha256")
echo "glibc>=$glibc"
cat "$DIST/$STEM.tar.gz.sha256"
