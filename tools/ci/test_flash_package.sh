#!/usr/bin/env bash
# Offline packaging fixtures: real scripts and tar, fake binary and cargo/toolchain stubs.
#   tools/ci/test_flash_package.sh      (run from anywhere; needs no network, no cargo, no model)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIGMOE=cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30
LLAMA=94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
FAILS=0

[ "$(sha256sum <"$ROOT/third_party/BigMoeOnEdge-LICENSE" | cut -d' ' -f1)" = "$BIGMOE" ] || { echo "pin mismatch: BigMoeOnEdge"; exit 1; }
[ "$(sha256sum <"$ROOT/third_party/llama.cpp-LICENSE" | cut -d' ' -f1)" = "$LLAMA" ] || { echo "pin mismatch: llama.cpp"; exit 1; }

# Fresh fixture per case: a repo copy, two upstream source trees, the prod bundle, stub tools.
setup() {
  rm -rf "$TMP/w"; mkdir -p "$TMP/w"
  W=$TMP/w FROOT=$TMP/w/repo
  local f
  for f in LICENSE NOTICE THIRD_PARTY_NOTICES vendor/ggml-subset/LICENSE vendor/ggml-flash/LICENSE \
           third_party/stable-diffusion.cpp-LICENSE tools/ci/package.sh tools/ci/crate_licenses.sh \
           tools/ci/flash_licenses.sh; do
    mkdir -p "$FROOT/$(dirname "$f")"; cp "$ROOT/$f" "$FROOT/$f"
  done
  mkdir -p "$W/BigMoeOnEdge" "$W/llama.cpp"
  cp "$ROOT/third_party/BigMoeOnEdge-LICENSE" "$W/BigMoeOnEdge/LICENSE"
  cp "$ROOT/third_party/llama.cpp-LICENSE" "$W/llama.cpp/LICENSE"
  PROD=$W/prod
  bash "$FROOT/tools/ci/flash_licenses.sh" collect "$W/BigMoeOnEdge" "$W/llama.cpp" "$PROD" || { echo "fixture collect failed"; exit 1; }
  echo 'fixture: no native libraries built or linked' >"$PROD/MANIFEST"
  mkdir -p "$W/target/release" "$W/stub"
  echo 'package fixture, not a runnable binary' >"$W/target/release/oxe-server"
  printf '#!/bin/sh\n' >"$W/stub/cargo"
  printf '#!/bin/sh\necho "rustc fixture"\n' >"$W/stub/rustc"
  printf '#!/bin/sh\necho "cc fixture"\n' >"$W/stub/cc"
  printf '#!/bin/sh\necho "fixture GLIBC_2.39"\n' >"$W/stub/objdump"
  chmod 700 "$W"/stub/*
  rm -rf "$W/dist"
}

# package: run package.sh; echoes nothing, returns its status.
package() {
  env PATH="$W/stub:$PATH" OXF_GGML_PROD="${PROD_OVERRIDE-$PROD}" SOURCE_DATE_EPOCH=1 \
    bash "$FROOT/tools/ci/package.sh" flash-fixture test "$W/dist" server flash "$W/target" oxe-server \
    >"$W/out" 2>&1
}
package_noprod() {
  env -u OXF_GGML_PROD PATH="$W/stub:$PATH" SOURCE_DATE_EPOCH=1 \
    bash "$FROOT/tools/ci/package.sh" flash-fixture test "$W/dist" server flash "$W/target" oxe-server \
    >"$W/out" 2>&1
}
archives() { ls "$W"/dist/*.tar.gz 2>/dev/null; }

ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; sed 's/^/     /' "$W/out" 2>/dev/null | head -20; FAILS=$((FAILS + 1)); }
# fails_closed NAME: the last package/collect call must have failed and left no archive.
fails_closed() { if [ "$1" -ne 0 ] && [ -z "$(archives)" ]; then ok "$2"; else bad "$2"; fi; }

# 1. the archive carries both upstream texts, attribution and the manifest
setup
if package && [ -n "$(archives)" ]; then
  a=$(archives); p=flash-fixture-test-linux-aarch64
  mkdir "$W/x"; tar -xzf "$a" -C "$W/x"
  # THIRD_PARTY_NOTICES is the repo file; the pins must be named in it or in the manifest.
  if grep -q 930b2c9ebd8c2cface6566bd9f11075ac07adfe0 "$W/x/$p/THIRD_PARTY_NOTICES" &&
     grep -q 7ac98f4a8089a36c0581b7c6e8973417a8fa175c "$W/x/$p/THIRD_PARTY_NOTICES" &&
     grep -q "\"sha256\": \"$BIGMOE\"" "$W/x/$p/FLASH-SOURCE-LICENSES.json" &&
     grep -q "\"sha256\": \"$LLAMA\"" "$W/x/$p/FLASH-SOURCE-LICENSES.json" &&
     [ "$(grep -c '"notice": null' "$W/x/$p/FLASH-SOURCE-LICENSES.json")" = 2 ] &&
     cmp -s "$W/x/$p/licenses/BigMoeOnEdge-LICENSE" "$ROOT/third_party/BigMoeOnEdge-LICENSE" &&
     cmp -s "$W/x/$p/licenses/llama.cpp-LICENSE" "$ROOT/third_party/llama.cpp-LICENSE"; then
    ok "archive carries exact both upstream texts and attribution"
  else bad "archive carries exact both upstream texts and attribution"; fi
else bad "archive carries exact both upstream texts and attribution"; fi

# 2. missing BigMoeOnEdge licence
setup; rm "$PROD/LICENSE-BigMoeOnEdge"; package; fails_closed $? "missing bigmoe licence fails closed"
# 3. tampered llama licence
setup; echo changed >"$PROD/LICENSE-llama.cpp"; package; fails_closed $? "tampered llama licence fails closed"
# 4. missing manifest
setup; rm "$PROD/FLASH-SOURCE-LICENSES.json"; package; fails_closed $? "missing source metadata fails closed"
# 5. changed revision
setup; sed -i 's/930b2c9ebd8c2cface6566bd9f11075ac07adfe0/unreviewed/' "$PROD/FLASH-SOURCE-LICENSES.json"
package; fails_closed $? "changed source revision fails closed"
# 6. flash without a prod bundle
setup; package_noprod; fails_closed $? "flash without prod bundle fails closed"
# 7. unreviewed upstream NOTICE
setup; echo 'new upstream notice requires reviewed pins' >"$W/BigMoeOnEdge/NOTICE"
bash "$FROOT/tools/ci/flash_licenses.sh" collect "$W/BigMoeOnEdge" "$W/llama.cpp" "$W/prod2" >"$W/out" 2>&1
fails_closed $? "unreviewed upstream notice fails closed"

[ "$FAILS" -eq 0 ] && echo "all flash package fixtures passed" || { echo "$FAILS fixture(s) failed"; exit 1; }
