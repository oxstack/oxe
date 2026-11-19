#!/usr/bin/env bash
# Carry the exact legal texts of the two pinned public Flash build sources.
#   flash_licenses.sh collect BIGMOE LLAMA DEST   copy the two LICENSE files and the manifest
#   flash_licenses.sh package SOURCE STAGE        verify them again and copy into a package stage
# Every check fails closed: a missing, symlinked, oversized or changed text, a changed manifest,
# or an upstream NOTICE that the pins do not know about stops the run.
set -euo pipefail

BIGMOE_SHA=cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30
LLAMA_SHA=94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d

die() { echo "$*" >&2; exit 1; }

manifest() {
  cat <<JSON
{
  "schema": "flash-source-licenses-v1",
  "sources": {
    "BigMoeOnEdge": {
      "license": "Apache-2.0",
      "notice": null,
      "repository": "https://github.com/Helldez/BigMoeOnEdge",
      "revision": "930b2c9ebd8c2cface6566bd9f11075ac07adfe0",
      "sha256": "$BIGMOE_SHA"
    },
    "llama.cpp": {
      "license": "MIT",
      "notice": null,
      "repository": "https://github.com/Helldez/llama.cpp",
      "revision": "7ac98f4a8089a36c0581b7c6e8973417a8fa175c",
      "sha256": "$LLAMA_SHA"
    }
  }
}
JSON
}

# checked FILE SHA: the file is a regular, non-symlink, at most 64 KiB, with the pinned hash.
checked() {
  local f=$1 pin=$2
  if [ -L "$f" ] || [ ! -f "$f" ] || [ "$(wc -c <"$f")" -gt 65536 ]; then
    die "missing or invalid Flash source licence: $(basename "$f")"
  fi
  [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$pin" ] || die "Flash source licence hash changed: $(basename "$f")"
}

collect() {
  local bigmoe=$1 llama=$2 dest=$3
  # Neither exact pinned source has a root NOTICE. A new notice requires explicit review
  # rather than silently dropping it or inventing one.
  [ -z "$(find "$bigmoe" -maxdepth 1 -iname 'NOTICE*' -print -quit)" ] || die "unexpected upstream NOTICE: BigMoeOnEdge"
  [ -z "$(find "$llama" -maxdepth 1 -iname 'NOTICE*' -print -quit)" ] || die "unexpected upstream NOTICE: llama.cpp"
  checked "$bigmoe/LICENSE" "$BIGMOE_SHA"
  checked "$llama/LICENSE" "$LLAMA_SHA"
  mkdir -p "$dest"
  cp "$bigmoe/LICENSE" "$dest/LICENSE-BigMoeOnEdge"
  cp "$llama/LICENSE" "$dest/LICENSE-llama.cpp"
  manifest >"$dest/FLASH-SOURCE-LICENSES.json"
}

package() {
  local src=$1 stage=$2 meta=$1/FLASH-SOURCE-LICENSES.json
  if [ -L "$meta" ] || [ ! -f "$meta" ] || [ "$(wc -c <"$meta")" -gt 16384 ]; then
    die "Flash source licence manifest missing or invalid"
  fi
  manifest | cmp -s - "$meta" || die "Flash source licence pins changed"
  checked "$src/LICENSE-BigMoeOnEdge" "$BIGMOE_SHA"
  checked "$src/LICENSE-llama.cpp" "$LLAMA_SHA"
  mkdir -p "$stage/licenses"
  cp "$src/LICENSE-BigMoeOnEdge" "$stage/licenses/BigMoeOnEdge-LICENSE"
  cp "$src/LICENSE-llama.cpp" "$stage/licenses/llama.cpp-LICENSE"
  cp "$meta" "$stage/FLASH-SOURCE-LICENSES.json"
}

case "${1:-}:$#" in
  collect:4) collect "$2" "$3" "$4" ;;
  package:3) package "$2" "$3" ;;
  *) die "usage: flash_licenses.sh collect BIGMOE LLAMA DEST | package SOURCE STAGE" ;;
esac
