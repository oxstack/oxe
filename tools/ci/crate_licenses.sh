#!/usr/bin/env bash
# Collect the licence texts of the third-party crates linked into one binary.
#
#   tools/ci/crate_licenses.sh PACKAGE OUT_DIR [--features F]
#
# Lists the package's normal (runtime) dependency closure with `cargo tree` and copies each
# crate's LICENSE*/COPYING*/NOTICE* files from the cargo registry sources to
# OUT_DIR/<name>-<version>/. Writes OUT_DIR/INDEX.txt (name, version, licence, repository).
# Workspace crates are skipped: the repository's LICENSE / NOTICE / THIRD_PARTY_NOTICES cover them.
# Fails if a crate's source directory is not found, or it has neither a licence file nor a
# licence expression. Run it from the workspace directory (where Cargo.lock is).
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,11p' "$0" >&2; exit 2; }
package=$1 out=$2
shift 2
feat=()
if [ "${1:-}" = "--features" ]; then feat=(--features "${2:?--features needs a value}"); fi
registry="${CARGO_HOME:-$HOME/.cargo}/registry/src"
mkdir -p "$out"
tree=$(cargo tree --locked -p "$package" ${feat[@]+"${feat[@]}"} -e normal --prefix none \
  --format '{p}|{l}|{r}')
rows=$(mktemp)
trap 'rm -f "$rows"' EXIT
# Lines look like "name v1.2.3|MIT OR Apache-2.0|https://repo"; "(proc-macro)" follows the
# version, a trailing "(*)" marks a repeated subtree, workspace crates carry "(/abs/path)".
while IFS='|' read -r head lic repo; do
  [ -n "$head" ] || continue
  read -r name ver rest <<<"$head"
  case $rest in "(/"*) continue ;; esac
  repo=${repo% (\*)}
  printf '%s %s\t%s\t%s\n' "$name" "${ver#v}" "${lic:-(see files)}" "$repo"
done <<<"$tree" | LC_ALL=C sort -u >"$rows"
bad=()
while IFS=$'\t' read -r id lic _; do
  name=${id% *} ver=${id#* }
  src=
  for d in "$registry"/*/"$name-$ver"; do [ -d "$d" ] && src=$d && break; done
  if [ -z "$src" ]; then bad+=("$id (source not found)"); continue; fi
  dest="$out/$name-$ver"
  mkdir -p "$dest"
  n=0
  while IFS= read -r f; do cp "$f" "$dest/"; n=$((n + 1)); done < <(
    find "$src" -maxdepth 1 -type f \( -iname 'LICENSE*' -o -iname 'LICENCE*' -o -iname 'COPYING*' \
      -o -iname 'NOTICE*' -o -iname 'UNLICENSE*' \) | LC_ALL=C sort)
  if [ "$n" -eq 0 ] && [ "$lic" = "(see files)" ]; then bad+=("$id (no licence information)"); fi
done <"$rows"
{
  echo "Third-party Rust crates linked into this binary (name version, licence, repository)."
  echo "Each crate's licence files are in the directory of the same name."
  echo
  cat "$rows"
} >"$out/INDEX.txt"
if [ ${#bad[@]} -gt 0 ]; then
  printf 'licence collection failed: %s\n' "${bad[@]}" >&2
  exit 1
fi
echo "$(wc -l <"$rows") crates -> $out"
