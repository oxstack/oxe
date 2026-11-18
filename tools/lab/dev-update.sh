#!/usr/bin/env bash
# Public, attested dev assets only; no sign-in, stable install, service or PATH changes.
set -euo pipefail
umask 077
repo=oxstack/ox-llm
test "$#" -le 1 || { echo 'Usage: dev-update.sh [separate-dev-prefix]' >&2; exit 2; }
for tool in curl jq gh sha256sum tar flock diff getent; do command -v "$tool" >/dev/null; done
case $(uname -m) in aarch64|x86_64) arch=$(uname -m);; *) echo 'Unsupported architecture' >&2; exit 1;; esac
# Match the sandbox's denied .local/state/oxc path under the real account home.
[[ $UID == "$EUID" ]]
account=$(getent passwd "$UID")
[[ $account != *$'\n'* && $account != *$'\r'* ]]
IFS=: read -r _ _ account_uid _ _ account_home _ <<< "$account"
[[ $account_uid == "$UID" && $account_home == /* && $account_home != / && /$account_home/ != */../* ]]
root=$(realpath -ms -- "$account_home/.local/state/oxc/dev-builds")
prefix=${1:-"$root/default"}
[[ $prefix != *$'\n'* && $prefix != *$'\r'* && /$prefix/ != */../* ]]
prefix=$(realpath -ms -- "$prefix")
if [[ ${prefix%/*} != "$root" ]]; then
  echo 'Refusing a dev prefix outside the protected dev-builds directory' >&2
  exit 1
fi
ensure_ancestor() {
  local path=$1 parent owner mode
  if [[ $path != / ]]; then
    parent=${path%/*}
    ensure_ancestor "${parent:-/}" || return 1
  fi
  if [[ ! -e $path && ! -L $path ]]; then
    mkdir -m 700 -- "$path" || return 1
  fi
  [[ -d $path && ! -L $path ]] || return 1
  read -r owner mode < <(stat -c '%u %a' -- "$path") || return 1
  [[ $owner == 0 || $owner == "$EUID" ]] || return 1
  # A trusted sticky directory protects entries owned by root or this user.
  (( (8#$mode & 8#022) == 0 || (8#$mode & 8#1000) != 0 ))
}
parent=${prefix%/*}
ensure_ancestor "${parent:-/}" || {
  echo 'Refusing a dev prefix with an unsafe ancestor directory' >&2
  exit 1
}
if [[ ! -e $prefix && ! -L $prefix ]]; then
  mkdir -m 700 -- "$prefix"
fi
if ! [[ -d $prefix && ! -L $prefix && -O $prefix && $(stat -c %a -- "$prefix") == 700 ]]; then
  echo 'Refusing a dev prefix that is not an owned mode-0700 directory' >&2
  exit 1
fi
for state in .dev-channel .update.lock; do
  if test -e "$prefix/$state" || test -L "$prefix/$state"; then
    { test -f "$prefix/$state" && test ! -L "$prefix/$state" && test -O "$prefix/$state"; } || exit 1
    test "$(stat -c %h "$prefix/$state")" = 1
  fi
done
if test ! -f "$prefix/.dev-channel"; then
  test -z "$(find "$prefix" -mindepth 1 -maxdepth 1 -print -quit)" || {
    echo 'Refusing a nonempty prefix without a dev-channel marker' >&2; exit 1;
  }
fi
exec 9>>"$prefix/.update.lock"
flock 9
test ! -L "$prefix/.dev-channel"
if test -f "$prefix/.dev-channel"; then
  test "$(cat "$prefix/.dev-channel")" = 'ox-llm dev v1'
else
  printf 'ox-llm dev v1\n' > "$prefix/.dev-channel"
fi
work=$(mktemp -d "$prefix/.download.XXXXXXXX")
trap 'rm -rf -- "$work"; rm -f -- "$prefix/.current.$$"' EXIT
fetch() {
  curl -q --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
    --connect-timeout 15 --max-time 120 --retry 2 --max-filesize "$3" "$1" -o "$2"
}
fetch "https://api.github.com/repos/$repo/releases/tags/dev" "$work/release.json" 5242880
sha=$(jq -er 'select(.tag_name=="dev" and .prerelease and (.draft|not)) | .target_commitish | select(test("^[0-9a-f]{40}$"))' "$work/release.json")
archive=$(jq -er --arg suffix "-dev.${sha:0:7}-linux-$arch.tar.gz" \
  '[.assets[]|select(.state=="uploaded" and (.name|startswith("ox-llm-v")) and (.name|endswith($suffix)))|.name] | select(length==1) | .[0]' "$work/release.json")
version=${archive#ox-llm-}; version=${version%-linux-"$arch".tar.gz}
[[ $version =~ ^v[0-9]+\.[0-9]+\.[0-9]+[0-9A-Za-z.+-]*-dev\.[0-9a-f]{7}$ ]]
download() {
  local url
  url=$(jq -er --arg name "$1" '[.assets[]|select(.state=="uploaded" and .name==$name)|.browser_download_url] | select(length==1) | .[0]' "$work/release.json")
  case "$url" in "https://github.com/$repo/releases/download/dev/"*) ;; *) return 1;; esac
  fetch "$url" "$work/$1" "$2"
}
download "$archive" 1073741824
download "SHA256SUMS-$version" 1048576
bundle="ox-llm-$version.provenance.jsonl"
download "$bundle" 16777216
digest=$(awk -v name="$archive" '$2==name && $1 ~ /^[0-9a-f]+$/ && length($1)==64 {print $1}' "$work/SHA256SUMS-$version")
[[ $digest =~ ^[0-9a-f]{64}$ ]]
printf '%s  %s\n' "$digest" "$work/$archive" | sha256sum -c - >/dev/null
# --bundle disables gh's authentication requirement; never consult a saved account.
env -u GH_TOKEN -u GITHUB_TOKEN -u GH_ENTERPRISE_TOKEN -u GITHUB_ENTERPRISE_TOKEN -u GH_DEBUG \
  GH_CONFIG_DIR="$work/gh" GH_PROMPT_DISABLED=1 gh attestation verify "$work/$archive" --bundle "$work/$bundle" \
  --repo "$repo" --hostname github.com --signer-workflow "$repo/.github/workflows/dev-build.yml" \
  --source-ref refs/heads/main --source-digest "$sha" --deny-self-hosted-runners >/dev/null
id="$sha-${digest:0:12}"
mkdir -p "$prefix/releases" "$work/payload"
tar -tzf "$work/$archive" > "$work/entries"
! grep -Ev '^usr(/|$)' "$work/entries"
! grep -E '(^|/)\.\.(/|$)' "$work/entries"
tar -xzf "$work/$archive" -C "$work/payload" --no-same-owner --no-same-permissions
{ test -x "$work/payload/usr/bin/oxa" && test ! -L "$work/payload/usr/bin/oxa"; } || exit 1
jq -n --arg sha "$sha" --arg version "$version" --arg digest "$digest" \
  --arg time "$(date -u +%FT%TZ)" --arg archive "$archive" \
  '{channel:"dev",source_sha:$sha,version:$version,archive:$archive,sha256:$digest,installed_at:$time}' \
  > "$work/payload/receipt.json"
cp "$work/$bundle" "$work/payload/provenance.jsonl"
if test -e "$prefix/releases/$id"; then
  diff --no-dereference -qr "$work/payload/usr" "$prefix/releases/$id/usr"
else
  mv "$work/payload" "$prefix/releases/$id"
fi
old=$(readlink "$prefix/current" || true)
test ! -e "$prefix/current" || test -L "$prefix/current"
if test -e "$prefix/bin" || test -L "$prefix/bin"; then
  { test -L "$prefix/bin" && test "$(readlink "$prefix/bin")" = current/usr/bin; } || exit 1
else
  ln -s current/usr/bin "$prefix/bin"
fi
ln -s "releases/$id" "$prefix/.current.$$"
mv -Tf "$prefix/.current.$$" "$prefix/current"
# Prune only after switching to a different build so reinstalls retain rollback.
if test "$old" != "releases/$id"; then
  for installed in "$prefix"/releases/*; do
    test -d "$installed" && test ! -L "$installed" || continue
    [[ ${installed##*/} =~ ^[0-9a-f]{40}-[0-9a-f]{12}$ ]] || continue
    test "$installed" = "$prefix/releases/$id" || test "$installed" = "$prefix/$old" || {
      jq -e '.channel=="dev"' "$installed/receipt.json" >/dev/null && rm -rf -- "$installed"
    }
  done
fi
printf 'Installed dev %s (%s); receipt: %s/current/receipt.json\n' "$version" "$sha" "$prefix"
