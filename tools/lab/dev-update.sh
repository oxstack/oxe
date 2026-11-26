#!/usr/bin/env bash
# Install the rolling oxe dev build (the public `dev` prerelease of oxstack/oxe) under
# ~/.local/state/oxe/dev-builds/default/, beside no stable install and with no PATH change.
#
# What it does: reads the public dev release metadata, downloads the newest
# oxe-*-linux-aarch64.tar.gz with its SHA256SUMS and its provenance bundle, verifies the checksum
# and the Sigstore attestation (from the bundle, so no GitHub sign-in is needed), checks that the
# packaged oxe reports the packaged version, extracts only the usr/ tree, and switches `current`
# to it. Nothing outside the dev-builds directory is written: no stable install, no unit, no PATH
# edit, no reboot.
#
# usage: dev-update.sh [--prefix DIR]
#   --prefix DIR  an existing or new directory under ~/.local/state/oxe/dev-builds
#                 (default: .../default)
set -euo pipefail
umask 077
repo=oxstack/oxe
signer=$repo/.github/workflows/dev-build.yml

usage() {
  echo 'Usage: dev-update.sh [--prefix DIR]' >&2
  echo '  Installs the rolling oxe dev build under ~/.local/state/oxe/dev-builds/default/.' >&2
  echo '  It does not touch PATH or a stable install.' >&2
  exit 2
}

prefix=
while test "$#" -gt 0; do
  case $1 in
    --prefix)
      test "$#" -ge 2 || usage
      prefix=$2
      shift 2
      ;;
    -h | --help) usage ;;
    *) usage ;;
  esac
done

die() {
  echo "dev-update: $1" >&2
  exit 1
}

for tool in curl jq gh sha256sum tar flock mktemp realpath stat uname awk sed getent; do
  command -v "$tool" >/dev/null || die "$tool is required"
done
case $(uname -m) in
  aarch64) arch=aarch64 ;;
  *) die "the oxe dev tarballs are linux-aarch64 only" ;;
esac

# The install root is this account's own: resolve the home from the passwd entry, never $HOME.
[[ $UID == "$EUID" ]] || die 'run it as the account that owns the home directory'
account=$(getent passwd "$UID")
[[ $account != *$'\n'* && $account != *$'\r'* ]]
IFS=: read -r _ _ account_uid _ _ account_home _ <<<"$account"
[[ $account_uid == "$UID" && $account_home == /* && $account_home != / && /$account_home/ != */../* ]] ||
  die 'cannot resolve this account home directory'
root=$(realpath -ms -- "$account_home/.local/state/oxe/dev-builds")
prefix=${prefix:-"$root/default"}
[[ $prefix != *$'\n'* && $prefix != *$'\r'* && /$prefix/ != */../* ]] || die 'unsafe dev prefix path'
prefix=$(realpath -ms -- "$prefix")
[[ ${prefix%/*} == "$root" ]] || die "refusing a dev prefix outside $root"

# umask 077 above: new directories here are 0700 without a separate mode argument.
mkdir -p -- "$root"
[[ -d $root && ! -L $root && -O $root ]] || die "unsafe dev-builds directory: $root"
(( (8#$(stat -c %a -- "$root") & 8#022) == 0 )) || die "dev-builds directory is writable by others: $root"
if [[ ! -e $prefix && ! -L $prefix ]]; then
  mkdir -m 700 -- "$prefix"
fi
[[ -d $prefix && ! -L $prefix && -O $prefix ]] || die "unsafe dev prefix: $prefix"
[[ $(stat -c %a -- "$prefix") == 700 ]] || die "dev prefix must be mode 0700: $prefix"

# A prefix this script made carries the channel marker; never take over another directory.
if [[ ! -f $prefix/.dev-channel ]]; then
  [[ -z $(find "$prefix" -mindepth 1 -maxdepth 1 -print -quit) ]] ||
    die "refusing a nonempty prefix without a channel marker: $prefix"
  printf 'oxe dev v1\n' >"$prefix/.dev-channel"
fi
[[ $(cat -- "$prefix/.dev-channel") == 'oxe dev v1' ]] || die "unexpected channel marker in $prefix"

exec 9>>"$prefix/.update.lock"
flock 9

work=$(mktemp -d "$prefix/.download.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
fetch() {
  curl -q --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
    --connect-timeout 15 --max-time 120 --retry 2 --max-filesize "$3" "$1" -o "$2"
}

# 1. The public dev release: its commit, its newest aarch64 tarball and that build's version.
fetch "https://api.github.com/repos/$repo/releases/tags/dev" "$work/release.json" 5242880
sha=$(jq -er 'select(.tag_name=="dev" and .prerelease and (.draft|not))
  | .target_commitish | select(test("^[0-9a-f]{40}$"))' "$work/release.json") ||
  die 'the dev release does not name a commit'
archive=$(jq -er --arg suffix "-linux-$arch.tar.gz" '
  [.assets[] | select(.state=="uploaded" and (.name|startswith("oxe-v")) and (.name|endswith($suffix))) | .name]
  | select(length>0)
  | max_by(capture("-dev\\.(?<n>[0-9]+)\\.[0-9a-f]{7}-linux-").n | tonumber)' "$work/release.json") ||
  die 'the dev release has no aarch64 tarball'
version=${archive#oxe-}
version=${version%-linux-"$arch".tar.gz}
[[ $version =~ ^v[0-9]+\.[0-9]+\.[0-9]+-dev\.[0-9]+\.[0-9a-f]{7}$ ]] ||
  die "unexpected dev asset name: $archive"
semver=$(sed -E 's/(-dev\.[0-9]+)\.([0-9a-f]{7})$/\1+\2/' <<<"$version")

# 2. Sums and provenance for exactly this file, then the bytes.
download() {
  local url
  url=$(jq -er --arg name "$1" '[.assets[] | select(.state=="uploaded" and .name==$name) | .browser_download_url]
    | select(length==1) | .[0]' "$work/release.json") ||
    die "the dev release has no unique asset $1"
  case "$url" in "https://github.com/$repo/releases/download/dev/"*) ;; *) die "unexpected asset URL for $1" ;; esac
  fetch "$url" "$work/$1" "$2"
}
download "$archive" 1073741824
download SHA256SUMS 1048576
bundle="oxe-$version.provenance.jsonl"
download "$bundle" 16777216
# sha256sum may print "./name"; match on the bare asset name either way.
digest=$(awk -v name="$archive" '{n=$2; sub(/^\.\//, "", n)}
  n==name && $1 ~ /^[0-9a-f]+$/ && length($1)==64 {print $1}' "$work/SHA256SUMS")
[[ $digest =~ ^[0-9a-f]{64}$ ]] || die "SHA256SUMS has no digest for $archive"
printf '%s  %s\n' "$digest" "$work/$archive" | sha256sum -c - >/dev/null ||
  die "checksum mismatch: $archive"
# --bundle disables gh's authentication requirement; never consult a saved account.
env -u GH_TOKEN -u GITHUB_TOKEN -u GH_ENTERPRISE_TOKEN -u GITHUB_ENTERPRISE_TOKEN -u GH_DEBUG \
  GH_CONFIG_DIR="$work/gh" GH_PROMPT_DISABLED=1 \
  gh attestation verify "$work/$archive" --bundle "$work/$bundle" --repo "$repo" \
  --hostname github.com --signer-workflow "$signer" --source-ref refs/heads/main \
  --source-digest "$sha" --deny-self-hosted-runners >/dev/null ||
  die "provenance mismatch: $archive"
id="$sha-${digest:0:12}"

# 3. The tarball holds one top directory with the usr/ tree; nothing else is ours to install, and
# the packaged binaries must match their stamp.
mkdir -p "$work/payload" "$work/unpack" "$prefix/releases"
# The tarball unpacks to one directory named after the file (tools/ci/package.sh).
top=${archive%.tar.gz}
tar -tzf "$work/$archive" >"$work/entries"
while IFS= read -r entry; do
  case $entry in
    .. | ../* | */.. | */../*) die "refusing a tarball with a .. path: $entry" ;;
    "$top" | "$top"/*) ;;
    *) die "refusing a tarball with a file outside $top/: $entry" ;;
  esac
done <"$work/entries"
tar -xzf "$work/$archive" -C "$work/unpack" --no-same-owner --no-same-permissions
[[ -d $work/unpack/$top/usr ]] || die "the tarball has no usr/ tree: $archive"
mv -- "$work/unpack/$top/usr" "$work/payload/usr"
for name in oxe oxe-server local-ai-memlat-boot; do
  [[ -f $work/payload/usr/bin/$name && ! -L $work/payload/usr/bin/$name && -x $work/payload/usr/bin/$name ]] ||
    die "the tarball has no executable usr/bin/$name"
done
want=${semver#v}
got=$(env -i "$work/payload/usr/bin/oxe" --version)
[[ $got == "oxe $want" ]] || die "oxe reports '$got', the release says 'oxe $want'"
jq -n --arg sha "$sha" --arg version "$semver" --arg digest "$digest" --arg archive "$archive" \
  --arg time "$(date -u +%FT%TZ)" \
  '{channel:"dev",source_sha:$sha,version:$version,archive:$archive,sha256:$digest,installed_at:$time}' \
  >"$work/payload/receipt.json"
cp -- "$work/$bundle" "$work/payload/provenance.jsonl"

# 4. Install this build once, then point `current` at it. Reinstalling the same id reuses it.
release=$prefix/releases/$id
if [[ -e $release || -L $release ]]; then
  [[ -d $release && ! -L $release && -O $release ]] || die "unsafe release directory: $release"
  diff -qr --no-dereference "$work/payload/usr" "$release/usr" >/dev/null ||
    die "installed build $id differs from the downloaded one"
  cp -- "$work/payload/receipt.json" "$release/receipt.json"
  cp -- "$work/payload/provenance.jsonl" "$release/provenance.jsonl"
else
  mv -- "$work/payload" "$release"
fi
old=$(readlink -- "$prefix/current" 2>/dev/null) || old=
if [[ -e $prefix/bin || -L $prefix/bin ]]; then
  [[ -L $prefix/bin && $(readlink -- "$prefix/bin") == current/usr/bin ]] ||
    die "refusing an unexpected $prefix/bin"
else
  ln -s current/usr/bin "$prefix/bin"
fi
ln -s "releases/$id" "$prefix/.current.$$"
mv -Tf -- "$prefix/.current.$$" "$prefix/current"
if [[ $old != "releases/$id" ]]; then
  for installed in "$prefix"/releases/*; do
    [[ -d $installed && ! -L $installed ]] || continue
    [[ ${installed##*/} =~ ^[0-9a-f]{40}-[0-9a-f]{12}$ ]] || continue
    [[ $installed == "$release" || $installed == "$prefix/$old" ]] && continue
    jq -e '.channel=="dev"' "$installed/receipt.json" >/dev/null 2>&1 && rm -rf -- "$installed"
  done
fi

printf 'Installed oxe dev %s (%s)\n  release: %s\n  binaries: %s/bin (oxe, oxe-server, local-ai-memlat-boot)\n' \
  "$semver" "$sha" "$release" "$prefix"
printf 'No PATH change and no stable install were made.\n'
