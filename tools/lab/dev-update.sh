#!/usr/bin/env bash
# Public, attested dev assets only; no sign-in or stable install. The optional
# --promote mode activates the verified build for the operator's existing N0
# units by replacing their regular executable and checksum manifest together,
# and rolls the exact previous pair back when the result cannot be verified.
#
# Registry exception: this is a repository operator script under tools/lab/,
# not a shipped oxa CLI surface, so its flags have no declaration in
# oxa/oxa-interactive/src/command_registry.rs. Its usage lives in this file and
# in docs/DEV.md, next to the units it edits.
set -euo pipefail
umask 077
repo=oxstack/ox-llm

usage() {
  echo 'Usage: dev-update.sh [separate-dev-prefix]' >&2
  echo '       dev-update.sh [separate-dev-prefix] --promote --unit UNIT --board-unit UNIT' >&2
  echo '         --binary PATH --manifest PATH --install-lock PATH --session PATH' >&2
  echo '         --receipt PATH [--lead-url URL]' >&2
  exit 2
}

# Promotion inputs are the operator's own existing artifacts: the Lead user
# unit, the shadow board sampler unit, the regular executable both units run
# with its checksum manifest, the operator artifact-install lock, the dedicated
# Session file, and the rollout receipt this mode rewrites.
mode=install
positional=0
prefix=
unit=
board_unit=
binary=
manifest=
install_lock=
session=
rollout=
lead_url=http://127.0.0.1:8743/chat
while test "$#" -gt 0; do
  case $1 in
    --promote)
      test "$mode" = install || usage
      mode=promote
      shift
      ;;
    --unit | --board-unit | --binary | --manifest | --install-lock | --session | --receipt | --lead-url)
      test "$#" -ge 2 || usage
      case $1 in
        --unit) unit=$2 ;;
        --board-unit) board_unit=$2 ;;
        --binary) binary=$2 ;;
        --manifest) manifest=$2 ;;
        --install-lock) install_lock=$2 ;;
        --session) session=$2 ;;
        --receipt) rollout=$2 ;;
        --lead-url) lead_url=$2 ;;
      esac
      shift 2
      ;;
    --*) usage ;;
    *)
      test "$positional" = 0 || usage
      prefix=$1
      positional=1
      shift
      ;;
  esac
done
if test "$mode" = promote; then
  test -n "$unit" && test -n "$board_unit" && test -n "$binary" && test -n "$manifest" &&
    test -n "$install_lock" && test -n "$session" && test -n "$rollout" || usage
fi
for tool in curl jq gh sha256sum tar flock diff getent mktemp; do command -v "$tool" >/dev/null; done
test "$mode" = install || command -v systemctl >/dev/null
case $(uname -m) in aarch64 | x86_64) arch=$(uname -m) ;; *) echo 'Unsupported architecture' >&2; exit 1 ;; esac
# Match the sandbox's denied .local/state/oxc path under the real account home.
[[ $UID == "$EUID" ]]
account=$(getent passwd "$UID")
[[ $account != *$'\n'* && $account != *$'\r'* ]]
IFS=: read -r _ _ account_uid _ _ account_home _ <<<"$account"
[[ $account_uid == "$UID" && $account_home == /* && $account_home != / && /$account_home/ != */../* ]]
root=$(realpath -ms -- "$account_home/.local/state/oxc/dev-builds")
prefix=${prefix:-"$root/default"}
[[ $prefix != *$'\n'* && $prefix != *$'\r'* && /$prefix/ != */../* ]]
prefix=$(realpath -ms -- "$prefix")
if [[ ${prefix%/*} != "$root" ]]; then
  echo 'Refusing a dev prefix outside the protected dev-builds directory' >&2
  exit 1
fi
# Read-only form of the stock ancestor rule, used before anything is activated:
# real directories owned by root or this account, never writable by another
# account unless sticky.
trusted_parent() {
  local path=$1 parent owner mode
  test -n "$path" && test -d "$path" && test ! -L "$path" || return 1
  if [[ $path != / ]]; then
    # "/home" trims to an empty string: the root of the chain is "/" exactly as
    # the stock ancestor check spells it.
    parent=${path%/*}
    trusted_parent "${parent:-/}" || return 1
  fi
  read -r owner mode < <(stat -c '%u %a' -- "$path") || return 1
  [[ $owner == 0 || $owner == "$EUID" ]] || return 1
  (( (8#$mode & 8#022) == 0 || (8#$mode & 8#1000) != 0 ))
}

# Promotion activates an existing service, so it never creates its own install.
if test "$mode" = promote; then
  { test -d "$prefix" && test ! -L "$prefix"; } || {
    echo 'No deploy: the dev prefix is not installed' >&2
    exit 1
  }
  for name in "$unit" "$board_unit"; do
    [[ $name =~ ^[A-Za-z0-9_.@:-]+\.service$ ]] || {
      echo 'Refusing an unsafe promotion unit name' >&2
      exit 1
    }
  done
  for input in "$binary" "$manifest" "$install_lock" "$session" "$rollout"; do
    case $input in
      /*) ;;
      *) echo 'Refusing an unsafe promotion input path' >&2; exit 1 ;;
    esac
    [[ $input != *$'\n'* && $input != *$'\r'* && /$input/ != */../* ]] || {
      echo 'Refusing an unsafe promotion input path' >&2
      exit 1
    }
  done
  # Each file is staged and renamed inside its own directory, so the two live
  # paths may differ (the N0 layout keeps board-bin/oxa and board/binary.sha256);
  # each parent chain must be trusted on its own.
  trusted_parent "${binary%/*}" || {
    echo 'Refusing an executable directory that is not trusted' >&2
    exit 1
  }
  trusted_parent "${manifest%/*}" || {
    echo 'Refusing a manifest directory that is not trusted' >&2
    exit 1
  }
  for input in "$install_lock" "$session" "$rollout"; do
    trusted_parent "${input%/*}" || {
      echo 'Refusing an untrusted promotion input directory' >&2
      exit 1
    }
  done
  binary_real=$(realpath -ms -- "$binary")
  # The canary is a loopback denial probe, never a remote or proxied request.
  [[ $lead_url =~ ^http://(127\.0\.0\.1|\[::1\]):[0-9]{1,5}/chat$ ]] || {
    echo 'Refusing a promotion canary outside loopback' >&2
    exit 1
  }
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
    echo 'Refusing a nonempty prefix without a dev-channel marker' >&2
    exit 1
  }
fi
# The stock download lock of this install: held for the whole download and
# import, and never used as the operator's artifact-install lock.
exec 9>>"$prefix/.update.lock"
flock 9
test ! -L "$prefix/.dev-channel"
if test -f "$prefix/.dev-channel"; then
  test "$(cat "$prefix/.dev-channel")" = 'ox-llm dev v1'
else
  printf 'ox-llm dev v1\n' >"$prefix/.dev-channel"
fi
work=$(mktemp -d "$prefix/.download.XXXXXXXX")
trap 'rm -rf -- "$work"; rm -f -- "$prefix/.current.$$"' EXIT
fetch() {
  curl -q --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
    --connect-timeout 15 --max-time 120 --retry 2 --max-filesize "$3" "$1" -o "$2"
}

# --- executable digest and guarded promotion helpers ------------------------
#
# The stock receipt keeps the archive digest. A build's executable gets its own
# digest, computed from the extracted bytes and stored beside the release; the
# promoted executable and its manifest carry that same value. Everything below
# is unused unless --promote was given.

no_baseline() {
  echo 'No deploy: no verified source baseline for the current install' >&2
  exit 1
}


# Owner-owned, not a symlink, and never writable by another account: a checksum
# manifest and a receipt are not secrets, but they must not be replaceable.
trusted_file() {
  test -f "$1" && test ! -L "$1" && test -O "$1" &&
    (( (8#$(stat -c %a -- "$1") & 8#022) == 0 ))
}

digest_of() {
  local sum
  sum=$(sha256sum -- "$1") || return 1
  printf '%s\n' "${sum%% *}"
}

# The operator's checksum manifest: a bare digest or sha256sum's fixed layout.
read_manifest() {
  local first=${1:-$manifest}
  trusted_file "$first" || return 1
  first=$(head -n 1 -- "$first")
  manifest_digest=${first%%[[:space:]]*}
  [[ $manifest_digest =~ ^[0-9a-f]{64}$ ]] || return 1
  manifest_name=
  if [[ $first == *[[:space:]]* ]]; then
    manifest_name=${first#*[[:space:]]}
    manifest_name=${manifest_name#"${manifest_name%%[![:space:]]*}"}
    [[ $manifest_name =~ ^[A-Za-z0-9._/-]+$ ]] || return 1
  fi
  return 0
}

unit_state() {
  local state
  state=$(systemctl --user show --property=ActiveState --value "$1") || return 1
  [[ $state =~ ^[A-Za-z-]+$ ]] || return 1
  printf '%s\n' "$state"
}

main_pid() {
  local pid
  pid=$(systemctl --user show --property=MainPID --value "$1") || return 1
  [[ $pid =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "$pid"
}

# The shadow board sampler must be idle: a timer run or an active sample would
# race the activation, and only the unit's own state proves it is not running.
board_idle() {
  local state
  state=$(unit_state "$board_unit") || return 1
  [[ $state == inactive ]]
}

# The digest of the process the configured Lead unit actually runs: its path must
# be the operator's regular executable, and the bytes hashed are the live ones
# from /proc, never what the path happens to hold now.
running_digest() {
  local pid exe
  pid=$(main_pid "$unit") || return 1
  exe=$(readlink -f "/proc/$pid/exe") || return 1
  [[ $exe == "$binary_real" ]] || return 1
  digest_of "/proc/$pid/exe"
}

verify_running() {
  local actual
  actual=$(running_digest) || return 1
  [[ $actual == "$1" ]]
}

manifest_matches() {
  read_manifest || return 1
  [[ $manifest_digest == "$1" ]]
}

# Public compare metadata only: the same ancestry answer the dev workflow uses,
# and never an attestation parser of its own.
compare_status() {
  local body=$work/compare.json
  fetch "https://api.github.com/repos/$repo/compare/$1...$2" "$body" 5242880 || return 1
  jq -er '.status | select(test("^(ahead|behind|diverged|identical)$"))' "$body"
}

# Make replacement files on the target filesystem with an exclusive,
# unpredictable name.  In particular, never redirect to a name another user
# could have planted as a symlink in an accepted sticky directory.
replacement_temp() {
  local target=$1 parent name
  parent=${target%/*}
  name=${target##*/}
  mktemp --tmpdir="$parent" ".$name.new.XXXXXXXX"
}

# The native dedicated-session sidecar lock, taken exactly like oxa does
# (<session>.lock, flock exclusive, non-blocking). It is held only across the
# activation and restart, never across the download; 2 means a turn is running.
session_held=0
session_take() {
  local lock=$session.lock
  if test -e "$lock" || test -L "$lock"; then
    { test -f "$lock" && test ! -L "$lock" && test -O "$lock"; } || return 1
  fi
  if test ! -e "$lock"; then
    (set -C; : >>"$lock") 2>/dev/null || return 1
  fi
  { : >>"$lock"; } 2>/dev/null || return 1
  exec 8>>"$lock"
  [[ $(readlink /proc/self/fd/8) == "$lock" ]] || return 1
  { test -f "$lock" && test ! -L "$lock" && test -O "$lock"; } || return 1
  if flock -n 8; then :; else
    exec 8>&-
    return 2
  fi
  # The native writer unlinks this sidecar when it releases. A lock on that dead
  # inode would let a new turn start beside the activation, so the held
  # descriptor and the path must still name the same file.
  session_held=1
  { [[ $(stat -Lc '%d:%i' /proc/self/fd/8) == "$(stat -Lc '%d:%i' -- "$lock")" ]]; } || {
    session_release
    return 1
  }
  return 0
}
session_release() {
  if test "$session_held" = 1; then
    exec 8>&-
    session_held=0
  fi
}

# The operator's artifact-install lock is a separate existing file: read-only
# open for the lock itself, so it can never be modified from here.
install_take() {
  { test -f "$install_lock" && test ! -L "$install_lock" && test -O "$install_lock" &&
    test "$(stat -c %h -- "$install_lock")" = 1; } || return 1
  { : <"$install_lock"; } 2>/dev/null || return 1
  exec 7<"$install_lock"
  if flock -n 7; then
    return 0
  fi
  exec 7<&-
  return 2
}

stage_previous() {
  promote_root=$prefix/promote
  stage=$promote_root/$sha
  if test -e "$promote_root" || test -L "$promote_root"; then
    { test -d "$promote_root" && test ! -L "$promote_root" && test -O "$promote_root"; } || {
      echo 'Refusing an unsafe promotion staging directory' >&2
      exit 1
    }
  else
    mkdir -m 700 -- "$promote_root"
  fi
  if test -e "$stage" || test -L "$stage"; then
    { test -d "$stage" && test ! -L "$stage" && test -O "$stage"; } || {
      echo 'Refusing an unsafe promotion staging directory' >&2
      exit 1
    }
  else
    mkdir -m 700 -- "$stage"
  fi
  # The exact binary and manifest being replaced, verified before the swap.
  test "$(digest_of "$binary")" = "$old_binary" || {
    echo 'No deploy: the installed binary changed before activation' >&2
    exit 1
  }
  trusted_file "$manifest" || {
    echo 'No deploy: the binary manifest is unavailable' >&2
    exit 1
  }
  read_manifest || {
    echo 'No deploy: the binary manifest is unavailable' >&2
    exit 1
  }
  test "$manifest_digest" = "$old_binary" || {
    echo 'No deploy: the running service does not match its binary manifest' >&2
    exit 1
  }
  binary_mode=$(stat -c %a -- "$binary")
  manifest_mode=$(stat -c %a -- "$manifest")
  mkdir -p -m 700 -- "$stage/previous"
  cp -- "$binary" "$stage/previous/oxa"
  chmod 600 -- "$stage/previous/oxa"
  cp -- "$manifest" "$stage/previous/manifest"
  chmod 600 -- "$stage/previous/manifest"
  printf '%s\n' "$old_binary" >"$stage/previous.sha256"
  test "$(digest_of "$stage/previous/oxa")" = "$old_binary" || {
    echo 'Refusing an unsafe promotion staging directory' >&2
    exit 1
  }
  jq -n --arg sha "$old_sha" --arg digest "$old_binary" --arg archive "$old_archive" \
    --arg binary_mode "$binary_mode" \
    --arg manifest_mode "$manifest_mode" \
    '{previous_source_sha:$sha,previous_archive_sha256:$archive,
      previous_binary_sha256:$digest,binary_mode:$binary_mode,manifest_mode:$manifest_mode}' \
    >"$stage/state.json"
}

# Everything decided before any artifact is downloaded: the baseline the
# operator installed, main ancestry of both ends, the idle board sampler and a
# free Session. The Session lock is only probed here, not held.
promote_gate() {
  local receipt=$prefix/current/receipt.json status running stock_release
  { test -f "$receipt" && test ! -L "$receipt" && test -O "$receipt"; } || no_baseline
  running=$(running_digest) || {
    echo 'No deploy: the running service does not match its executable path' >&2
    exit 1
  }
  old_binary=$running
  old_sha=
  old_archive=
  rollout_baseline=0
  if test -e "$rollout" || test -L "$rollout"; then
    trusted_file "$rollout" || {
      echo 'Refusing an unsafe rollout receipt path' >&2
      exit 1
    }
    # Only digest-bound rollout metadata may name the source of what runs.
    if test "$(jq -er '.binary_sha256 | select(test("^[0-9a-f]{64}$"))' "$rollout")" = "$running"; then
      old_sha=$(jq -er '.source_sha | select(test("^[0-9a-f]{40}$"))' "$rollout") || no_baseline
      old_archive=$(jq -er '.archive_sha256 | select(test("^[0-9a-f]{64}$"))' "$rollout") || no_baseline
      rollout_baseline=1
    fi
  fi
  if test -z "$old_sha"; then
    # Without that receipt the stock install may only stand in when its own
    # executable digest is the binary the units are running.
    stock_release=$(readlink "$prefix/current") || no_baseline
    [[ $stock_release =~ ^releases/[0-9a-f]{40}-[0-9a-f]{12}$ ]] || no_baseline
    read_manifest "$prefix/$stock_release/binary.sha256" || no_baseline
    test "$manifest_digest" = "$running" || no_baseline
    old_sha=$(jq -er 'select(.channel=="dev") | .source_sha | select(test("^[0-9a-f]{40}$"))' "$receipt") || no_baseline
    old_archive=$(jq -er '.sha256 | select(test("^[0-9a-f]{64}$"))' "$receipt") || no_baseline
  fi
  test -e "$manifest" || test -L "$manifest" || {
    echo 'No deploy: the binary manifest is unavailable' >&2
    exit 1
  }
  read_manifest || {
    echo 'No deploy: the binary manifest is unavailable' >&2
    exit 1
  }
  test "$manifest_digest" = "$running" || {
    echo 'No deploy: the running service does not match its binary manifest' >&2
    exit 1
  }
  status=$(compare_status "$old_sha" "$sha") || {
    echo 'No deploy: the source ancestry could not be checked' >&2
    exit 1
  }
  case $status in
    identical) echo 'No deploy: unchanged source'; exit 0 ;;
    ahead) ;;
    *) echo 'No deploy: older source'; exit 0 ;;
  esac
  status=$(compare_status "$sha" main) || {
    echo 'No deploy: the source ancestry could not be checked' >&2
    exit 1
  }
  case $status in
    ahead | identical) ;;
    *) echo 'No deploy: the release source is not on main' >&2; exit 1 ;;
  esac
  board_idle || { echo 'No deploy: the board sampler is not idle'; exit 0; }
  if session_take; then
    session_release
  else
    case $? in
      2) echo 'No deploy: an owner chat turn is in flight'; exit 0 ;;
      *) echo 'Refusing an unsafe session lock path' >&2; exit 1 ;;
    esac
  fi
}

# A plain GET on the chat endpoint must be refused (403): the served mutation
# boundary is up and denies a direct unauthenticated read.
denied_get() {
  local status
  status=$(curl -q --silent --show-error --output /dev/null --write-out '%{http_code}' \
    --noproxy '*' --proto '=http' --connect-timeout 5 --max-time 10 \
    --retry 10 --retry-connrefused --retry-delay 1 --retry-max-time 15 "$lead_url") || return 1
  [[ $status == 403 ]]
}

# Fixed safe text only: no message bytes, identities, URLs or header values. The
# receipt always describes the build that is activated, and is replaced
# atomically beside itself.
write_receipt() {
  local tmp
  tmp=$(replacement_temp "$rollout") || return 1
  jq -n --arg action "$1" --arg verification "$2" --arg source "$3" --arg archive "$4" \
    --arg binary "$5" --arg previous_source "$6" --arg previous_binary "$7" \
    --arg time "$(date -u +%FT%TZ)" \
    '{channel:"dev",action:$action,verification:$verification,source_sha:$source,
      archive_sha256:$archive,binary_sha256:$binary,previous_source_sha:$previous_source,
      previous_binary_sha256:$previous_binary,promoted_at:$time}' >"$tmp"
  chmod 600 -- "$tmp"
  mv -Tf -- "$tmp" "$rollout"
}

prune_stage() {
  local staged
  for staged in "$promote_root"/*; do
    test -d "$staged" && test ! -L "$staged" || continue
    [[ ${staged##*/} =~ ^[0-9a-f]{40}$ ]] || continue
    test "$staged" = "$stage" || rm -rf -- "$staged"
  done
}

# The exact verified pair goes back, executable and manifest together, and the
# running digest decides whether the rollback itself is verified.
rollback() {
  local previous_archive previous_binary binary_mode manifest_mode state binary_tmp manifest_tmp
  state=$stage/state.json
  previous_binary=$(jq -er '.previous_binary_sha256 | select(test("^[0-9a-f]{64}$"))' "$state") || previous_binary=
  previous_archive=$(jq -er '.previous_archive_sha256 | select(test("^[0-9a-f]{64}$"))' "$state") || previous_archive=
  binary_mode=$(jq -er '.binary_mode | select(test("^[0-7]{3,4}$"))' "$state") || binary_mode=700
  manifest_mode=$(jq -er '.manifest_mode | select(test("^[0-7]{3,4}$"))' "$state") || manifest_mode=600
  test "$(digest_of "$stage/previous/oxa")" = "$previous_binary" || {
    write_receipt failed unavailable "$old_sha" "$old_archive" "$new_digest" "$sha" "$old_binary"
    echo 'Deploy failed; the staged previous binary does not match its checksum' >&2
    exit 1
  }
  binary_tmp=$(replacement_temp "$binary") && manifest_tmp=$(replacement_temp "$manifest") &&
  { cp -- "$stage/previous/oxa" "$binary_tmp" &&
    chmod "$binary_mode" -- "$binary_tmp" &&
    cp -- "$stage/previous/manifest" "$manifest_tmp" &&
    chmod "$manifest_mode" -- "$manifest_tmp" &&
    mv -Tf -- "$binary_tmp" "$binary" &&
    mv -Tf -- "$manifest_tmp" "$manifest"; } || {
    rm -f -- "${binary_tmp:-}" "${manifest_tmp:-}"
    write_receipt failed unavailable "$old_sha" "${previous_archive:-$old_archive}" "$previous_binary" "$sha" "$new_digest"
    echo 'Deploy failed; the previous pair could not be restored' >&2
    exit 1
  }
  if systemctl --user restart "$unit" 2>/dev/null && verify_running "$previous_binary" &&
    manifest_matches "$previous_binary"; then
    session_release
    write_receipt failed verified "$old_sha" "$previous_archive" "$previous_binary" "$sha" "$new_digest"
    echo "Deploy failed; rolled back to the previous binary ($previous_binary)" >&2
  else
    write_receipt failed unverified "$old_sha" "$previous_archive" "$previous_binary" "$sha" "$new_digest"
    echo 'Deploy failed; the previous binary did not come back verified' >&2
  fi
  exit 1
}

# The verified receipt is checked again after the download, and the activation
# holds the operator install lock and the Session lock only for its own short
# window: swap the pair, restart the configured unit, verify, or roll back.
activate() {
  local receipt=$prefix/current/receipt.json new_sha sampler binary_tmp manifest_tmp
  { test -f "$receipt" && test ! -L "$receipt"; } || no_baseline
  new_sha=$(jq -er 'select(.channel=="dev") | .source_sha | select(test("^[0-9a-f]{40}$"))' "$receipt") || no_baseline
  if [[ $new_sha != "$sha" ]]; then
    echo 'No deploy: the verified receipt does not match the attested asset' >&2
    exit 1
  fi
  # The executable digest of this release, computed from its extracted bytes.
  read_manifest "$prefix/releases/$id/binary.sha256" || {
    echo 'No deploy: the verified release has no executable digest' >&2
    exit 1
  }
  new_digest=$manifest_digest
  test "$(digest_of "$prefix/releases/$id/usr/bin/oxa")" = "$new_digest" || {
    echo 'No deploy: the promoted binary does not match the verified release' >&2
    exit 1
  }
  # Short activation boundary: the operator install lock, an idle board sampler
  # and the native Session lock, taken only now and never across the download.
  if install_take; then :; else
    case $? in
      2) echo 'No deploy: another artifact install is in progress'; exit 0 ;;
      *) echo 'No deploy: the artifact-install lock is unavailable' >&2; exit 1 ;;
    esac
  fi
  board_idle || { echo 'No deploy: the board sampler is not idle'; exit 0; }
  if session_take; then :; else
    case $? in
      2) echo 'No deploy: an owner chat turn is in flight'; exit 0 ;;
      *) echo 'Refusing an unsafe session lock path' >&2; exit 1 ;;
    esac
  fi
  stage_previous
  printf '%s\n' "$new_digest" >"$stage/next.sha256"
  # Both files are prepared beside their targets, so each rename is atomic and
  # the pair is only ever promoted or restored together.
  binary_tmp=$(replacement_temp "$binary")
  manifest_tmp=$(replacement_temp "$manifest")
  cp -- "$prefix/releases/$id/usr/bin/oxa" "$binary_tmp"
  chmod "$binary_mode" -- "$binary_tmp"
  if test -n "$manifest_name"; then
    printf '%s  %s\n' "$new_digest" "$manifest_name" >"$manifest_tmp"
  else
    printf '%s\n' "$new_digest" >"$manifest_tmp"
  fi
  chmod "$manifest_mode" -- "$manifest_tmp"
  if { mv -Tf -- "$binary_tmp" "$binary" && mv -Tf -- "$manifest_tmp" "$manifest"; } &&
    systemctl --user restart "$unit" && verify_running "$new_digest" &&
    manifest_matches "$new_digest" && denied_get; then
    sampler=$(unit_state "$board_unit") || sampler=unknown
    session_release
    write_receipt promoted verified "$sha" "$digest" "$new_digest" "$old_sha" "$old_binary"
    prune_stage
    printf 'Promoted dev %s (%s); binary %s; previous %s preserved; running digest verified; direct GET refused\n' \
      "$version" "$sha" "$new_digest" "$old_binary"
    test "$sampler" = inactive ||
      printf 'A board sample ran during the activation; the verified binary serves the next sample\n'
    exit 0
  fi
  rollback
}

# --- end guarded promotion helpers -----------------------------------------

fetch "https://api.github.com/repos/$repo/releases/tags/dev" "$work/release.json" 5242880
sha=$(jq -er 'select(.tag_name=="dev" and .prerelease and (.draft|not)) | .target_commitish | select(test("^[0-9a-f]{40}$"))' "$work/release.json")
if test "$mode" = promote; then
  # Unchanged and older sources are decided here, before any artifact download.
  promote_gate
fi
archive=$(jq -er --arg suffix "-dev.${sha:0:7}-linux-$arch.tar.gz" \
  '[.assets[]|select(.state=="uploaded" and (.name|startswith("ox-llm-v")) and (.name|endswith($suffix)))|.name] | select(length>0) | sort_by(sub("^ox-llm-v";"")|split("-")[0]|split(".")|map(tonumber)) | .[-1]' "$work/release.json")
version=${archive#ox-llm-}; version=${version%-linux-"$arch".tar.gz}
[[ $version =~ ^v[0-9]+\.[0-9]+\.[0-9]+[0-9A-Za-z.+-]*-dev\.[0-9a-f]{7}$ ]]
download() {
  local url
  url=$(jq -er --arg name "$1" '[.assets[]|select(.state=="uploaded" and .name==$name)|.browser_download_url] | select(length==1) | .[0]' "$work/release.json")
  case "$url" in "https://github.com/$repo/releases/download/dev/"*) ;; *) return 1 ;; esac
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
tar -tzf "$work/$archive" >"$work/entries"
! grep -Ev '^usr(/|$)' "$work/entries"
! grep -E '(^|/)\.\.(/|$)' "$work/entries"
tar -xzf "$work/$archive" -C "$work/payload" --no-same-owner --no-same-permissions
{ test -x "$work/payload/usr/bin/oxa" && test ! -L "$work/payload/usr/bin/oxa"; } || exit 1
# The archive digest never describes the executable: the extracted bytes get
# their own digest, kept beside the release for activation and rollback.
binary_digest=$(digest_of "$work/payload/usr/bin/oxa")
printf '%s  %s\n' "$binary_digest" usr/bin/oxa >"$work/payload/binary.sha256"
jq -n --arg sha "$sha" --arg version "$version" --arg digest "$digest" \
  --arg time "$(date -u +%FT%TZ)" --arg archive "$archive" \
  '{channel:"dev",source_sha:$sha,version:$version,archive:$archive,sha256:$digest,installed_at:$time}' \
  >"$work/payload/receipt.json"
cp "$work/$bundle" "$work/payload/provenance.jsonl"
if test -e "$prefix/releases/$id"; then
  diff --no-dereference -qr "$work/payload/usr" "$prefix/releases/$id/usr"
  installed_manifest=$prefix/releases/$id/binary.sha256
  if test -e "$installed_manifest" || test -L "$installed_manifest"; then
    read_manifest "$installed_manifest" && test "$manifest_digest" = "$binary_digest"
  else
    installed_manifest_tmp=$(replacement_temp "$installed_manifest")
    cp -- "$work/payload/binary.sha256" "$installed_manifest_tmp"
    chmod 600 -- "$installed_manifest_tmp"
    mv -Tf -- "$installed_manifest_tmp" "$installed_manifest"
  fi
else
  mv "$work/payload" "$prefix/releases/$id"
fi
old=$(readlink "$prefix/current" || true)
test ! -e "$prefix/current" || test -L "$prefix/current"
# On a first promotion, retain digest-bound provenance for the executable that
# is still live before current moves to the downloaded candidate.  A busy
# activation lock can then defer safely and the next invocation still has a
# source baseline for the unchanged running binary.
if test "$mode" = promote && test "$rollout_baseline" = 0; then
  write_receipt retained verified "$old_sha" "$old_archive" "$old_binary" "$old_sha" "$old_binary"
  rollout_baseline=1
fi
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
if test "$mode" = promote; then
  activate
fi
printf 'Installed dev %s (%s); receipt: %s/current/receipt.json\n' "$version" "$sha" "$prefix"
