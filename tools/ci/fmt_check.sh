#!/bin/sh
# Checkout-local done check. Hosted CI selects its workspace and runs Cargo directly.
set -eu

is_workspace() {
    [ -f "$1/Cargo.toml" ] && grep -q '^\[workspace\]' "$1/Cargo.toml"
}

# A physical caller path keeps symlinked members in their own workspace. A standalone
# package owns its check. Other callers use the helper checkout's oxa (or sole) workspace.
helper=$0
case $helper in
    */*) ;;
    *) helper=$(command -v "$helper") ;;
esac
checkout=$(CDPATH= cd -P -- "${helper%/*}/../.." && pwd -P)
dir=$(pwd -P)
workspace=
package=
while :; do
    if is_workspace "$dir"; then workspace=$dir; break; fi
    if [ -z "$package" ] && [ -f "$dir/Cargo.toml" ] && grep -q '^\[package\]' "$dir/Cargo.toml"; then
        package=$dir
    fi
    [ "$dir" = / ] && break
    dir=${dir%/*}
    [ -n "$dir" ] || dir=/
done
workspace=${workspace:-$package}
if [ -z "$workspace" ]; then
    if is_workspace "$checkout/oxa"; then
        workspace=$checkout/oxa
    else
        for dir in "$checkout"/*; do
            is_workspace "$dir" || continue
            if [ -n "$workspace" ]; then
                echo 'fmt_check: several workspaces beside this helper; choose a caller workspace' >&2
                exit 1
            fi
            workspace=$dir
        done
    fi
fi
[ -n "$workspace" ] || {
    echo 'fmt_check: no Cargo.toml declaring a workspace for this caller or helper' >&2
    exit 1
}

account=
if command -v getent >/dev/null 2>&1 && command -v id >/dev/null 2>&1; then
    account=$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6) || account=
fi
rustup_roots() {
    for root in "${RUSTUP_HOME:-}" "${HOME:+$HOME/.rustup}" "${account:+$account/.rustup}"; do
        [ -n "$root" ] && printf '%s\n' "$root"
    done
    return 0
}
# Prefer an explicit/default selection before considering other installations.
preferred_dirs() {
    rustup_roots | while IFS= read -r root; do
        selected=${RUSTUP_TOOLCHAIN:-}
        if [ -z "$selected" ] && [ -f "$root/settings.toml" ]; then
            while IFS= read -r line; do
                case $line in
                    default_toolchain*\"*)
                        selected=${line#*\"}
                        selected=${selected%%\"*}
                        break ;;
                esac
            done < "$root/settings.toml"
        fi
        if [ -n "$selected" ]; then
            selected=$root/toolchains/$selected
            if [ ! -d "$selected/bin" ]; then
                case ${selected##*/} in
                    stable|beta|nightly|nightly-[0-9]*|[0-9]*)
                        set -- "$selected"-*/bin
                        if [ "$#" -eq 1 ] && [ -d "$1" ]; then selected=${1%/bin}; fi
                        ;;
                esac
            fi
            printf '%s\n' "$selected/bin"
        fi
    done
    return 0
}
preferred=$(preferred_dirs)
dirs=$(
    printf '%s\n' "$preferred"
    for dir in "${CARGO_HOME:+$CARGO_HOME/bin}" "${HOME:+$HOME/.cargo/bin}" "${account:+$account/.cargo/bin}"; do
        [ -n "$dir" ] && printf '%s\n' "$dir"
    done
    rustup_roots | while IFS= read -r root; do
        for dir in "$root"/toolchains/*/bin; do
            [ -d "$dir" ] && printf '%s\n' "$dir"
        done
        :
    done
)
cargo_path=$(command -v cargo 2>/dev/null || true)
if [ -z "$cargo_path" ] && command -v rustup >/dev/null 2>&1; then
    cargo_path=$(rustup which cargo 2>/dev/null || true)
    [ -x "$cargo_path" ] || cargo_path=
fi
if [ -z "$cargo_path" ]; then
    search=${preferred:-$dirs}
    fallback=
    while IFS= read -r dir; do
        [ -x "$dir/cargo" ] || continue
        [ -n "$fallback" ] || fallback=$dir/cargo
        if [ -x "$dir/cargo-fmt" ] && [ -x "$dir/rustfmt" ]; then
            cargo_path=$dir/cargo
            break
        fi
    done <<EOF
$search
EOF
    cargo_path=${cargo_path:-$fallback}
    [ -n "$cargo_path" ] || {
        echo 'fmt_check: cargo not found for the selected or installed Rust toolchain' >&2
        exit 127
    }
    PATH=${cargo_path%/*}:${PATH:-}
fi
# Remember cargo itself before adding companions: another installation must not shadow it.
case $cargo_path in
    /*) ;;
    */*) cargo_path="$(CDPATH= cd -P -- "${cargo_path%/*}" && pwd -P)/${cargo_path##*/}" ;;
    *) cargo_path="$(pwd -P)/$cargo_path" ;;
esac
for name in cargo-fmt rustfmt; do
    command -v "$name" >/dev/null 2>&1 && continue
    found=
    while IFS= read -r dir; do
        if [ -x "$dir/$name" ]; then
            PATH=${PATH:+$PATH:}$dir
            found=$dir
            break
        fi
    done <<EOF
${cargo_path%/*}
$dirs
EOF
    [ -n "$found" ] || {
        echo "fmt_check: $name not found on PATH or in an installed Rust toolchain" >&2
        exit 127
    }
done
export PATH
cd "$workspace"
exec "$cargo_path" fmt --check "$@"
