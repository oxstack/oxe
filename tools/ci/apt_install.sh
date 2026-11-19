#!/usr/bin/env bash
# Install apt packages on a hosted runner without failing on a mirror blip. The Ubuntu mirrors
# answer "Connection failed" / 503 now and then (exit 100); that is a transient network error,
# not a property of the change under test. Three attempts with backoff; the last one points
# apt at the Azure mirror, which the runner images are built from. A package that cannot be
# installed after that still fails the step. Callers keep their own check that the tool works.
set -euo pipefail
[ "$#" -ge 1 ] || { echo 'usage: apt_install.sh PACKAGE...' >&2; exit 2; }
opts=(-o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
case "$(dpkg --print-architecture)" in
  amd64 | i386) azure=http://azure.archive.ubuntu.com/ubuntu ;;
  *) azure=http://azure.ports.ubuntu.com/ubuntu-ports ;;
esac

use_azure_mirror() {
  local f
  for f in /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list; do
    [ -f "$f" ] || continue
    sudo sed -i -E "s#https?://([a-z]{2}\.)?(archive|security)\.ubuntu\.com/ubuntu/?#$azure/#g; s#https?://ports\.ubuntu\.com/ubuntu-ports/?#$azure/#g; s#^(URIs:[[:space:]]*)mirror\+file:/etc/apt/apt-mirrors.txt#\1$azure/#; s#^(deb[^#]*[[:space:]])mirror\+file:/etc/apt/apt-mirrors.txt#\1$azure/#" "$f"
  done
}

for attempt in 1 2 3; do
  if [ "$attempt" -eq 3 ]; then
    echo "::warning::apt mirrors failed twice; retrying through $azure"
    use_azure_mirror
  fi
  if sudo apt-get "${opts[@]}" update -q \
    && sudo apt-get "${opts[@]}" install -y -q --no-install-recommends --fix-missing "$@"; then
    exit 0
  fi
  [ "$attempt" -eq 3 ] && break
  delay=$((attempt * 20))
  echo "::warning::apt attempt $attempt failed; retrying in ${delay}s"
  sleep "$delay"
done
echo "::error::apt-get could not install: $*"
exit 1
