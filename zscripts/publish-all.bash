#!/bin/bash
#
# Publishes every crate, in dependency order. `cargo publish` is run with the
# default features: feature selection only affects the verification build, never
# the published artifact, and enabling `intel_mkl`/`umfpack`/`mumps` would make a
# release depend on those libraries being installed on this machine.
set -euo pipefail

if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is required to wait for the crates.io index" >&2
    exit 1
fi

# Polls the sparse index until `crate` at `version` is visible.
#
# The next `cargo publish` resolves its dependencies against that index, so a
# stale entry fails the build of the crate that depends on this one. Polling
# replaces a fixed sleep, which was both slower than needed and not always long
# enough for the index to catch up.
wait_for_index() {
    local crate="$1" version="$2" path url body
    case ${#crate} in
        1) path="1/$crate" ;;
        2) path="2/$crate" ;;
        3) path="3/${crate:0:1}/$crate" ;;
        *) path="${crate:0:2}/${crate:2:2}/$crate" ;;
    esac
    url="https://index.crates.io/$path"
    echo "waiting for $crate $version on the index"
    for _ in $(seq 1 60); do
        if body=$(curl -fsSL --max-time 10 "$url" 2>/dev/null); then
            if grep -q "\"vers\":\"$version\"" <<<"$body"; then
                echo "$crate $version is on the index"
                return 0
            fi
        fi
        sleep 2
    done
    echo "error: $crate $version did not reach $url within two minutes" >&2
    echo "the package was uploaded; rerun the remaining crates once the index catches up" >&2
    return 1
}

# Publishes one crate and waits until the index serves the new version.
publish() {
    local crate="$1" version
    version=$(grep -m1 '^version = "' "$crate/Cargo.toml" | cut -d'"' -f2)
    ( cd "$crate" && cargo publish )
    wait_for_index "$crate" "$version"
}

# Dependency order: every crate is published after everything it depends on.
publish russell_lab
publish russell_sparse
publish russell_stat
publish russell_tensor
publish russell_pde
publish russell_ode
publish russell_nonlin

echo
echo "All done!"
