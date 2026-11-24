#!/usr/bin/env bash
set -euo pipefail

profile="debug"
declare -a cargo_args=()
declare -a crates=()

discover_plugin_crates() {
    find crates/plugins \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name 'riscv-fuzz-core-*' \
        -print0 \
        | while IFS= read -r -d '' crate_dir; do
            manifest="${crate_dir}/Cargo.toml"
            if [[ -f "${manifest}" ]] && awk '
                BEGIN {
                    in_lib = 0
                    in_crate_type = 0
                    found = 0
                }
                /^\[/ {
                    in_lib = ($0 == "[lib]")
                    in_crate_type = 0
                }
                in_lib && /^[[:space:]]*crate-type[[:space:]]*=/ {
                    in_crate_type = 1
                }
                in_lib && in_crate_type && /"cdylib"/ {
                    found = 1
                }
                in_crate_type && /\]/ {
                    in_crate_type = 0
                }
                END {
                    exit(found ? 0 : 1)
                }
            ' "${manifest}"; then
                basename "${crate_dir}"
            fi
        done \
        | sort
}

while (($# > 0)); do
    case "$1" in
        --release)
            profile="release"
            cargo_args+=(--release)
            shift
            ;;
        *)
            crates+=("$1")
            shift
            ;;
    esac
done

if ((${#crates[@]} == 0)); then
    while IFS= read -r crate; do
        crates+=("${crate}")
    done < <(discover_plugin_crates)
fi

if ((${#crates[@]} == 0)); then
    echo "no plugin crates found under crates/plugins/riscv-fuzz-core-*" >&2
    exit 1
fi

case "$(uname -s)" in
    Linux)
        dyn_prefix="lib"
        dyn_ext="so"
        ;;
    Darwin)
        dyn_prefix="lib"
        dyn_ext="dylib"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        dyn_prefix=""
        dyn_ext="dll"
        ;;
    *)
        echo "unsupported host OS: $(uname -s)" >&2
        exit 1
        ;;
esac

echo "building plugin crates (${profile})..."
for crate in "${crates[@]}"; do
    cargo build "${cargo_args[@]}" -p "${crate}"
done

out_dir="riscv_fuzz_plugins"
mkdir -p "${out_dir}"

echo "staging plugin libraries into ${out_dir}/ ..."
for crate in "${crates[@]}"; do
    artifact_name="${dyn_prefix}${crate//-/_}.${dyn_ext}"
    artifact_path="target/${profile}/${artifact_name}"
    if [[ ! -f "${artifact_path}" ]]; then
        echo "missing plugin artifact: ${artifact_path}" >&2
        exit 1
    fi
    cp "${artifact_path}" "${out_dir}/"
done

echo
echo "done."
echo "dynamic plugins staged in ${out_dir}/"
echo "runtime can auto-discover plugins from ${out_dir}/"
echo "or you can set:"
echo "  export RISCV_FUZZ_PLUGIN_DIRS=\"\$PWD/${out_dir}\""
