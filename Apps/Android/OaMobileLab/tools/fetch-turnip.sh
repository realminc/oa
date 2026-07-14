#!/usr/bin/env bash
set -euo pipefail

readonly version="26.1.4"
readonly archive_name="mesa-turnip-android-${version}.zip"
readonly archive_sha256="a559b4257d7964e8082d1bfcfc3fb77ea95cd1f8071ee6e0b3e6c0d859539fa0"
readonly driver_sha256="5a9bdaa51e31c4579dfea7217039bca65d99ff9aeee503f55860939039c4f043"
readonly url="https://github.com/nihui/mesa-turnip-android-driver/releases/download/${version}/${archive_name}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
readonly repo_root
readonly output_dir="${repo_root}/Build/Android/OaMobileLab/Assets/turnip"
readonly output="${output_dir}/libvulkan_freedreno.so"

if [[ -f "${output}" ]] && echo "${driver_sha256}  ${output}" | sha256sum --check --status; then
    exit 0
fi

mkdir -p "${output_dir}"
work_dir="$(mktemp -d)"
trap 'rm -rf -- "${work_dir}"' EXIT

archive="${work_dir}/${archive_name}"
curl --fail --location --show-error --silent "${url}" --output "${archive}"
echo "${archive_sha256}  ${archive}" | sha256sum --check --status
unzip -p "${archive}" libvulkan_freedreno.so > "${output}.tmp"
echo "${driver_sha256}  ${output}.tmp" | sha256sum --check --status
mv "${output}.tmp" "${output}"

echo "Pinned Mesa Turnip ${version} installed at ${output}"
