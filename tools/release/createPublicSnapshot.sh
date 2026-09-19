#!/usr/bin/env bash
# Build a sanitized, version-translated public Git commit from private OARS.
#
# Usage:
#   tools/release/createPublicSnapshot.sh <source-ref> <public-version> [public-parent-ref]
#
# The script writes Git objects only. It does not change the working tree, index,
# branches, tags, or remotes. Pass public/main as the optional parent to keep the
# public history linear.

set -euo pipefail

if (( $# < 2 || $# > 3 )); then
	echo "usage: $0 <source-ref> <public-version> [public-parent-ref]" >&2
	exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

source_ref="$1"
public_version="${2#v}"
parent_ref="${3:-}"
source_commit="$(git rev-parse --verify "${source_ref}^{commit}")"

if [[ ! "$public_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
	echo "error: public version must be a stable X.Y.Z semantic version" >&2
	exit 1
fi

private_version="$(
	git show "${source_commit}:Cargo.toml" |
		python3 -c 'import sys, tomllib; print(tomllib.loads(sys.stdin.read())["package"]["version"])'
)"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
snapshot="$tmp_dir/snapshot"
tmp_index="$tmp_dir/index"
mkdir -p "$snapshot"
git archive "$source_commit" | tar -xf - -C "$snapshot"

# Private agent/editor configuration is never part of the public product tree.
# Architecture and release documentation and this reproducibility tool remain public.
private_paths=(
	.cursor
	.devin
	.vscode
	.zed
	AGENTS.md
)
for path in "${private_paths[@]}"; do
	rm -rf -- "${snapshot:?}/$path"
done

python3 - "$snapshot" "$private_version" "$public_version" <<'PY'
from __future__ import annotations

import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
private_version = sys.argv[2]
public_version = sys.argv[3]

direct_sources = (
    "Cargo.toml",
    "sdk/py/Cargo.toml",
    "sdk/py/pyproject.toml",
    "sdk/py/uv.lock",
    "docs/internal/porting/oaPython.md",
    "sdk/py/README.md",
)
for relative in direct_sources:
    path = root / relative
    text = path.read_text()
    updated = text.replace(private_version, public_version)
    if updated == text:
        raise SystemExit(f"version source did not contain {private_version}: {relative}")
    path.write_text(updated)

lock_path = root / "Cargo.lock"
lock = lock_path.read_text()
for package in ("oa", "oa-python"):
    pattern = re.compile(
        rf'(\[\[package\]\]\nname = "{re.escape(package)}"\nversion = ")'
        rf'{re.escape(private_version)}("\n)'
    )
    lock, count = pattern.subn(rf"\g<1>{public_version}\2", lock, count=1)
    if count != 1:
        raise SystemExit(f"expected one {package} {private_version} Cargo.lock row")
lock_path.write_text(lock)
PY

cat > "$snapshot/.oa-public-snapshot" <<EOF
OA sanitized public source snapshot
source_commit=$source_commit
private_version=$private_version
public_version=$public_version
EOF

export GIT_INDEX_FILE="$tmp_index"
export GIT_WORK_TREE="$snapshot"
git read-tree --empty
git add -A -f -- .
public_tree="$(git write-tree)"
public_paths="$(git ls-tree -r --name-only "$public_tree")"

for path in "${private_paths[@]}"; do
	if grep -Eq "^${path}(/|$)" <<<"$public_paths"; then
		echo "error: private path survived public snapshot: $path" >&2
		exit 1
	fi
done

if symlink_paths="$(git ls-tree -r "$public_tree" | awk '$1 == "120000" {print $4}')" \
	&& [[ -n "$symlink_paths" ]]; then
	printf '%s\n' "$symlink_paths" >&2
	echo "error: symlink survived public snapshot" >&2
	exit 1
fi

required_paths=(
	.oa-public-snapshot
	.github/workflows/ci.yml
	Cargo.lock
	Cargo.toml
	LICENSE
	NOTICE.md
	README.md
	sdk/py/pyproject.toml
	tools/release/createPublicSnapshot.sh
)
for path in "${required_paths[@]}"; do
	if ! grep -Fxq "$path" <<<"$public_paths"; then
		echo "error: required public path is missing: $path" >&2
		exit 1
	fi
done

if grep -Ei '(^|/)(\.env($|\.)|\.pypirc$|credentials?($|\.)|secrets?($|\.)|id_(rsa|dsa|ecdsa|ed25519)$)' \
	<<<"$public_paths" >&2; then
	echo "error: credential-shaped filename found in public snapshot" >&2
	exit 1
fi

secret_pattern='AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----|pypi-[A-Za-z0-9_-]{20,}'
if secret_hits="$(git grep -I -n -E "$secret_pattern" "$public_tree" -- 2>/dev/null)"; then
	printf '%s\n' "$secret_hits" >&2
	echo "error: secret-shaped content found in public snapshot" >&2
	exit 1
fi

workstation_pattern='(/home/[A-Za-z0-9._-]+/|C:\\Users\\[A-Za-z0-9._-]+\\|C:/Users/[A-Za-z0-9._-]+/)'
if workstation_hits="$(git grep -I -n -E "$workstation_pattern" "$public_tree" -- 2>/dev/null)"; then
	printf '%s\n' "$workstation_hits" >&2
	echo "error: personal workstation path found in public snapshot" >&2
	exit 1
fi

for version_source in Cargo.toml sdk/py/Cargo.toml sdk/py/pyproject.toml sdk/py/uv.lock; do
	if ! git show "${public_tree}:${version_source}" | grep -Fq "$public_version"; then
		echo "error: translated version missing from $version_source" >&2
		exit 1
	fi
done
if git grep -I -n -F "$private_version" "$public_tree" -- \
	Cargo.toml Cargo.lock sdk/py/Cargo.toml sdk/py/pyproject.toml sdk/py/uv.lock \
	>/dev/null 2>&1; then
	echo "error: private version survived in canonical public version sources" >&2
	exit 1
fi

commit_args=("$public_tree")
if [[ -n "$parent_ref" ]]; then
	parent_commit="$(git rev-parse --verify "${parent_ref}^{commit}")"
	commit_args+=( -p "$parent_commit" )
fi

message="OA v${public_version} public snapshot (OARS v${private_version})"
public_commit="$(printf '%s\n' "$message" | git commit-tree "${commit_args[@]}")"

echo "private source:  $source_commit (v$private_version)" >&2
echo "public version:  v$public_version" >&2
echo "public tree:     $public_tree" >&2
echo "public commit:   $public_commit" >&2
printf '%s\n' "$public_commit"
