#!/usr/bin/env python3
"""Check OA source-module dependency direction.

The checker is deliberately independent of the build system. It scans direct
``#include <Oa/Module/...>`` dependencies, applies the canonical target graph,
and treats the v0.6.98 violations as a ratcheted numeric baseline. Existing
debt may shrink; a new forbidden edge or growth of an existing edge fails.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]Oa/([^/]+)/([^>"]+)[>"]')
SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".inl"})


@dataclass(frozen=True, order=True)
class IncludeSite:
    source_module: str
    target_module: str
    path: str
    line: int
    include: str

    @property
    def edge(self) -> str:
        return f"{self.source_module}->{self.target_module}"


@dataclass
class CheckResult:
    sites: list[IncludeSite]
    errors: list[str]
    notices: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("version") != 1:
        raise ValueError(f"{path}: unsupported or missing version")
    return value


def _iter_sources(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES:
            yield path


def scan(repo: Path, config: dict) -> tuple[list[IncludeSite], list[str]]:
    modules = set(config["modules"])
    sites: list[IncludeSite] = []
    errors: list[str] = []

    for root_name in config["source_roots"]:
        root = repo / root_name
        if not root.is_dir():
            errors.append(f"missing source root: {root_name}")
            continue

        for path in _iter_sources(root):
            relative = path.relative_to(root)
            if len(relative.parts) < 2:
                continue
            source_module = relative.parts[0]
            if source_module not in modules:
                errors.append(f"unknown source module {source_module}: {path.relative_to(repo)}")
                continue

            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line_number, line in enumerate(stream, 1):
                    match = INCLUDE_RE.match(line)
                    if not match:
                        continue
                    target_module = match.group(1)
                    if target_module not in modules:
                        errors.append(
                            f"unknown target module {target_module}: "
                            f"{path.relative_to(repo)}:{line_number}"
                        )
                        continue
                    if target_module == source_module:
                        continue
                    sites.append(
                        IncludeSite(
                            source_module=source_module,
                            target_module=target_module,
                            path=path.relative_to(repo).as_posix(),
                            line=line_number,
                            include=f"Oa/{target_module}/{match.group(2)}",
                        )
                    )

    return sorted(sites), errors


def check(repo: Path, config_path: Path, baseline_path: Path) -> CheckResult:
    config = _load_json(config_path)
    baseline = _load_json(baseline_path)
    sites, errors = scan(repo, config)
    modules = config["modules"]
    exception_limits: dict[str, int] = baseline["exceptions"]
    counts = Counter(site.edge for site in sites)
    notices: list[str] = []

    for edge, limit in sorted(exception_limits.items()):
        source, separator, target = edge.partition("->")
        if not separator or source not in modules or target not in modules:
            errors.append(f"invalid baseline edge: {edge}")
            continue
        if target in modules[source]["allows"]:
            errors.append(f"baseline edge is allowed by manifest and must be removed: {edge}")
        current = counts.get(edge, 0)
        if current < limit:
            notices.append(f"baseline can shrink: {edge} {limit} -> {current}")

    for edge, count in sorted(counts.items()):
        source, target = edge.split("->", 1)
        if target in modules[source]["allows"]:
            continue
        limit = exception_limits.get(edge)
        if limit is None:
            errors.append(f"new forbidden dependency: {edge} ({count} includes)")
        elif count > limit:
            errors.append(f"dependency baseline exceeded: {edge} {count} > {limit}")

    return CheckResult(sites=sites, errors=sorted(set(errors)), notices=notices)


def _report_text(result: CheckResult, config: dict, baseline: dict) -> str:
    counts = Counter(site.edge for site in result.sites)
    lines = ["OA module dependency check"]
    for edge, count in sorted(counts.items()):
        source, target = edge.split("->", 1)
        allowed = target in config["modules"][source]["allows"]
        state = "allowed" if allowed else f"baseline {baseline['exceptions'].get(edge, 0)}"
        lines.append(f"  {edge:<28} {count:>4}  {state}")
    for notice in result.notices:
        lines.append(f"NOTICE: {notice}")
    for error in result.errors:
        lines.append(f"ERROR: {error}")
    lines.append("PASS" if result.ok else "FAIL")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo", type=Path, default=default_repo)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    tool_dir = Path(__file__).resolve().parent
    config_path = (args.config or tool_dir / "modules.json").resolve()
    baseline_path = (args.baseline or tool_dir / "dependency_baseline.json").resolve()

    try:
        config = _load_json(config_path)
        baseline = _load_json(baseline_path)
        result = check(repo, config_path, baseline_path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"oacheck: {error}", file=sys.stderr)
        return 2

    if args.format == "json":
        counts = Counter(site.edge for site in result.sites)
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "edges": dict(sorted(counts.items())),
                    "notices": result.notices,
                    "errors": result.errors,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_report_text(result, config, baseline))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
