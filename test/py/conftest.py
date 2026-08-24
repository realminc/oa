"""Capability profiles and zero-skip enforcement for OA Python tests."""

from __future__ import annotations

import pytest

from oa_python_test import REPO_ROOT  # noqa: F401 - bootstraps source builds


_PROFILES = ("host", "gpu", "crypto-host", "crypto-gpu", "external", "complete")


def pytest_addoption(parser: pytest.Parser) -> None:
	group = parser.getgroup("oa")
	group.addoption(
		"--oa-profile",
		choices=_PROFILES,
		default="complete",
		help="select an OA Python capability profile",
	)
	group.addoption(
		"--oa-forbid-skips",
		action="store_true",
		help="fail the profile if a selected test skips at collection or runtime",
	)


def pytest_configure(config: pytest.Config) -> None:
	config.addinivalue_line("markers", "oa_gpu: requires an initialized Vulkan engine")
	config.addinivalue_line("markers", "oa_crypto: requires the optional crypto build")
	config.addinivalue_line("markers", "oa_external: requires an external Python package")


def pytest_collection_modifyitems(
	config: pytest.Config,
	items: list[pytest.Item],
) -> None:
	# Engine-backed tests declare the shared fixture. Keep that requirement next
	# to the test signature instead of duplicating dozens of decorators.
	for item in items:
		if "engine" in getattr(item, "fixturenames", ()):
			item.add_marker(pytest.mark.oa_gpu)

	profile = config.getoption("--oa-profile")
	selected: list[pytest.Item] = []
	deselected: list[pytest.Item] = []
	for item in items:
		gpu = item.get_closest_marker("oa_gpu") is not None
		crypto = item.get_closest_marker("oa_crypto") is not None
		external = item.get_closest_marker("oa_external") is not None
		matches = {
			"host": not gpu and not crypto and not external,
			"gpu": not crypto and not external,
			"crypto-host": crypto and not gpu and not external,
			"crypto-gpu": crypto and not external,
			"external": external,
			"complete": True,
		}[profile]
		(selected if matches else deselected).append(item)

	if deselected:
		config.hook.pytest_deselected(items=deselected)
	items[:] = selected


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
	if not session.config.getoption("--oa-forbid-skips"):
		return
	reporter = session.config.pluginmanager.get_plugin("terminalreporter")
	if reporter is not None and reporter.stats.get("skipped"):
		session.exitstatus = pytest.ExitCode.TESTS_FAILED
