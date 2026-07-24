"""Locate OA's private nanobind extension in wheels and source builds."""

from importlib import import_module


try:
	native = import_module("oa._oa")
except ImportError:
	# CMake development builds place `_oa` in the build directory. Tutorial and
	# test helpers add that directory to sys.path, while wheels install it inside
	# this package.
	try:
		native = import_module("_oa")
	except ImportError as build_error:
		raise ImportError(
			"OA's native extension is unavailable. Build target `_oa`, install "
			"the oapython wheel, or add the OA build directory to PYTHONPATH."
		) from build_error


__all__ = ["native"]
