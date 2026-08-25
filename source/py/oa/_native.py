"""Locate OA's private nanobind extension in wheels and source builds."""

from importlib import import_module as importModule
import os


if os.getenv("OA_PYTHON_BUILD_DIR"):
	# An explicit development-build selection must win over an editable or wheel
	# installation already present in the interpreter environment.
	try:
		native = importModule("_oa")
	except ModuleNotFoundError as buildError:
		if buildError.name != "_oa":
			raise
		raise ImportError(
			"OA_PYTHON_BUILD_DIR is set, but its `_oa` extension is unavailable. "
			"Build target `_oa` and add that directory to PYTHONPATH."
		) from buildError
	except ImportError as buildError:
		raise ImportError(
			"OA's development native extension exists but could not be loaded: "
			f"{buildError}"
		) from buildError
else:
	try:
		native = importModule("oa._oa")
	except ModuleNotFoundError as packageError:
		if packageError.name != "oa._oa":
			raise
		# CMake development builds place `_oa` in the build directory. Tutorial
		# helpers may add that directory to sys.path without setting the explicit
		# test override, while wheels install it inside this package.
		try:
			native = importModule("_oa")
		except ModuleNotFoundError as buildError:
			if buildError.name != "_oa":
				raise
			raise ImportError(
				"OA's native extension is unavailable. Build target `_oa`, install "
				"the oapython wheel, or add the OA build directory to PYTHONPATH."
			) from buildError
		except ImportError as buildError:
			raise ImportError(
				"OA's development native extension exists but could not be loaded: "
				f"{buildError}"
			) from buildError
	except ImportError as packageError:
		raise ImportError(
			"OA's packaged native extension exists but could not be loaded: "
			f"{packageError}"
		) from packageError


__all__ = ["native"]
