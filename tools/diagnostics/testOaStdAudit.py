#!/usr/bin/env python3
"""Regression tests for OA's standard-library dependency ratchet."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

import oaStdAudit  # noqa: E402


class OaStdAuditTest(unittest.TestCase):
	def testStripCommentsStringsAndRawStrings(self) -> None:
		text = (
			"std::vector<int> live; // std::string hidden\n"
			"// #include <thread>\n"
			"/* #include <filesystem> */\n"
			'const char* value = "std::mutex hidden";\n'
			'const char* raw = R"tag(std::thread hidden)tag";\n'
			"/* std::filesystem hidden */ std::move(live);\n"
		)
		code = oaStdAudit.stripCommentsAndLiterals(text)
		self.assertEqual(["vector", "move"], oaStdAudit.STD_SYMBOL_RE.findall(code))
		self.assertEqual([], oaStdAudit.INCLUDE_RE.findall(code))

	def testReportClassifiesPublicVendorAndGeneratedSources(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			public = root / "source/cpp/include/oa/core/value.h"
			generated = root / "source/cpp/lib/oa/core/gen/value.inl"
			vendor = root / "source/cpp/thirdparty/lib/value.hpp"
			for path in (public, generated, vendor):
				path.parent.mkdir(parents=True, exist_ok=True)
			public.write_text("#include <vector>\nstd::vector<int> value;\n")
			generated.write_text("std::move(value);\n")
			vendor.write_text("#include <mutex>\nstd::mutex value;\n")

			report = oaStdAudit.buildReport(root)

		self.assertEqual(1, report["summary"]["public"]["totals"]["stdSymbols"])
		self.assertEqual(1, report["summary"]["generated"]["totals"]["stdSymbols"])
		self.assertEqual(1, report["summary"]["vendor"]["totals"]["stdSymbols"])
		self.assertEqual(1, report["summary"]["vendor"]["totals"]["stdIncludes"])

	def testPolicyAllowsRemovalAndRejectsEveryIncrease(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			header = root / "source/cpp/include/oa/core/value.h"
			header.parent.mkdir(parents=True)
			header.write_text("#include <vector>\nstd::vector<int> value;\n")
			policy = oaStdAudit.buildPolicy(oaStdAudit.buildReport(root))

			header.write_text("int value;\n")
			self.assertEqual([], oaStdAudit.checkPolicy(oaStdAudit.buildReport(root), policy))

			header.write_text("#include <vector>\nstd::vector<int> a; std::vector<int> b;\n")
			errors = oaStdAudit.checkPolicy(oaStdAudit.buildReport(root), policy)
			self.assertTrue(any("stdSymbols vector increased 1 -> 2" in error for error in errors))

			header.write_text("#include <deque>\nstd::deque<int> value;\n")
			errors = oaStdAudit.checkPolicy(oaStdAudit.buildReport(root), policy)
			self.assertTrue(any("stdSymbols deque increased 0 -> 1" in error for error in errors))
			self.assertTrue(any("stdIncludes deque increased 0 -> 1" in error for error in errors))

	def testNewFilesBeginWithZeroAllowance(self) -> None:
		with tempfile.TemporaryDirectory() as temp:
			root = Path(temp)
			base = root / "source/cpp/include/oa/core/base.h"
			base.parent.mkdir(parents=True)
			base.write_text("int base;\n")
			policy = oaStdAudit.buildPolicy(oaStdAudit.buildReport(root))
			newFile = root / "source/cpp/lib/oa/core/new.cpp"
			newFile.parent.mkdir(parents=True)
			newFile.write_text("std::string value;\n")

			errors = oaStdAudit.checkPolicy(oaStdAudit.buildReport(root), policy)
		self.assertTrue(any("new.cpp: stdSymbols string increased 0 -> 1" in error for error in errors))


if __name__ == "__main__":
	unittest.main()
