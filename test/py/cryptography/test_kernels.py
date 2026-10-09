"""One discovery entry point for the production Slang component proofs.

Component modules deliberately omit the test_ prefix so recursive discovery
cannot run them again. Each module also supports focused unittest invocation.
"""

import importlib
import unittest


COMPONENTS = (
    "dispatch",
    "keccak",
    "mldsa_field",
    "mldsa_keygen",
    "mldsa_ntt",
    "mldsa_packing",
    "mldsa_prehash_digest",
    "mldsa_sampling",
    "mldsa_sign",
    "mldsa_verification",
    "mlkem_compression",
    "mlkem_field",
    "mlkem_kem",
    "mlkem_kpke",
    "mlkem_ntt",
    "mlkem_packing",
    "mlkem_sampling",
    "sha2",
    "sha2_pair",
    "sha3",
    "shake",
)


def load_tests(loader, tests, pattern):
    """Collect component cases using the caller's loader (including -k filters)."""
    prefix = f"{__package__}." if __package__ else ""
    suite = unittest.TestSuite()
    for component in COMPONENTS:
        module = importlib.import_module(f"{prefix}kernel.{component}")
        suite.addTests(loader.loadTestsFromModule(module))
    return suite


if __name__ == "__main__":
    unittest.main()
