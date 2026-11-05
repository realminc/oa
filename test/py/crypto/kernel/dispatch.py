"""Independent dispatch proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest
else:
    from .._pqc_harness import SlangComponentTest

class DispatchTests(SlangComponentTest):
    def test_mlkem_empty_dispatch(self) -> None:
        for entry, width in (("mlkem_field_components", 5),
                             ("mlkem_compression_components", 4),
                             ("mlkem_polynomial_components", 768),
                             ("mlkem_packing_components", 641),
                             ("mlkem_sampling_components", 512),
                             ("mlkem_rejection_components", 257),
                             ("mlkem_kpke_components", 4705),
                             ("mlkem_kpke_decrypt_components", 32),
                             ("mlkem_kem_components", 6369),
                             ("mlkem_kem_decaps_components", 32),
                             ("mlkem_keygen_components", 4736),
                             ("mlkem_encaps_components", 1600),
                             ("mlkem_key_check_components", 3)):
            self.assertEqual(self.execute(entry, [], 0, width), [])



if __name__ == "__main__":
    unittest.main()
