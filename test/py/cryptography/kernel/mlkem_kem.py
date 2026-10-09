"""Independent mlkem kem proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import hashlib
import json
import random
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest, ROOT
else:
    from .._pqc_harness import SlangComponentTest, ROOT

class MlkemKemTests(SlangComponentTest):
    def test_mlkem_nist_vectors(self):
        folder = ROOT / "test/fixtures/cryptography/mlkem"
        counts = {"keyGen": 0, "encapsulation": 0, "decapsulation": 0, "keyCheck": 0}
        for mode in ("keyGen", "encapDecap"):
            prompt = json.loads((folder / f"{mode}-prompt.json").read_text())
            result = json.loads((folder / f"{mode}-expectedResults.json").read_text())
            results = {(g["tgId"], t["tcId"]): t for g in result["testGroups"] for t in g["tests"]}
            for group in prompt["testGroups"]:
                k = {"ML-KEM-512": 2, "ML-KEM-768": 3, "ML-KEM-1024": 4}[group["parameterSet"]]
                function = group.get("function", "keyGen")
                values, expected = [], []
                for case in group["tests"]:
                    answer = results[(group["tgId"], case["tcId"])]
                    def data(name, source=case):
                        return list(bytes.fromhex(source[name]))
                    def padded(name, capacity, source=case):
                        value = data(name, source)
                        self.assertLessEqual(len(value), capacity)
                        return value + [0xA5] * (capacity - len(value))
                    if function == "keyGen":
                        d = bytes.fromhex(case["d"])
                        values += [k] + [int.from_bytes(d[i:i+4], "little") for i in range(0, 32, 4)] + data("z")
                        expected += padded("ek", 1568, answer) + padded("dk", 3168, answer)
                        entry, width = "mlkem_keygen_components", 4736
                    elif function == "encapsulation":
                        values += [k] + padded("ek", 1568) + data("m")
                        expected += padded("c", 1568, answer) + data("k", answer)
                        entry, width = "mlkem_encaps_components", 1600
                    elif function == "decapsulation":
                        values += [k] + padded("dk", 3168) + padded("c", 1568)
                        expected += data("k", answer)
                        entry, width = "mlkem_kem_decaps_components", 32
                    else:
                        is_ek = function == "encapsulationKeyCheck"
                        ek = data("ek") if is_ek else [0xA5] * 1568
                        dk = data("dk") if not is_ek else [0xA5] * 3168
                        values += [k, len(ek) if is_ek else 0, len(dk) if not is_ek else 0, 0]
                        values += ek + [0xA5] * (1568 - len(ek)) + dk + [0xA5] * (3168 - len(dk))
                        expected += [int(answer["testPassed"]) if is_ek else 0,
                                     int(answer["testPassed"]) if not is_ek else 0, 0]
                        entry, width = "mlkem_key_check_components", 3
                    counts[function if function in counts else "keyCheck"] += 1
                with self.subTest(function=function, parameter=k):
                    self.assertEqual(self.execute(entry, values, len(group["tests"]), width), expected)
        self.assertEqual(counts, {"keyGen": 75, "encapsulation": 75, "decapsulation": 30, "keyCheck": 60})


    def test_mlkem_input_boundaries(self):
        values, expected = [], []
        for k in (2, 3, 4):
            ek = [0] * (384 * k + 32)
            dk = [0] * (768 * k + 96)
            dk[384*k:768*k+32] = ek
            dk[768*k+32:768*k+64] = hashlib.sha3_256(bytes(ek)).digest()
            size = 32 * (11*k + 5 if k == 4 else 10*k + 4)
            cases = [(k, len(ek), len(dk), size, ek, dk, [1, 1, 1])]
            for offset in (-1, 1):
                cases += [(k, len(ek)+offset, len(dk)+offset, size+offset, ek, dk, [0, 0, 0])]
            for invalid_k in (0, 1, 5, 0xFFFFFFFF):
                cases += [(invalid_k, len(ek), len(dk), size, ek, dk, [0, 0, 0])]
            # Every coefficient position rejects q and accepts q-1, including
            # both packed nibbles and the final polynomial coefficient.
            for coefficient in range(256*k):
                for value in (3328, 3329):
                    packed = (value << (12*coefficient)).to_bytes(384*k, "little")
                    altered = list(packed) + [0]*32
                    cases += [(k, len(ek), len(dk), size, altered, dk, [int(value < 3329), 1, 1])]
            # A byte-valued uint is checked even in rho, secret and z.
            for index in (0, len(ek)-1):
                altered = ek.copy(); altered[index] = 256
                cases += [(k, len(ek), len(dk), size, altered, dk, [0, 1, 1])]
            for index in (0, 384*k, 768*k+32, len(dk)-1):
                altered = dk.copy(); altered[index] ^= 256 if index in (0, len(dk)-1) else 1
                cases += [(k, len(ek), len(dk), size, ek, altered, [1, 0, 1])]
            # The FIPS dk hash check does not validate secret coefficient range.
            altered = dk.copy(); altered[0:384*k] = [255] * (384*k)
            cases += [(k, len(ek), len(dk), size, ek, altered, [1, 1, 1])]
            for parameter, el, dl, cl, public, secret, flags in cases:
                values += [parameter, el, dl, cl] + public + [0xA5] * (1568-len(public))
                values += secret + [0xA5] * (3168-len(secret))
                expected += flags
        self.assertEqual(self.execute("mlkem_key_check_components", values, len(expected)//3, 3), expected)


    def test_mlkem_kem(self):
        if __package__ == "kernel":
            from _mlkem_oracle import kem, decaps
        else:
            from .._mlkem_oracle import kem, decaps
        rng = random.Random(2031618)
        values, expected, rejected_inputs, rejected_expected = [], [], [], []
        for k in (2, 3, 4):
            for case in range(2):
                d = bytes(32) if case == 0 else rng.randbytes(32)
                z = bytes([255] * 32) if case == 0 else rng.randbytes(32)
                message = bytes(32) if case == 0 else rng.randbytes(32)
                ek, dk, ciphertext, key = kem(d, z, message, k)
                values += [k] + [int.from_bytes(d[i:i+4], "little") for i in range(0, 32, 4)]
                values += list(z + message)
                expected += list(ek) + [0xA5] * (1568 - len(ek))
                expected += list(dk) + [0xA5] * (3168 - len(dk))
                expected += list(ciphertext) + [0xA5] * (1568 - len(ciphertext))
                expected += list(key + key) + [1]
                for position in (0, len(ciphertext)//2, len(ciphertext)-1):
                    altered = bytearray(ciphertext)
                    altered[position] ^= 128
                    altered = bytes(altered)
                    oracle = decaps(dk, altered, k)
                    self.assertEqual(oracle, hashlib.shake_256(z + altered).digest(32))
                    rejected_inputs += [k] + list(dk) + [0xA5] * (3168 - len(dk))
                    rejected_inputs += list(altered) + [0xA5] * (1568 - len(altered))
                    rejected_expected += list(oracle)
        self.assertEqual(self.execute("mlkem_kem_components", values, 6, 6369), expected)
        self.assertEqual(self.execute("mlkem_kem_decaps_components", rejected_inputs, 18, 32), rejected_expected)



if __name__ == "__main__":
    unittest.main()
