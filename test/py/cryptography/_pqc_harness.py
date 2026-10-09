"""FIPS 203/204 component oracles for the actual Slang implementation.

Slang's explicit host target is a test laboratory, never an OA CPU fallback.
These checks prove formulas and compile both Vulkan targets; they do not prove
GPU execution, full ML-DSA conformance, or constant-time machine code.
"""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import struct
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[3]
Q = 8380417
N = 256
MASK = (1 << 32) - 1


class BufferView(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("count", ctypes.c_size_t)]


class Parameters(ctypes.Structure):
    _fields_ = [("inputs", BufferView), ("outputs", BufferView)]


class VaryingInput(ctypes.Structure):
    _fields_ = [("start", ctypes.c_uint32 * 3), ("end", ctypes.c_uint32 * 3)]


def decompose(value: int, gamma2: int) -> tuple[int, int]:
    """Algorithm 36 using independent integer division, including q-1 wrap."""
    high = (value + gamma2 - 1) // (2 * gamma2)
    low = value - high * (2 * gamma2)
    if high * (2 * gamma2) == Q - 1:
        return 0, low - 1
    return high, low


def evaluate_ntt(poly: list[int]) -> list[int]:
    """Equation 7.1 via Horner evaluation, without butterfly/twiddle code."""
    result = []
    for i in range(N):
        reversed_bits = int(f"{i:08b}"[::-1], 2)
        root = pow(1753, 2 * reversed_bits + 1, Q)
        value = 0
        for coefficient in reversed(poly):
            value = (value * root + coefficient) % Q
        result.append(value)
    return result


def negacyclic_product(a: list[int], b: list[int]) -> list[int]:
    """Schoolbook multiplication in Z_q[X]/(X^256+1), independent of NTT."""
    result = [0] * N
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            if i + j < N:
                result[i + j] += left * right
            else:
                result[i + j - N] -= left * right
    return [value % Q for value in result]


# Independent hashlib/NIST identities for fixture oracles, not generated metadata.
MLDSA_PREHASH_HASHLIB = {
    "SHA2-256": (1, "sha256", 32), "SHA2-384": (2, "sha384", 48),
    "SHA2-512": (3, "sha512", 64), "SHA2-224": (4, "sha224", 28),
    "SHA2-512/224": (5, "sha512_224", 28), "SHA2-512/256": (6, "sha512_256", 32),
    "SHA3-224": (7, "sha3_224", 28), "SHA3-256": (8, "sha3_256", 32),
    "SHA3-384": (9, "sha3_384", 48), "SHA3-512": (10, "sha3_512", 64),
    "SHAKE-128": (11, "shake_128", 32), "SHAKE-256": (12, "shake_256", 64),
}


ENTRY_SOURCES = {
    "mldsa_prehash_digest_components": "mldsa/prehash_digest.slang",
    "sha2_u32_pair_components": "common/sha2_pair_digest.slang",
    "sha2_u32_pair_arithmetic_components": "common/sha2_pair_digest.slang",
    "sha2_u32_components": "common/sha2_digest.slang",
    "sha2_u32_length_components": "common/sha2_digest.slang",
    "sha3_components": "common/sha3_digest.slang",
    "sha3_range_components": "common/sha3_digest.slang",
    "mldsa_sign_workspace_components": "mldsa/sign_workspace.slang",
    "hash_message_verification_components": "mldsa/verification.slang",
    "prehashed_verification_components": "mldsa/verification.slang",
    "mldsa_prehash_components": "mldsa/prehash.slang",
    "mldsa_sign_message_components": "mldsa/sign_message.slang",
    "mldsa_sign_message_workspace_components": "mldsa/sign_message_workspace.slang",
    "mldsa_prehash_workspace_components": "mldsa/prehash_workspace.slang",
    "mldsa_hash_message_workspace_components": "mldsa/hash_message_workspace.slang",
    "mldsa_sign_components": "mldsa/sign.slang",
    "mldsa_mask_components": "mldsa/sign.slang",
    "mldsa_secret_sampling_components": "mldsa/keygen.slang",
    "mldsa_keygen_components": "mldsa/keygen.slang",
    "keccak_components": "common/keccak_permutation.slang",
    "parameter_verification_components": "mldsa/verification.slang",
    "bit_packing_components": "mldsa/packing.slang",
    "field_components": "mldsa/field.slang",
    "mlkem_compression_components": "mlkem/compression.slang",
    "mlkem_encaps_components": "mlkem/kem.slang",
    "mlkem_field_components": "mlkem/field.slang",
    "mlkem_kem_components": "mlkem/kem.slang",
    "mlkem_kem_decaps_components": "mlkem/kem.slang",
    "mlkem_key_check_components": "mlkem/kem.slang",
    "mlkem_keygen_components": "mlkem/kem.slang",
    "mlkem_kpke_components": "mlkem/kpke.slang",
    "mlkem_kpke_decrypt_components": "mlkem/kpke.slang",
    "mlkem_packing_components": "mlkem/packing.slang",
    "mlkem_polynomial_components": "mlkem/ntt.slang",
    "mlkem_rejection_components": "mlkem/sampling.slang",
    "mlkem_sampling_components": "mlkem/sampling.slang",
    "packing_components": "mldsa/packing.slang",
    "polynomial_components": "mldsa/ntt.slang",
    "range_components": "mldsa/verification.slang",
    "sampling_components": "mldsa/sampling.slang",
    "streaming_components": "common/shake_streaming.slang",
    "verification_components": "mldsa/verification.slang",
    "wide_components": "mldsa/field.slang",
}


# Keep ByteAddressBuffer counts in bytes; StructuredBuffer counts are elements.
BYTE_ADDRESS_ENTRIES = frozenset((
    "mldsa_prehash_digest_components",
    "sha2_u32_pair_components", "sha2_u32_pair_arithmetic_components",
    "sha2_u32_components", "sha2_u32_length_components",
    "sha3_components", "sha3_range_components",
    "mldsa_sign_message_workspace_components", "mldsa_prehash_workspace_components",
    "mldsa_hash_message_workspace_components",
    "hash_message_verification_components", "prehashed_verification_components",
    "mldsa_prehash_components",
    "mldsa_sign_message_components",
    "packing_components", "bit_packing_components", "streaming_components",
    "verification_components", "parameter_verification_components", "range_components",
))

OUTPUT_BYTE_ADDRESS_ENTRIES = frozenset(("mldsa_sign_workspace_components",
    "mldsa_sign_message_workspace_components", "mldsa_prehash_workspace_components",
    "mldsa_hash_message_workspace_components"))


# Run the compiler-generated host translation in a separate sanitized process.
# The driver contains only buffer ABI, guarded storage and file transport.
SANITIZER_DRIVER = r"""
#include <vector>
#include <fstream>
#include <cstdlib>
#include <limits>
int main(int argc, char** argv) {
    if (argc != 5) return 1;
    const size_t rows = std::stoull(argv[3]), width = std::stoull(argv[4]);
    if (rows > std::numeric_limits<uint32_t>::max() ||
        (width && rows > (std::numeric_limits<size_t>::max() - 2) / width)) return 2;
    std::ifstream source(argv[1], std::ios::binary | std::ios::ate);
    if (!source) return 3;
    const auto bytes = source.tellg();
    if (bytes < 0 || bytes % 4) return 4;
    std::vector<uint32_t> inputs(static_cast<size_t>(bytes) / 4);
    source.seekg(0);
    source.read(reinterpret_cast<char*>(inputs.data()), bytes);
    if (!source) return 5;
    const size_t count = rows * width;
    std::vector<uint32_t> outputs(count + 2, 0xa5a5a5a5u);
    GlobalParams_0 params{};
    params.inputs_0 = {inputs.data(), INPUT_COUNT};
    params.outputs_0 = {outputs.data() + 1, OUTPUT_COUNT};
    ComputeVaryingInput varying{};
    varying.endGroupID = {static_cast<uint32_t>(rows), 1, 1};
    ENTRY(&varying, nullptr, &params);
    if (outputs.front() != 0xa5a5a5a5u || outputs.back() != 0xa5a5a5a5u) return 6;
    std::ofstream result(argv[2], std::ios::binary);
    result.write(reinterpret_cast<const char*>(outputs.data() + 1), count * 4);
    return result ? 0 : 7;
}
"""


class SlangComponentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="oa-pqc-components-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        configuration = tomllib.loads((ROOT / ".cargo/config.toml").read_text())
        compiler = shutil.which(os.environ.get("SLANGC", configuration["env"]["SLANGC"]))
        validator = shutil.which("spirv-val")
        disassembler = shutil.which("spirv-dis")
        if compiler is None or validator is None or disassembler is None:
            raise RuntimeError("PQC component proofs require slangc, spirv-val and spirv-dis")
        cls.libraries = {}
        cls.sanitizer_binaries = {}
        cls.sanitize = os.environ.get("OA_PQC_SANITIZE", "0") == "1"
        cls.native_compiler = shutil.which("clang++") if cls.sanitize else None
        if cls.sanitize and cls.native_compiler is None:
            raise RuntimeError("PQC sanitizer proofs require clang++")

        def run(args: list[str], timeout: int) -> str:
            result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                raise RuntimeError(f"component compilation/validation failed: {args}\n"
                                   f"{result.stdout}{result.stderr}")
            return result.stdout

        cls.compiler = compiler
        cls.validator = validator
        cls.disassembler = disassembler
        cls.run_tool = staticmethod(run)

    @classmethod
    def compile_entry(cls, entry: str) -> None:
        if entry in cls.libraries:
            return
        compiler, validator, disassembler = cls.compiler, cls.validator, cls.disassembler
        run = cls.run_tool
        source = ENTRY_SOURCES[entry]
        args = [compiler, str(ROOT / "test/slang/cryptography" / source),
                "-entry", entry, "-stage", "compute", "-I",
                str(ROOT / "src/slang/cryptography/pqc/mldsa"), "-I",
                str(ROOT / "src/slang/cryptography/pqc/mlkem"), "-I",
                str(ROOT / "src/slang/cryptography/pqc/common"), "-I",
                str(ROOT / "src/slang/cryptography/hash"), "-I",
                str(ROOT / "src/slang/core/math")]
        library = cls.directory / f"{entry}.so"
        run(args + ["-target", "shader-sharedlib", "-o", str(library)], timeout=120)
        for version, environment in (("1_5", "vulkan1.2"), ("1_6", "vulkan1.3")):
            artifact = cls.directory / f"{entry}-{version}.spv"
            run(args + ["-target", "spirv", "-profile", "glsl_460",
                        "-capability", f"spirv_{version}", "-o", str(artifact)], timeout=120)
            run([validator, "--target-env", environment, str(artifact)], timeout=30)
            assembly = run([disassembler, str(artifact)], timeout=30)
            if "OpCapability Int64" in assembly:
                raise RuntimeError("PQC arithmetic unexpectedly requires shaderInt64")
        if cls.sanitize:
            source = cls.directory / f"{entry}.cpp"
            run(args + ["-target", "cpp", "-o", str(source)], timeout=120)
            input_count = "inputs.size() * 4" if entry in BYTE_ADDRESS_ENTRIES else "inputs.size()"
            with source.open("a") as output:
                output_count = "count * 4" if entry in OUTPUT_BYTE_ADDRESS_ENTRIES else "count"
                output.write(SANITIZER_DRIVER.replace("INPUT_COUNT", input_count)
                             .replace("OUTPUT_COUNT", output_count).replace("ENTRY", entry))
            binary = cls.directory / f"{entry}-sanitized"
            run([cls.native_compiler, "-std=c++17", "-O1", "-g",
                 "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                 str(source), "-o", str(binary)], timeout=120)
            cls.sanitizer_binaries[entry] = binary
        cls.libraries[entry] = ctypes.CDLL(str(library))


    def execute(self, entry: str, values: list[int], rows: int, width: int) -> list[int]:
        self.compile_entry(entry)
        if self.sanitize:
            input_file = self.directory / f"{entry}-input.bin"
            output_file = self.directory / f"{entry}-output.bin"
            input_file.write_bytes(struct.pack(f"<{len(values)}I", *values))
            self.run_tool([str(self.sanitizer_binaries[entry]), str(input_file),
                           str(output_file), str(rows), str(width)], timeout=120)
            raw = output_file.read_bytes()
            self.assertEqual(len(raw), rows * width * 4)
            return list(struct.unpack(f"<{rows * width}I", raw))
        inputs = (ctypes.c_uint32 * len(values))(*values)
        output_count = rows * width
        outputs = (ctypes.c_uint32 * (output_count + 2))(*([0xA5A5A5A5] * (output_count + 2)))
        input_count = len(inputs)
        if entry in BYTE_ADDRESS_ENTRIES:
            input_count *= 4  # Slang ByteAddressBuffer views count bytes.
        parameters = Parameters(BufferView(ctypes.cast(inputs, ctypes.c_void_p), input_count),
                                BufferView(ctypes.cast(ctypes.byref(outputs, 4), ctypes.c_void_p),
                                           output_count * (4 if entry in OUTPUT_BYTE_ADDRESS_ENTRIES else 1)))
        varying = VaryingInput((0, 0, 0), (rows, 1, 1))
        function = getattr(self.libraries[entry], entry)
        function.argtypes = [ctypes.POINTER(VaryingInput), ctypes.c_void_p,
                             ctypes.POINTER(Parameters)]
        function.restype = None
        # Both arrays and the exact prelude ABI structs stay live through this
        # synchronous call; the test shader uses only the admitted row ranges.
        function(ctypes.byref(varying), None, ctypes.byref(parameters))
        self.assertEqual(outputs[0], 0xA5A5A5A5, "shader overwrote leading output guard")
        self.assertEqual(outputs[-1], 0xA5A5A5A5, "shader overwrote trailing output guard")
        return list(outputs)[1:-1]
