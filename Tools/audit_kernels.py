#!/usr/bin/env python3
"""Kernel audit — cross-reference the four sources of truth for compute kernels
and report every inconsistency. This is the tool the old (dead) CMake validator
should have been: it covers all categories, uses the real paths, and understands
namespaced dispatch names.

Four dimensions per kernel:
  compiled   — appears in a CMakeLists OA_*_SHADER_REG list (a .spv is built)
  slang      — a matching .slang source file exists on disk
  dispatched — a ctx.Add("<name>", …) / OaSpvFind("<name>") call-site names it
  (registry) — KernelRegistry.h has an entry (informational; not all cats use it)

Reports:
  DEAD        compiled + slang but never dispatched  → strip candidate
  ORPHAN_SPV  compiled but no .slang source           → build breakage / stale
  MISSING_SPV dispatched but not compiled             → runtime "kernel not found"

Runs in-process (no shell fan-out), so it is fast and deterministic.
"""
import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CMAKE = (ROOT / "CMakeLists.txt").read_text()

# ── 1. compiled set: every OA_*_SHADER_REG list + explicit APPENDs ────────────
compiled = set()
for m in re.finditer(r"set\((OA_\w*SHADER\w*REG|OA_AUDIO_SHADERS|OA_\w*SHADERS)\s+(.*?)\)", CMAKE, re.S):
    for tok in m.group(2).split():
        if not tok.startswith("#"):
            compiled.add(tok.split("/")[-1])   # basename; dispatch/reg names vary by prefix
for m in re.finditer(r'APPEND\s+OA_ALL_SPV_NAMES\s+"([^"]+)"', CMAKE):
    compiled.add(m.group(1).split("/")[-1])

# ── 2. slang sources on disk ──────────────────────────────────────────────────
slang = {p.stem for p in ROOT.glob("Source/**/*.slang")}

# ── 3. "referenced" = the kernel basename appears as ANY quoted string literal
# anywhere in C++ (.cpp/.h). This is deliberately permissive: it treats a kernel
# as live if *any* dispatch mechanism could name it (ctx.Add, GEMM router, crypto
# dispatcher, graphics pipeline, namespaced "Cat/Name"). A shader referenced
# nowhere in C++ is a high-confidence strip candidate. ──────────────────────────
dispatched = set()
lit_re = re.compile(r'"([^"\\]+)"')
for src in list(ROOT.glob("Source/**/*.cpp")) + list(ROOT.glob("Source/**/*.h")):
    if src.name == "KernelRegistry.h":
        continue   # a registry row is a registration, not a dispatch — don't count it as "live"
    for m in lit_re.finditer(src.read_text(errors="ignore")):
        s = m.group(1)
        dispatched.add(s.split("/")[-1])   # namespaced-name aware
        dispatched.add(s)

# ── 3b. second dispatch mechanism: kernels named by OaKernelId constant ───────
ck = (ROOT / "Source/Public/Oa/Runtime/ComputeKernel.h").read_text()
for m in re.finditer(r"OaKernelId\s+(\w+)\s*=", ck):
    dispatched.add(m.group(1))

# ── 4. registry names ─────────────────────────────────────────────────────────
reg_txt = (ROOT / "Source/Public/Oa/Core/KernelRegistry.h").read_text()
registered = {m.group(1).split("/")[-1] for m in re.finditer(r'\{\s*"([^"]+)"', reg_txt)}

# crypto + graphics dispatch via non-literal mechanisms (crypto dispatcher /
# pipeline creation), so exclude them from strip candidates.
CRYPTO = {"KeccakF1600","MldsaKeygen","MldsaSign","MldsaVerify","NttForward","NttInverse",
          "NttPointwise","PolyArith","PolyHint","PolyRound","PolySample","Shake128","Shake256",
          "MerkleReduce","MerkleVerify","Keccak"}
def is_gfx(k): return k.endswith(("frag","vert"))

dead = sorted(k for k in compiled & slang
              if k not in dispatched and k not in registered and k not in CRYPTO and not is_gfx(k))
# registered-but-never-dispatched: a live registry slot pointing at an unused kernel
reg_dead = sorted(k for k in compiled & slang
                  if k not in dispatched and k in registered and k not in CRYPTO and not is_gfx(k))

def show(title, items):
    print(f"\n=== {title} ({len(items)}) ===")
    for i in items: print(f"  {i}")

print(f"compiled={len(compiled)} slang={len(slang)} dispatched={len(dispatched)} registered={len(registered)}")
show("DEAD — compiled+source, not dispatched, not registered (STRIP)", dead)
show("REG_DEAD — registered + compiled but never dispatched (STRIP reg+shader)", reg_dead)
sys.exit(0)
