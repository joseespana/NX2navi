"""Smoke test for build/xlate/nx2navi_xlate(.exe): no game files needed.

Sends hand-assembled Maxwell and SM86 shaders through --serve and validates every result with spirv-val.
Covers global memory ordering, multisampled array fetches, signed surface atomics, mesh output reads and global
compare-and-swap operand order, and an image read without a format (StorageImageReadWithoutFormat). Checks the
lowered Maxwell words before optimization, and recovery after malformed JSON. It also sends a compute shader that adds 1.0 with
RED.E.ADD.F32.FTZ.RN to a global address twice, once without and once with the job flag "float_atomic_add": the first
must come back as a compare-and-swap loop, the second as one OpAtomicFAddEXT with its capability; both are validated.

  python tools/xlate_smoke.py [--translator /path/to/nx2navi_xlate]
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "build" / "xlate" / ("nx2navi_xlate.exe" if os.name == "nt" else "nx2navi_xlate")
SPIRV_VAL = (Path(r"C:\msys64\ucrt64\bin\spirv-val.exe") if os.name == "nt"
             else Path(shutil.which("spirv-val") or "/usr/bin/spirv-val"))

PT = 7 << 16  # predicate "always"


def mov32i(dest: int, value: float) -> int:
    imm = struct.unpack("<I", struct.pack("<f", value))[0]
    return 0x0100000000000000 | (imm << 20) | (0xF << 12) | PT | dest


EXIT = 0xE30000000000000F | PT
SCHED = 0x001F8000FC0007E0  # scheduling control word; the decoder skips it


def fragment_code(instructions: list[int], sm86: bool = False, output_mask: int = 15) -> bytes:
    header = [0] * (32 if sm86 else 20)
    header[0] = 2 | (3 << 5) | (5 << 10)  # SPH type 2, version 3, stage fragment
    header[18] = output_mask
    if sm86:
        return struct.pack("<32I", *header) + b"".join(word.to_bytes(16, "little") for word in instructions)
    words = []
    instructions = instructions + [EXIT]
    for start in range(0, len(instructions), 3):
        group = instructions[start:start + 3]
        words.extend([SCHED, *group, *([0] * (3 - len(group)))])
    return struct.pack("<20I", *header) + struct.pack(f"<{len(words)}Q", *words)


RZ = 0xFF
IMAGE0 = 0x120 // 4  # first NVN image handle in c[2] (tools/xlate/Program.cs ImgSlot): image slot 0


SULD_P, SUST_P = 0x1D61, 0x1D65  # opcode bits 51-63 of SULD.P and SUST.P


def surface_p(opcode: int, data: int, coord: int, handle: int) -> int:
    """SULD.P / SUST.P 1D RGBA: the data registers from `data` on, coordinate register, image handle (TidB)."""
    no_pred_out = 7 << 30 if opcode == SULD_P else 0  # SULD's second destination predicate: PT, none written
    return (opcode << 51) | (handle << 36) | no_pred_out | (0xF << 20) | PT | (coord << 8) | data


def image_read_code() -> bytes:
    """Compute code: R0-R3 = image 0 at x 0 (SULD.P), stored back there (SUST.P)."""
    words = [SCHED, surface_p(SULD_P, 0, RZ, IMAGE0), surface_p(SUST_P, 0, RZ, IMAGE0), EXIT]
    return struct.pack(f"<{len(words)}Q", *words)


def shader_cases() -> list[tuple[str, list[int], dict]]:
    constant = [mov32i(0, 1.0), mov32i(1, 0.5), mov32i(2, 0.25), mov32i(3, 1.0)]
    cases = [("const_ps", constant, {})]
    # Each expected word includes the always predicate used by AmpereLowering.cs EmitG.
    for ordering in (0, 0x2000, 0x4000, 0x6000):
        for uniform in (False, True):
            load = (0x981 if uniform else 0x381) | (7 << 12) | (2 << 24) | ((0x081e8900 if uniform else 0x001e8900) | ordering) << 64
            expected = 0xeed0000000000000 | PT | ((244 if uniform else 2) << 8) | (1 << 45) | (4 << 48)
            cases.append((f"ldg_{uniform}_{ordering:x}", [load], {"sm86": True, "_check": {"lowered": [expected]}}))
    for ordering in (0, 0x4000):
        for uniform in (False, True):
            store = (0x986 if uniform else 0x386) | (7 << 12) | (2 << 24) | ((0x0810a900 if uniform else 0x0010a900) | ordering) << 64
            expected = 0xeed8000000000000 | PT | ((244 if uniform else 2) << 8) | (1 << 45) | (4 << 48)
            cases.append((f"stg_{uniform}_{ordering:x}", [store], {"sm86": True, "_check": {"lowered": [expected]}}))
    for ordering in (0xa100, 0xe100):
        # Distinct compare/swap registers must remain in that order in Maxwell's consecutive pair.
        cas = 0x3a9 | (7 << 12) | (2 << 24) | (4 << 32) | ((0x001e0000 | ordering | 6) << 64)
        moves = [0x5c98078000000000 | PT | (4 << 20) | 252,
                 0x5c98078000000000 | PT | (6 << 20) | 253]
        atomic = 0xeef0000000000000 | PT | (2 << 8) | (252 << 20) | (1 << 48)
        cases.append((f"cas_{ordering:x}", [cas], {"sm86": True, "_check": {"lowered": moves + [atomic]}}))
    for signed in (False, True):
        for ordering in (0xa000, 0xe000):
            atom = 0x593 | (7 << 12) | (2 << 24) | (4 << 32) | (0x48 << 40) | (2 << 54) | (3 << 61)
            atom |= (0x001e0000 | ordering | (int(signed) << 9)) << 64
            expected = 0xea00000000000000 | PT | (2 << 8) | (4 << 20) | (3 << 33) | (0x48 << 36) | (int(signed) << 51)
            cases.append((f"suatom_{signed}_{ordering:x}", [atom],
                          {"sm86": True, "img": {"0": {"type": "2d", "fmt": "R32Sint" if signed else "R32Uint"}},
                           "_check": {"lowered": [expected]}}))
    for offsets in (False, True):
        fetch = 0xb66 | (7 << 12) | (2 << 24) | (4 << 32) | (0x5a << 40) | (2 << 54) | (5 << 61)
        fetch |= (255 | (15 << 8) | (int(offsets) << 12) | (1 << 14) | (7 << 17) | (1 << 23)) << 64
        expected = 0xdc00000000000000 | PT | 248 | (240 << 8) | (244 << 20) | (3 << 28) | (15 << 31) | (int(offsets) << 35) | (0x5a << 36) | (1 << 50) | (7 << 51)
        cases.append((f"tld_ms_array_{offsets}", [fetch],
                      {"sm86": True, "tex": {"0": "2dmsarray"}, "_check": {"lowered": [expected]}}))
    read_output = 0x923 | (7 << 12) | (255 << 24) | (15 << 75) | (1 << 79)
    cases.append(("isberd_mesh_output", [read_output],
                  {"sm86": True, "stage": "ms", "mesh": {"stage": 7, "threads": 32, "maxv": 1,
                   "maxp": 1, "topo": 1, "payload": 0, "task": False},
                   "_check": {"lowered": [0xef48000000000000 | PT | (255 << 8) | (4 << 48)]}}))
    return cases


def mov_c(dest: int, offset: int) -> int:
    return 0x4C98078000000000 | ((offset // 4) << 20) | PT | dest   # MOV Rd, c[0x0][offset]


def red_f32_add(addr: int, data: int) -> int:
    return 0xEBF8000000000000 | (1 << 48) | (3 << 20) | (addr << 8) | PT | data   # RED.E.ADD.F32.FTZ.RN [Ra.64], Rb


def compute_code() -> bytes:
    """Compute program: [c0[0x140]] (a pointer, taken as a storage buffer) += 1.0, no SPH."""
    words = [SCHED, mov_c(0, 0x140), mov_c(1, 0x144), mov32i(2, 1.0),
             SCHED, red_f32_add(0, 2), EXIT, 0]
    return struct.pack(f"<{len(words)}Q", *words)


def opcodes(path: Path) -> dict[int, int]:
    """Instruction count per SPIR-V opcode, and under -1 - capability the declared capabilities."""
    words = struct.unpack(f"<{path.stat().st_size // 4}I", path.read_bytes())
    count: dict[int, int] = {}
    i = 5
    while i < len(words):
        op, n = words[i] & 0xFFFF, words[i] >> 16
        count[op] = count.get(op, 0) + 1
        if op == 17:
            count[-1 - words[i + 1]] = 1
        i += n
    return count


def check_float_atomic(tmp: Path) -> int:
    OP_ATOMIC_CAS, OP_ATOMIC_FADD, CAP_ATOMIC_FLOAT32_ADD = 230, 6035, 6033
    code = tmp / "cs.bin"
    code.write_bytes(compute_code())
    out = tmp / "cs_out"
    lines = []
    for key, native in (("red_loop", False), ("red_native", True)):
        job = {"key": key, "stage": "cs", "code": str(code), "debug": False, "local": [32, 1, 1], "shared": 0}
        if native:
            job["float_atomic_add"] = True
        lines.append(json.dumps({"job": job, "out": str(out)}))
    r = subprocess.run([str(EXE), "--serve"], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=60)
    if r.stdout.splitlines() != ["ok red_loop", "ok red_native"]:
        print("float atomic add: unexpected replies", r.stdout.splitlines(), r.stderr, file=sys.stderr)
        return 1
    for key, native in (("red_loop", False), ("red_native", True)):
        spv = out / f"{key}.spv"
        ops = opcodes(spv)
        want = (ops.get(OP_ATOMIC_FADD, 0) == 1 and ops.get(OP_ATOMIC_CAS, 0) == 0 and (-1 - CAP_ATOMIC_FLOAT32_ADD) in ops) if native \
            else (ops.get(OP_ATOMIC_FADD, 0) == 0 and ops.get(OP_ATOMIC_CAS, 0) == 1 and (-1 - CAP_ATOMIC_FLOAT32_ADD) not in ops)
        v = subprocess.run([str(SPIRV_VAL), "--target-env", "vulkan1.3", str(spv)], capture_output=True, text=True)
        print(f"{key}: {'OpAtomicFAddEXT' if ops.get(OP_ATOMIC_FADD) else 'compare-and-swap loop' if ops.get(OP_ATOMIC_CAS) else 'no atomic'}, "
              f"spirv-val exit {v.returncode} {v.stdout}{v.stderr}")
        if not want or v.returncode:
            print(f"float atomic add: {key} is not what the job asked for", file=sys.stderr)
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translator", type=Path, default=EXE, help="translator executable; default: build/xlate")
    translator = parser.parse_args().translator.resolve()
    if not translator.is_file():
        print(f"missing {translator}; run tools/build_xlate.py or pass --translator", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out = tmp / "out"
        jobs = []
        checks = {}
        for key, instructions, options in shader_cases():
            options = dict(options)
            checks[key] = options.pop("_check", {})
            code = tmp / (key + ".bin")
            code.write_bytes(fragment_code(instructions, options.get("sm86", False), 0 if "mesh" in options else 15))
            jobs.append({"key": key, "stage": "ps", "code": str(code), "debug": False,
                         "fout": {"0": "float"}, "lowered_out": str(tmp / (key + ".maxwell")), **options})
        # A compute shader that loads a texel from an image without a format and stores it back: its read needs
        # StorageImageReadWithoutFormat.
        image_code = tmp / "image_read_cs.bin"
        image_code.write_bytes(image_read_code())
        jobs.append({"key": "image_read_cs", "stage": "cs", "code": str(image_code), "debug": False, "local": [1, 1, 1]})
        checks["image_read_cs"] = {}
        lines = [json.dumps({"job": job, "out": str(out)}) for job in jobs]
        lines.extend(["this is not json", json.dumps({"job": dict(jobs[0], key="again"), "out": str(out)})])
        r = subprocess.run([str(translator), "--serve"], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=60)
        replies = r.stdout.splitlines()
        print("replies:", replies)
        expected = ["ok " + job["key"] for job in jobs]
        if (len(replies) != len(expected) + 2 or replies[:-2] != expected
                or not replies[-2].startswith("fail ") or replies[-1] != "ok again"):
            print("unexpected replies; stderr:", r.stderr, file=sys.stderr)
            return 1
        for job in jobs:
            spv = out / (job["key"] + ".spv")
            data = spv.read_bytes()
            if len(data) < 20 or struct.unpack("<I", data[:4])[0] != 0x07230203:
                print("reply is not SPIR-V", file=sys.stderr)
                return 1
            v = subprocess.run([str(SPIRV_VAL), "--relax-block-layout", "--uniform-buffer-standard-layout",
                                "--scalar-block-layout", "--allow-localsizeid", "--target-env", "vulkan1.3",
                                str(spv)], capture_output=True, text=True)
            print(f"{job['key']}: spirv-val exit {v.returncode} ({len(data)} bytes) {v.stdout}{v.stderr}")
            if v.returncode:
                return v.returncode
            if checks[job["key"]]:
                lowered = Path(job["lowered_out"]).read_bytes()[0 if "mesh" in job else 0x50:]
                words = [word[0] for i, word in enumerate(struct.iter_unpack("<Q", lowered)) if i % 4]
                expected_words = checks[job["key"]]["lowered"]
                if not any(words[i:i + len(expected_words)] == expected_words for i in range(len(words))):
                    print(f"{job['key']}: lowered operand contract failed: {[hex(word) for word in words]}", file=sys.stderr)
                    return 1
        return check_float_atomic(tmp)


if __name__ == "__main__":
    sys.exit(main())
