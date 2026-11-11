"""Smoke test for build/xlate/nx2navi_xlate(.exe): no game files needed.

Sends hand-assembled Maxwell and SM86 shaders through --serve and validates every result with spirv-val.
Covers global memory ordering, multisampled array fetches, signed surface atomics, mesh output reads and global
compare-and-swap operand order, an image read without a format (StorageImageReadWithoutFormat), and a size query that
reads only the layer count of a texture the translator has no type answer for, which must be declared as an array. Checks the
lowered Maxwell words before optimization, and recovery after malformed JSON. It also sends a compute shader that adds 1.0 with
RED.E.ADD.F32.FTZ.RN to a global address twice, once without and once with the job flag "float_atomic_add": the first
must come back as a compare-and-swap loop, the second as one OpAtomicFAddEXT with its capability; both are validated.
Then typed image and texture cases (image formats, typed fetch/sample/gather and channel masks, dimension queries,
signed integer conversion) are translated and their numeric contracts checked. Last, the depth comparison rewrite that
fragment shaders get on macOS (--frag-z-snap) runs on two hand-built modules. No game files.

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
RZ = 0xFF
IMAGE0 = 0x120 // 4  # first NVN image handle in c[2] (tools/xlate/Program.cs ImgSlot): image slot 0


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
TXQ = 0x1BE9  # opcode bits 51-63 of TXQ (bound handle)


def surface_p(opcode: int, data: int, coord: int, handle: int) -> int:
    """SULD.P / SUST.P 1D RGBA: the data registers from `data` on, coordinate register, image handle (TidB)."""
    no_pred_out = 7 << 30 if opcode == SULD_P else 0  # SULD's second destination predicate: PT, none written
    return (opcode << 51) | (handle << 36) | no_pred_out | (0xF << 20) | PT | (coord << 8) | data


def image_read_code() -> bytes:
    """Compute code: R0-R3 = image 0 at x 0 (SULD.P), stored back there (SUST.P)."""
    words = [SCHED, surface_p(SULD_P, 0, RZ, IMAGE0), surface_p(SUST_P, 0, RZ, IMAGE0), EXIT]
    return struct.pack(f"<{len(words)}Q", *words)


def layer_count_code() -> bytes:
    """Compute code: R0 = the layer count (component 2 of the dimension query) of texture handle 0x10, stored to image 0."""
    txq = (TXQ << 51) | (0x10 << 36) | (0x4 << 31) | (1 << 22) | PT | (RZ << 8) | 0  # TexHeaderDimension, mask z, LOD RZ
    sust = (SUST_P << 51) | (IMAGE0 << 36) | (0x1 << 20) | PT | (RZ << 8) | 0  # SUST.P 1D R of R0 at x 0
    words = [SCHED, txq, sust, EXIT]
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
    # A mesh shader without a task stage that reads its workgroup ID (S2R R0, SR_CTAID.X; STS [RZ], R0). On macOS the
    # module also takes its block's first workgroup from a one-word payload (tools/xlate/MeshSpirv.cs): "mac_payload".
    s2r = 0x919 | (7 << 12) | (0x25 << 72)
    sts = 0x388 | (7 << 12) | (RZ << 24) | (4 << 73)
    cases.append(("mesh_ctaid", [s2r, sts],
                  {"sm86": True, "stage": "ms", "mesh": {"stage": 7, "threads": 32, "maxv": 1,
                   "maxp": 1, "topo": 1, "payload": 0, "task": False},
                   "_check": {"lowered": [0xf0c8000000000000 | PT | (0x25 << 20)], "mac_payload": True}}))
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


def typed_cases() -> list[tuple[str, list[int], dict]]:
    constant = [mov32i(0, 1.0), mov32i(1, 0.5), mov32i(2, 0.25), mov32i(3, 1.0)]
    cases = [("const_ps", constant, {})]
    # TXQ query 1, mask xyz, c[2][0x168] (texture slot 0), LOD zero from RZ.
    # Requesting z on 2D/cube previously emitted CompositeExtract of vec2 at index 2.
    txq = 0xDF48000000000000 | (0x5A << 36) | (7 << 31) | (1 << 22) | (255 << 8) | PT
    for kind in ("2d", "2dms", "cube", "2darray", "2dmsarray", "cubearray", "3d"):
        cases.append(("txq_" + kind, [txq, mov32i(3, 1.0)], {"tex": {"0": kind}}))
    # SULD.2D.RGBA, c[2][0x120] (image slot 0), integer coordinates zero from RZ.
    suld = 0xEB08000000000000 | (0x48 << 36) | (3 << 33) | (7 << 30) | (15 << 20) | (255 << 8) | PT
    # SPIR-V ImageFormat values: test every supported integer storage format, not only Rg16ui.
    image_formats = (("R8G8B8A8Unorm", 0, "float"), ("R32Uint", 33, "uint"),
                     ("R16G16Uint", 36, "uint"), ("R16G16Sint", 26, "sint"),
                     ("R8Uint", 39, "uint"), ("R16Uint", 38, "uint"), ("R8G8Uint", 37, "uint"),
                     ("R32G32Uint", 35, "uint"), ("R10G10B10A2Uint", 34, "uint"),
                     ("R8G8B8A8Uint", 32, "uint"), ("R16G16B16A16Uint", 31, "uint"),
                     ("R32G32B32A32Uint", 30, "uint"), ("R8Sint", 29, "sint"),
                     ("R16Sint", 28, "sint"), ("R8G8Sint", 27, "sint"), ("R32G32Sint", 25, "sint"),
                     ("R32Sint", 24, "sint"), ("R8G8B8A8Sint", 23, "sint"),
                     ("R16G16B16A16Sint", 22, "sint"), ("R32G32B32A32Sint", 21, "sint"),
                     ("R16G16Float", 0, "float"))  # Formatted float loads deliberately declare Unknown.
    for fmt, image_format, component in image_formats:
        cases.append(("image_" + fmt, [suld],
                      {"img": {"0": {"type": "2d", "fmt": fmt}},
                       "tex": {"0": {"type": "2d", "fmt": "R32Uint" if component == "float" else "R32Float"}},
                       "_check": {"storage_format": (image_format, component)}}))
    # Combined-handle byte 0x40 maps to texture slot 4, independently of storage-image slot 4.
    # Poison the image answer to detect accidental sharing of the two format namespaces.
    for kind, fmt, component in (("uint", "R16G16Uint", "uint"), ("sint", "R16G16Sint", "sint"),
                                 ("float", "R32Float", "float")):
        for operation, base, opcode, masks in (("fetch", 0xDC00000000000000, 95, (15, 4, 5)),
                                               ("sample", 0xC080000000000000, 88, (15,)),
                                               ("gather", 0xC800000000000000, 96, (15,))):
            for mask in masks if kind != "float" else (15,):
                inst = base | (0x10 << 36) | (2 << 28) | (mask << 31) | (7 << 51) | (255 << 20) | (255 << 8) | PT
                options = {"tex": {"4": {"type": "2d", "fmt": fmt}},
                           "img": {"4": {"type": "2d", "fmt": "R32Float" if kind != "float" else "R32Uint"}},
                           "_check": {"sample": (component, opcode, mask), "format": fmt}}
                cases.append((f"texture_{operation}_{kind}_{mask}", constant + [inst], options))
    depth = 0xC080000000000000 | (0x10 << 36) | (2 << 28) | (1 << 31) | (7 << 51) | (1 << 50) | (255 << 20) | (255 << 8) | PT
    cases.append(("texture_depth", constant + [depth],
                  {"tex": {"4": {"type": "2d_depth", "fmt": "R32Float"}},
                   "_check": {"sample": ("float", 90, 1), "format": "R32Float"}}))
    # -1 interpreted as S32 becomes -1.0; the same U32 word rounds to 2**32 in F32.
    # Integer output keeps the resulting float bits, bypassing the float-output clamp.
    move = 0x802 | (7 << 12) | (4 << 16) | (0xFFFFFFFF << 32) | (15 << 72)
    for signed, expected in ((True, 0xBF800000), (False, 0x4F800000)):
        convert = 0x306 | (7 << 12) | (4 << 32) | (int(signed) << 74) | (2 << 75) | (2 << 84)
        cases.append(("sm86_i2f_" + ("signed" if signed else "unsigned"), [move, convert],
                      {"sm86": True, "fout": {"0": "uint"}, "_omap": 1,
                       "_check": {"conversion_bits": expected}}))
    return cases


def spirv_instructions(data: bytes) -> list[tuple[int, tuple[int, ...]]]:
    words = struct.unpack(f"<{len(data) // 4}I", data)
    instructions = []
    offset = 5
    while offset < len(words):
        count, opcode = words[offset] >> 16, words[offset] & 0xFFFF
        if count == 0 or offset + count > len(words):
            raise ValueError("invalid SPIR-V instruction length")
        instructions.append((opcode, words[offset + 1:offset + count]))
        offset += count
    return instructions


def check_numeric_contract(instructions: list[tuple[int, tuple[int, ...]]], check: dict, resources: dict) -> None:
    types = {args[0]: (opcode, args[1:]) for opcode, args in instructions if opcode in (21, 22, 23, 25)}
    value_ops = {1, 43, 44, 46, 61, 65, 79, 80, 81, 83, 87, 88, 89, 90, 95, 96, 97, 98, 111, 112, 124}
    values = {args[1]: (opcode, args) for opcode, args in instructions if opcode in value_ops}

    def component(type_id: int) -> tuple[str, int]:
        opcode, args = types[type_id]
        if opcode == 23:  # OpTypeVector
            return component(args[0])[0], args[1]
        assert opcode in (21, 22) and args[0] == 32, "expected 32-bit numeric component"
        return ("float" if opcode == 22 else "sint" if args[1] else "uint"), 1

    if "storage_format" in check:
        image_format, kind = check["storage_format"]
        images = [args for opcode, args in types.values() if opcode == 25 and args[5] == 2]
        assert len(images) == 1 and images[0][6] == image_format, "storage image format answer changed"
        assert component(images[0][0]) == (kind, 1), "wrong storage image numeric type"
        assert resources["img_queried"] == [0] and not resources["tex_queried"], "image used sampled-texture format answers"
        capabilities = {args[0] for opcode, args in instructions if opcode == 17}
        # StorageImageExtendedFormats=49; Shader-only integer formats are rgba* and r32*.
        extended = image_format in (*range(6, 21), *range(25, 30), *range(34, 40))
        assert (49 in capabilities) == extended, "incorrect extended storage image format capability"
        assert (55 in capabilities) == (image_format == 0), "incorrect unformatted image read capability"
        assert any(opcode == 98 for opcode, _ in instructions), "storage image read was dropped"

    if "sample" in check:
        kind, sample_opcode, mask = check["sample"]
        images = [args for opcode, args in types.values() if opcode == 25 and args[5] == 1]
        assert len(images) == 1 and component(images[0][0]) == (kind, 1), "wrong sampled image numeric type"
        assert images[0][6] == 0, "sampled image format must remain Unknown"
        assert images[0][2] == int(sample_opcode == 90), "depth-compare image declaration changed"
        samples = {args[1] for opcode, args in instructions if opcode == sample_opcode}
        assert samples, "expected texture operation was dropped"
        for sample in samples:
            assert component(values[sample][1][0]) == (kind, 1 if sample_opcode == 90 else 4), "wrong texture result type"
        assert resources["tex_queried"] == [4] and not resources["img_queried"], "combined texture handle used image answers"
        assert any(texture["format"] == check["format"] for texture in resources["textures"]), "texture format answer lost"

        def lanes(value: int) -> tuple[int, ...] | None:
            if value in samples:
                return (0,) if sample_opcode == 90 else (0, 1, 2, 3)
            if value not in values:
                return None
            opcode, args = values[value]
            if opcode in (83, 124):  # CopyObject / Bitcast preserve all bits and lanes.
                return lanes(args[2])
            if opcode == 81:
                source = lanes(args[2])
                return None if source is None else (source[args[3]],)
            if opcode == 79:
                left, right = lanes(args[2]), lanes(args[3])
                return None if left is None or right is None else tuple((left + right)[index] for index in args[4:])
            return None

        if kind != "float":
            selected = tuple(index for index in range(4) if mask & (1 << index))
            casts = [lanes(args[2]) for opcode, args in instructions
                     if opcode == 124 and component(args[0])[0] == "float" and lanes(args[2]) is not None]
            assert selected in casts or tuple(index for cast in casts for index in cast) == selected, "integer sample bits or channel mask lost"
            assert not any(opcode in (111, 112) and lanes(args[2]) is not None
                           for opcode, args in instructions), "integer texture result was numerically converted to float"

    if "conversion_bits" in check:
        # Evaluate only the constant conversion's output slice, not Maxwell instruction flags.
        # This accepts either an emitted conversion or its constant-folded SPIR-V equivalent.
        memory = {args[0]: args[1] for opcode, args in instructions if opcode == 62}
        outputs = {args[1] for opcode, args in instructions if opcode == 59 and args[2] == 3}

        def output_pointer(value: int) -> bool:
            if value in outputs:
                return True
            opcode, args = values.get(value, (0, ()))
            return opcode == 65 and output_pointer(args[2])

        def bits(value: int):
            opcode, args = values[value]
            if opcode == 43:
                return args[2]
            if opcode == 46:  # OpConstantNull
                return 0
            if opcode == 1:  # OpUndef
                return None
            if opcode in (83, 124):
                return bits(args[2])
            if opcode == 61:
                return bits(memory[args[2]])
            if opcode in (44, 80):
                return tuple(bits(source) for source in args[2:])
            if opcode == 81:
                return bits(args[2])[args[3]]
            if opcode in (111, 112):
                number = bits(args[2])
                if opcode == 111 and number & 0x80000000:
                    number -= 1 << 32
                return struct.unpack("<I", struct.pack("<f", float(number)))[0]
            raise AssertionError(f"unexpected operation {opcode} in constant I2F output")

        actual = []
        for opcode, args in instructions:
            if opcode == 62 and output_pointer(args[0]):
                value = bits(args[1])
                actual.extend(value if isinstance(value, tuple) else (value,))
        assert actual and {value for value in actual if value is not None} == {check["conversion_bits"]}, "signed/unsigned I2F output value changed"


def run_typed(translator: Path, tmp: Path) -> int:
    """The typed image and texture cases: numeric contracts of each module besides spirv-val."""
    out = tmp / "typed_out"
    jobs = []
    checks = {}
    for key, instructions, options in typed_cases():
        options = dict(options)
        checks[key] = options.pop("_check", {})
        output_mask = options.pop("_omap", 15)
        code = tmp / (key + ".bin")
        code.write_bytes(fragment_code(instructions, options.get("sm86", False), output_mask))
        jobs.append({"key": key, "stage": "ps", "code": str(code), "debug": False,
                     "fout": {"0": "float"}, **options})
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
        module = spirv_instructions(data)
        if checks[job["key"]]:
            resources = json.loads(spv.with_suffix(".res.json").read_text(encoding="utf-8"))
            try:
                check_numeric_contract(module, checks[job["key"]], resources)
            except (AssertionError, KeyError, TypeError) as error:
                print(f"{job['key']}: numeric contract failed: {error}", file=sys.stderr)
                return 1
    return 0


def depth_test_module(against_texture: bool) -> bytes:
    """A fragment module: discard where gl_FragCoord.z is above a texture's red channel (or above the constant 0)."""
    (void, fn, flt, v4, v2, integer, boolean, p_in, p_in_f, p_out, image, sampled, p_tex, frag_coord, out, tex, two, zero,
     uv, black, main, entry, z_ptr, z, handle, texel, d, above, kill, merge) = range(1, 31)
    def string(text: str) -> list[int]:
        raw = text.encode() + b"\0"
        raw += b"\0" * (-len(raw) % 4)
        return list(struct.unpack(f"<{len(raw) // 4}I", raw))
    module = [
        (17, [1]), (14, [0, 1]), (15, [4, main, *string("main"), frag_coord, out]), (16, [main, 7]),
        (71, [frag_coord, 11, 15]), (71, [out, 30, 0]), (71, [tex, 34, 0]), (71, [tex, 33, 0]),
        (19, [void]), (33, [fn, void]), (22, [flt, 32]), (23, [v4, flt, 4]), (23, [v2, flt, 2]), (21, [integer, 32, 1]),
        (20, [boolean]), (32, [p_in, 1, v4]), (32, [p_in_f, 1, flt]), (32, [p_out, 3, v4]),
        (25, [image, flt, 1, 0, 0, 0, 1, 0]), (27, [sampled, image]), (32, [p_tex, 0, sampled]),
        (59, [p_in, frag_coord, 1]), (59, [p_out, out, 3]), (59, [p_tex, tex, 0]),
        (43, [integer, two, 2]), (43, [flt, zero, 0]), (44, [v2, uv, zero, zero]), (44, [v4, black, zero, zero, zero, zero]),
        (54, [void, main, 0, fn]), (248, [entry]),
        (65, [p_in_f, z_ptr, frag_coord, two]), (61, [flt, z, z_ptr]), (61, [sampled, handle, tex]),
        (88, [v4, texel, handle, uv, 2, zero]), (81, [flt, d, texel, 0]),
        (186, [boolean, above, z, d if against_texture else zero]),   # OpFOrdGreaterThan
        (247, [merge, 0]), (250, [above, kill, merge]), (248, [kill]), (252, []), (248, [merge]),
        (62, [out, black]), (253, []), (56, []),
    ]
    words = [0x07230203, 0x00010300, 0, 31, 0]
    for opcode, operands in module:
        words += [(len(operands) + 1) << 16 | opcode, *operands]
    return struct.pack(f"<{len(words)}I", *words)


def check_depth_snap(translator: Path, tmp: Path) -> int:
    """--frag-z-snap (tools/xlate/FragDepthSnap.cs, applied to fragment shaders on macOS): a comparison of gl_FragCoord.z
    with a texture value takes a snapped z (one OpFwidth, the comparison's z operand an OpSelect) and still validates;
    a comparison with a constant leaves the module byte-identical."""
    for name, against_texture in (("depth_snap", True), ("depth_snap_const", False)):
        source, result = tmp / (name + ".spv"), tmp / (name + "_out.spv")
        source.write_bytes(depth_test_module(against_texture))
        r = subprocess.run([str(translator), "--frag-z-snap", str(source), str(result)], capture_output=True, text=True, timeout=60)
        v = subprocess.run([str(SPIRV_VAL), "--target-env", "vulkan1.3", str(result)], capture_output=True, text=True)
        print(f"{name}: rewritten {r.stdout.strip()}, spirv-val exit {v.returncode} {v.stdout}{v.stderr}")
        if r.returncode or v.returncode:
            return 1
        module = spirv_instructions(result.read_bytes())
        if not against_texture:
            if r.stdout.strip() != "0" or result.read_bytes() != source.read_bytes():
                print(f"{name}: a comparison with a constant was changed", file=sys.stderr)
                return 1
            continue
        selects = {args[1] for opcode, args in module if opcode == 169}
        compare = [args for opcode, args in module if opcode == 186]
        if (r.stdout.strip() != "1" or sum(opcode == 209 for opcode, _ in module) != 1 or len(compare) != 1
                or compare[0][2] not in selects):
            print(f"{name}: the depth comparison was not rewritten as expected", file=sys.stderr)
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
        # A compute shader that reads only the layer count of texture handle 0x10, for which the translator has no type
        # answer: the texture must be declared as an array, or the count reads as 1.
        layer_code = tmp / "layer_count_cs.bin"
        layer_code.write_bytes(layer_count_code())
        jobs.append({"key": "layer_count_cs", "stage": "cs", "code": str(layer_code), "debug": False, "local": [1, 1, 1]})
        checks["layer_count_cs"] = {}
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
            if checks[job["key"]].get("mac_payload") and sys.platform == "darwin":
                words = struct.unpack(f"<{len(data) // 4}I", data)
                at, payloads = 5, 0
                while at < len(words):   # OpVariable (59) in TaskPayloadWorkgroupEXT (5402)
                    n, op = words[at] >> 16, words[at] & 0xFFFF
                    payloads += op == 59 and n >= 4 and words[at + 3] == 5402
                    at += max(n, 1)
                if payloads != 1:
                    print(f"{job['key']}: {payloads} payload variables, 1 expected on macOS", file=sys.stderr)
                    return 1
            if checks[job["key"]]:
                lowered = Path(job["lowered_out"]).read_bytes()[0 if "mesh" in job else 0x50:]
                words = [word[0] for i, word in enumerate(struct.iter_unpack("<Q", lowered)) if i % 4]
                expected_words = checks[job["key"]]["lowered"]
                if not any(words[i:i + len(expected_words)] == expected_words for i in range(len(words))):
                    print(f"{job['key']}: lowered operand contract failed: {[hex(word) for word in words]}", file=sys.stderr)
                    return 1
        layer_types = [t["type"] for t in json.loads((out / "layer_count_cs.res.json").read_text())["textures"]]
        if layer_types != [0x103]:  # SamplerType Texture2D (3) | Array (0x100)
            print(f"layer_count_cs: texture types {layer_types}, expected a 2D array [259]", file=sys.stderr)
            return 1
        return check_float_atomic(tmp) or run_typed(translator, tmp) or check_depth_snap(translator, tmp)


if __name__ == "__main__":
    sys.exit(main())
