"""Build and run tests/mesh_parity.c: the same geometry drawn by a vertex shader pipeline and by a mesh shader pipeline
through the game's own MoltenVK, compared pixel by pixel (colour and depth). macOS only; no window, no game files.

The mesh and task modules have the shape the translator gives the game's: each is a compute program in GLSL
(tests/mesh_parity/mesh.comp, task.comp) that fills the stage buffer in shared memory, turned into a MeshEXT / TaskEXT
module by tools/xlate/MeshSpirv.cs itself (tests/mesh_parity/convert builds that file into a small program). Outputs the
translator does not write today (a clip distance, per-primitive values) and a mesh entry that writes another
thread's vertices are added to the converted module's assembly here.

Needs: the game's MoltenVK (tools/build_moltenvk.py --install), the Vulkan headers and loader, glslangValidator,
spirv-as, spirv-dis and spirv-val (Homebrew: vulkan-headers vulkan-loader glslang spirv-tools), the .NET SDK.

A flat input is expected from its primitive's last vertex, as the game expects it. With a driver that has
provokingVertexLast (VK_EXT_provoking_vertex) both pipelines ask for it, as runtime/vk_pipeline.c does. Without it, or
with --rotate, the mesh modules are converted with the translator's "flat_first" option (each triangle handed over last
vertex first) and the vertex pipeline's shader stands in for the last vertex; lines then keep the first.

Usage: mesh_parity.py [--moltenvk DIR] [--rotate] [--filter TEXT] [--mvk-debug]
  --moltenvk   the folder with MoltenVK_icd.json (default ~/Games/NX2navi/Ocarina/moltenvk)
  --rotate     test the path of a driver without provokingVertexLast even if this one has it
  --filter     run only the cases whose name contains TEXT
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'tests' / 'mesh_parity'
WORK = ROOT / 'build' / 'test' / 'mesh_parity'

POS = [0x70, 0x74, 0x78, 0x7c]
PSIZE = [0x6c]


def generic(g, comps=4):
    return [0x80 + 16 * g + 4 * c for c in range(comps)]


STD = POS + generic(0) + generic(1) + generic(2)
BIG = POS + generic(0) + generic(1) + generic(2) + generic(3) + generic(4, 3) + generic(5, 1) + generic(6, 1) + generic(7, 1)
PAYLOAD = 256 + 0x20        # what tools/xlate/Program.cs gives a task whose size field is 0
BIG_PAYLOAD = 3072 + 0x20   # the game's foliage task

# name: threads, vertices, primitives, topology (1 points, 2 lines, 3 triangles), output lines, task payload bytes (0:
# no task stage), defines of the vertex and fragment shaders, extra (outputs added to the assembly)
VARIANTS = {
    'tri': (32, 96, 32, 3, STD, 0, [], None),
    'grid': (32, 64, 126, 3, STD, 0, ['GRID'], None),
    'tgrid': (32, 64, 126, 3, STD, PAYLOAD, ['GRID'], None),
    'line': (32, 64, 32, 2, STD, 0, [], None),
    'point': (32, 32, 32, 1, PSIZE + STD, 0, ['PSIZE'], None),
    'point_nosize': (32, 32, 32, 1, STD, 0, [], None),
    'count': (32, 1, 1, 1, PSIZE + STD, 0, ['PSIZE'], None),
    'tcount': (32, 1, 1, 1, PSIZE + STD, PAYLOAD, ['PSIZE'], None),
    'ttri': (32, 96, 32, 3, STD, PAYLOAD, [], None),
    'big': (32, 160, 53, 3, BIG, 0, ['BIG'], None),
    'tbig': (32, 160, 53, 3, BIG, BIG_PAYLOAD, ['BIG'], None),
    'bigx': (32, 160, 53, 3, BIG, 0, ['BIG'], 'cross'),
    'clip': (32, 96, 32, 3, STD, 0, ['CLIP'], 'clip'),
    'prim': (32, 96, 32, 3, STD, 0, ['PRIM'], 'prim'),
    'layer': (32, 96, 32, 3, STD, 0, ['PRIM', 'LAYER'], 'layer'),
}


def tool(name):
    for d in ('/opt/homebrew/bin', '/usr/local/bin'):
        if (Path(d) / name).is_file():
            return str(Path(d) / name)
    path = shutil.which(name)
    if not path:
        sys.exit(f'mesh_parity: {name} not found')
    return path


def run(*cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode:
        sys.exit(f'mesh_parity: {" ".join(str(c) for c in cmd)}\n{r.stdout}{r.stderr}')
    return r.stdout


def glsl(src, stage, out, defines, body_defines=''):
    """Compile src for Vulkan 1.0 SPIR-V (storage buffers as BufferBlock, which the translator's own modules have)."""
    text = (SRC / src).read_text()
    first, rest = text.split('\n', 1)
    tmp = out.with_suffix('.glsl')
    tmp.write_text(first + '\n' + ''.join(f'#define {d}\n' for d in defines) + body_defines + rest)
    env = '--target-env', ('vulkan1.0' if stage == 'comp' else 'vulkan1.2')
    run(tool('glslangValidator'), '-V', *env, '-S', stage, '-o', out, tmp)


def layout(threads, maxv, maxp, topo, lines, payload):
    """tools/xlate/AmpereLowering.cs MeshInfo.Layout, for the GLSL's defines; the converter checks these numbers."""
    ipp = {1: 1, 2: 2}.get(topo, 3)
    groups = (maxv + 31) // 32
    amap = (groups * len(lines) * 128 + 15) & ~15
    pay = (amap + 4 + ipp * maxp + 15) & ~15
    return {'attr': 0, 'map': amap, 'pay': pay, 'shared': (pay + payload) or 16}


def line_values(lines):
    out = []
    for n, a in enumerate(lines):
        if a == 0x6c:
            v = 'geo.data[b + 9].x'
        elif a < 0x80:
            v = f'geo.data[b][{(a - 0x70) // 4}]'
        else:
            v = f'geo.data[b + {1 + (a - 0x80) // 16}][{(a - 0x80) % 16 // 4}]'
        out.append(f'L({n}, {v})')
    return '#define LINE_VALUES ' + ' '.join(out) + '\n'


class Asm:
    """A converted module's assembly (spirv-dis --raw-id), for adding what the translator does not write."""

    def __init__(self, path):
        self.path = path
        self.lines = run(tool('spirv-dis'), '--raw-id', path).splitlines()
        self.n = 0

    def find(self, pattern):
        for line in self.lines:
            m = re.search(pattern, line)
            if m:
                return m.group(1)
        return None

    def must(self, pattern):
        r = self.find(pattern)
        if r is None:
            sys.exit(f'mesh_parity: {self.path.name}: no "{pattern}" in the converted module')
        return r

    def new(self, stem='t'):
        self.n += 1
        return f'%x_{stem}{self.n}'

    def insert_before(self, pattern, new):
        at = next(i for i, line in enumerate(self.lines) if re.search(pattern, line))
        self.lines[at:at] = new

    def add_globals(self, decorations, globals_, interface):
        self.insert_before(r'= OpType|= OpConstant', decorations)   # the first type: the end of the annotations
        self.insert_before(r'= OpFunction ', globals_)
        at = next(i for i, line in enumerate(self.lines) if 'OpEntryPoint' in line)
        self.lines[at] += ''.join(' ' + v for v in interface)

    def write(self):
        text = self.path.with_suffix('.spvasm')
        text.write_text('\n'.join(self.lines) + '\n')
        run(tool('spirv-as'), '--target-env', 'spv1.4', '-o', self.path, text)

    # ids of what MeshSpirv.Convert made
    def common(self):
        self.uint = '%' + self.must(r'%(\d+) = OpTypeInt 32 0')
        self.float = '%' + self.must(r'%(\d+) = OpTypeFloat 32')
        self.shared = '%' + self.must(r'%(\d+) = OpVariable %\d+ Workgroup')
        self.p_shared = '%' + self.must(rf'%(\d+) = OpTypePointer Workgroup {self.uint}$')

    def shared_word(self, index, out):
        p, v = self.new(), self.new()
        out += [f'{p} = OpAccessChain {self.p_shared} {self.shared} {index}', f'{v} = OpLoad {self.uint} {p}']
        return v


def add_cross(path):
    """Every thread writes its neighbour's vertices and primitives (thread t those of t ^ 1) instead of its own: all
    are still written once."""
    a = Asm(path)
    a.common()
    lii = a.must(r'OpDecorate %(\d+) BuiltIn LocalInvocationIndex')
    at = max(i for i, line in enumerate(a.lines) if re.search(rf'= OpLoad {a.uint} %{lii}$', line))
    tid = a.lines[at].split()[0]
    a.lines[at + 1:] = [re.sub(rf'{tid}\b', '%x_other', line) for line in a.lines[at + 1:]]
    a.lines.insert(at + 1, f'%x_other = OpBitwiseXor {a.uint} {tid} %x_one')
    a.add_globals([], [f'%x_one = OpConstant {a.uint} 1'], [])
    a.write()


def add_clip(path, maxv, words):
    """gl_ClipDistance[1] per vertex, as an array of its own, from shared word words + v."""
    a = Asm(path)
    a.common()
    pos = a.must(r'OpDecorate %(\d+) BuiltIn Position')
    g = [f'%x_n = OpConstant {a.uint} {maxv}', f'%x_one = OpConstant {a.uint} 1',
         f'%x_zero = OpConstant {a.uint} 0', f'%x_words = OpConstant {a.uint} {words}',
         f'%x_a1 = OpTypeArray {a.float} %x_one', '%x_an = OpTypeArray %x_a1 %x_n', '%x_pan = OpTypePointer Output %x_an',
         f'%x_pf = OpTypePointer Output {a.float}', '%x_clip = OpVariable %x_pan Output']
    a.add_globals(['OpDecorate %x_clip BuiltIn ClipDistance'], g, ['%x_clip'])
    cap = next(i for i, line in enumerate(a.lines) if 'OpCapability' in line)
    a.lines[cap:cap] = ['OpCapability ClipDistance']
    out, done = [], 0
    for line in a.lines:
        m = re.search(rf'= OpAccessChain %\d+ %{pos} (%\d+) (%\d+)$', line)
        out.append(line)
        if m and a.find(rf'{m.group(2)} = OpConstant {a.uint} (0)$') is not None:   # component x: once per vertex
            v, add = m.group(1), []
            at = a.new()
            add += [f'{at} = OpIAdd {a.uint} {v} %x_words']
            word = a.shared_word(at, add)
            f, p = a.new(), a.new()
            add += [f'{f} = OpBitcast {a.float} {word}', f'{p} = OpAccessChain %x_pf %x_clip {v} %x_zero', f'OpStore {p} {f}']
            out[-1:-1] = add
            done += 1
    if not done:
        sys.exit('mesh_parity: clip: no position store found')
    a.lines = out
    a.write()


def add_prim(path, maxp, words, layer):
    """Per-primitive outputs as arrays of their own: gl_PrimitiveID and a colour at location 3, with layer gl_Layer and
    gl_ViewportIndex too, from shared words words + 8 p (the primitive's number, then the colour from word 4)."""
    a = Asm(path)
    a.common()
    idx = a.must(r'OpDecorate %(\d+) BuiltIn PrimitiveTriangleIndicesEXT')
    v4 = '%' + a.must(rf'%(\d+) = OpTypeVector {a.float} 4')
    g = [f'%x_n = OpConstant {a.uint} {maxp}', f'%x_one = OpConstant {a.uint} 1', f'%x_eight = OpConstant {a.uint} 8',
         f'%x_words = OpConstant {a.uint} {words}']
    sint = a.find(r'%(\d+) = OpTypeInt 32 1')
    if sint is None:
        g.append('%x_int = OpTypeInt 32 1')
        sint = 'x_int'
    sint = '%' + sint
    g += [f'%x_ai = OpTypeArray {sint} %x_n', '%x_pai = OpTypePointer Output %x_ai', f'%x_pi = OpTypePointer Output {sint}',
          f'%x_av = OpTypeArray {v4} %x_n', '%x_pav = OpTypePointer Output %x_av', f'%x_pv = OpTypePointer Output {v4}',
          '%x_id = OpVariable %x_pai Output', '%x_colour = OpVariable %x_pav Output']
    outs = [('%x_id', 'PrimitiveId')]
    if layer:
        g += ['%x_layer = OpVariable %x_pai Output', '%x_view = OpVariable %x_pai Output']
        outs += [('%x_layer', 'Layer'), ('%x_view', 'ViewportIndex')]
    g += [f'%x_c{k} = OpConstant {a.uint} {k}' for k in (4, 5, 6, 7)]
    dec = [f'OpDecorate {v} BuiltIn {b}' for v, b in outs] + ['OpDecorate %x_colour Location 3']
    dec += [f'OpDecorate {v} PerPrimitiveEXT' for v in [v for v, _ in outs] + ['%x_colour']]
    a.add_globals(dec, g, [v for v, _ in outs] + ['%x_colour'])
    out, done = [], 0
    for line in a.lines:
        m = re.search(rf'= OpAccessChain %\d+ %{idx} (%\d+)$', line)
        if m:
            p, add = m.group(1), []
            at, at2 = a.new(), a.new()
            add += [f'{at} = OpIMul {a.uint} {p} %x_eight', f'{at2} = OpIAdd {a.uint} {at} %x_words']
            number = a.shared_word(at2, add)
            low, high, bit = a.new(), a.new(), a.new()
            add += [f'{low} = OpBitwiseAnd {a.uint} {number} %x_one', f'{high} = OpShiftRightLogical {a.uint} {number} %x_one',
                    f'{bit} = OpBitwiseAnd {a.uint} {high} %x_one']
            for var, value in (('%x_id', number), ('%x_layer', low), ('%x_view', bit))[:len(outs)]:
                s, ptr = a.new(), a.new()
                add += [f'{s} = OpBitcast {sint} {value}', f'{ptr} = OpAccessChain %x_pi {var} {p}', f'OpStore {ptr} {s}']
            comps = []
            for k in (4, 5, 6, 7):
                i, f = a.new(), a.new()
                add += [f'{i} = OpIAdd {a.uint} {at2} %x_c{k}']
                word = a.shared_word(i, add)
                add += [f'{f} = OpBitcast {a.float} {word}']
                comps.append(f)
            c, ptr = a.new(), a.new()
            add += [f'{c} = OpCompositeConstruct {v4} {" ".join(comps)}', f'{ptr} = OpAccessChain %x_pv %x_colour {p}',
                    f'OpStore {ptr} {c}']
            out += add
            done += 1
        out.append(line)
    if not done:
        sys.exit('mesh_parity: prim: no index store found')
    a.lines = out
    a.write()


def build_shaders(spv):
    jobs, extras = [], []
    for name, (threads, maxv, maxp, topo, lines, payload, defs, extra) in VARIANTS.items():
        ipp = {1: 1, 2: 2}.get(topo, 3)
        lay = layout(threads, maxv, maxp, topo, lines, payload)
        words = lay['shared'] // 4
        d = [f'THREADS {threads}', f'MAXV {maxv}', f'MAXP {maxp}', f'IPP {ipp}', f'NLINES {len(lines)}',
             f'ATTR {lay["attr"]}', f'MAP {lay["map"]}', f'PAY {lay["pay"]}']
        if payload:
            d.append('HASTASK')
        if extra == 'clip':
            d.append(f'CLIPWORDS {words}u')
            extras += [(add_clip, spv / f'{name}.{k}.spv', maxv, words) for k in ('mesh', 'rmesh')]
            words += maxv
        elif extra in ('prim', 'layer'):
            d.append(f'PRIMWORDS {words}u')
            extras += [(add_prim, spv / f'{name}.{k}.spv', maxp, words, extra == 'layer') for k in ('mesh', 'rmesh')]
            words += 8 * maxp
        elif extra == 'cross':
            extras += [(add_cross, spv / f'{name}.{k}.spv') for k in ('mesh', 'rmesh')]
        d.append(f'SHARED_WORDS {words}')
        d += [x for x in defs if x == 'GRID']
        glsl('mesh.comp', 'comp', spv / f'{name}.mesh.comp.spv', d, line_values(lines))
        for kind, flat_first in (('mesh', False), ('rmesh', True)):   # rmesh: for a driver without provokingVertexLast
            jobs.append(dict(lay, **{'in': str(spv / f'{name}.mesh.comp.spv'), 'out': str(spv / f'{name}.{kind}.spv'),
                                     'stage': 7, 'threads': threads, 'maxv': maxv, 'maxp': maxp, 'topo': topo,
                                     'payload': payload, 'task': bool(payload), 'lines': lines, 'flat_first': flat_first}))
        if payload:
            glsl('task.comp', 'comp', spv / f'{name}.task.comp.spv', [f'THREADS {threads}', f'SHARED_WORDS {payload // 4}'])
            jobs.append({'in': str(spv / f'{name}.task.comp.spv'), 'out': str(spv / f'{name}.task.spv'), 'stage': 6,
                         'threads': threads, 'maxv': 0, 'maxp': 0, 'topo': topo, 'payload': payload, 'task': False,
                         'lines': [], 'attr': 0, 'map': 0, 'pay': 0, 'shared': payload})
        vdefs = [f'IPP {ipp}'] + defs
        glsl('scene.vert', 'vert', spv / f'{name}.vert.spv', vdefs)
        glsl('scene.frag', 'frag', spv / f'{name}.frag.spv', defs)
        glsl('scene.frag', 'frag', spv / f'{name}.mfrag.spv', defs + (['PRIM_MESH'] if extra in ('prim', 'layer') else []))
    (spv / 'jobs.json').write_text(json.dumps(jobs))
    # MeshSpirv.cs as a program of its own; its build output stays under build/
    obj = WORK / 'convert-obj'
    run(tool('dotnet'), 'build', SRC / 'convert' / 'convert.csproj', '-c', 'Release', '--nologo', '-v', 'q',
        '-o', WORK / 'convert', f'-p:BaseIntermediateOutputPath={obj}/', f'-p:MSBuildProjectExtensionsPath={obj}/')
    run(tool('dotnet'), WORK / 'convert' / 'mesh_parity_convert.dll', spv / 'jobs.json')
    for fn, *args in extras:
        fn(*args)
    for j in jobs:
        run(tool('spirv-val'), '--target-env', 'vulkan1.2', j['out'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--moltenvk', type=Path, default=Path.home() / 'Games' / 'NX2navi' / 'Ocarina' / 'moltenvk')
    ap.add_argument('--rotate', action='store_true')
    ap.add_argument('--filter', default='')
    ap.add_argument('--mvk-debug', action='store_true', help="MoltenVK's own log, with every shader's MSL")
    a = ap.parse_args()
    if sys.platform != 'darwin':
        print('mesh_parity: skipped (macOS only)')
        return 0
    icd = a.moltenvk / 'MoltenVK_icd.json'
    if not icd.is_file():
        sys.exit(f'mesh_parity: {icd} not found (tools/build_moltenvk.py --install)')
    spv = WORK / 'spv'
    shutil.rmtree(spv, ignore_errors=True)
    spv.mkdir(parents=True)
    build_shaders(spv)
    prefix = next((p for p in ('/opt/homebrew', '/usr/local') if (Path(p) / 'include' / 'vulkan' / 'vulkan.h').is_file()), None)
    if not prefix:
        sys.exit('mesh_parity: the Vulkan headers were not found (brew install vulkan-headers vulkan-loader)')
    exe = WORK / 'mesh_parity'
    run(os.environ.get('CC') or 'clang', '-std=gnu11', '-O1', '-g', '-Wall', '-I', f'{prefix}/include', ROOT / 'tests' /
        'mesh_parity.c', '-o', exe, f'-L{prefix}/lib', '-lvulkan', f'-Wl,-rpath,{prefix}/lib')
    env = dict(os.environ, VK_DRIVER_FILES=str(icd), VK_ICD_FILENAMES=str(icd))
    env.pop('VK_ADD_DRIVER_FILES', None)
    if a.mvk_debug:
        env.update(MVK_CONFIG_DEBUG='1', MVK_CONFIG_LOG_LEVEL='3')
    else:
        env.setdefault('MVK_CONFIG_LOG_LEVEL', '1')
    return subprocess.run([str(exe), str(spv), 'rotate' if a.rotate else 'auto', a.filter], env=env).returncode


if __name__ == '__main__':
    sys.exit(main())
