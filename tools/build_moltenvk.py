"""Build the game's own MoltenVK for macOS: toolchain/moltenvk/ (MoltenVK at a pinned commit, merged with a pinned pull
request) and install it beside a game.

The system's MoltenVK (Homebrew) has no VK_EXT_mesh_shader, and the game draws its foliage, particles and glows with
mesh shaders: without them those draws are skipped. KhronosGroup/MoltenVK pull request 2842 adds the extension on Metal's
own mesh pipelines (task shaders only from Apple9 GPUs on; Apple7 and Apple8 get mesh shaders without the task stage).
This fetches MoltenVK at MVK_MAIN into toolchain/src/MoltenVK (local changes and build output there are discarded),
merges the pull request's commit MVK_PR, pins SPIRV-Cross at SPIRV_CROSS (its mesh shader fixes, KhronosGroup/SPIRV-Cross
pull request 2686, came after the revision MoltenVK pins), and builds with MoltenVK's own fetchDependencies and Makefile
(Xcode). The build fetches MoltenVK's dependencies (SPIRV-Cross, SPIRV-Tools, SPIRV-Headers, Vulkan-Headers, cereal)
from GitHub. libMoltenVK.dylib, its ICD manifest (MoltenVK_icd.json, naming the library by a relative path) and
MoltenVK's LICENSE go to toolchain/moltenvk; with --install GAME they are copied to GAME/moltenvk, where the game's
Vulkan loader takes them (runtime/vk_device.c moltenvk_beside_exe). Run with .venv, on macOS only.

  python tools/build_moltenvk.py [--src DIR] [--out DIR] [--install GAME_DIR] [--skip-build]

Nothing of this goes into a Windows or Linux build.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = 'https://github.com/KhronosGroup/MoltenVK.git'
MVK_MAIN = '5189f8de86ec365860f63bf1da736815cfaea5cd'   # MoltenVK main, 2026-11-10
MVK_PR = 'c516eb423516187204c68cd1ed82fb81520315eb'     # pull request 2842, "Add VK_EXT_mesh_shader support"
MVK_PR_REF = 'pull/2842/head'
SPIRV_CROSS = '57e4efcc84112abcd426dc5ec248b5339bee04f2'  # SPIRV-Cross main, 2026-10-28, with pull request 2686
LIB = 'Package/Release/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib'


def git(src, *args, capture=False):
    r = subprocess.run(['git', '-C', str(src), *args], check=True, text=True, capture_output=capture)
    return r.stdout.strip() if capture else None


def fetch(src):
    """src holds MVK_MAIN merged with MVK_PR, with SPIRV-Cross pinned, and no other local changes."""
    if not (src / '.git').is_dir():
        src.mkdir(parents=True, exist_ok=True)
        git(src, 'init', '-q')
        git(src, 'remote', 'add', 'origin', URL)
    git(src, 'fetch', '-q', 'origin', MVK_MAIN, f'+{MVK_PR_REF}:refs/remotes/origin/pr')
    if git(src, 'rev-parse', 'refs/remotes/origin/pr', capture=True) != MVK_PR:
        print(f'build_moltenvk: note: pull request head is now {git(src, "rev-parse", "refs/remotes/origin/pr", capture=True)}; '
              f'building the pinned {MVK_PR[:12]}')
        git(src, 'fetch', '-q', 'origin', MVK_PR)
    git(src, 'checkout', '-q', '--detach', MVK_MAIN)
    git(src, 'reset', '-q', '--hard', MVK_MAIN)
    git(src, 'clean', '-qfdx', '-e', 'External/')   # External/: the fetched dependencies, kept between builds
    git(src, '-c', 'user.name=build_moltenvk', '-c', 'user.email=build@localhost', 'merge', '-q', '--no-edit', MVK_PR)
    (src / 'ExternalRevisions' / 'SPIRV-Cross_repo_revision').write_text(SPIRV_CROSS + '\n')


def build(src):
    for cmd in (['./fetchDependencies', '--macos'], ['make', 'macos']):
        print('build_moltenvk:', ' '.join(cmd), flush=True)
        if subprocess.run(cmd, cwd=src).returncode:
            sys.exit(f'build_moltenvk: {" ".join(cmd)} failed')
    lib = src / LIB
    if not lib.is_file():
        sys.exit(f'build_moltenvk: {lib} was not built')
    return lib


def stage(src, out):
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / LIB, out / 'libMoltenVK.dylib')
    shutil.copy2(src / 'LICENSE', out / 'LICENSE-MoltenVK.txt')
    icd = {'file_format_version': '1.0.0',
           'ICD': {'library_path': './libMoltenVK.dylib', 'api_version': '1.4.0', 'is_portability_driver': True}}
    (out / 'MoltenVK_icd.json').write_text(json.dumps(icd, indent=4) + '\n')
    (out / 'VERSION.txt').write_text(f'MoltenVK {MVK_MAIN} + pull request 2842 {MVK_PR}; SPIRV-Cross {SPIRV_CROSS}\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--src', type=Path, default=ROOT / 'toolchain' / 'src' / 'MoltenVK')
    ap.add_argument('--out', type=Path, default=ROOT / 'toolchain' / 'moltenvk')
    ap.add_argument('--install', type=Path, help='game folder: copies the build to GAME/moltenvk')
    ap.add_argument('--skip-build', action='store_true', help='install what toolchain/moltenvk holds')
    a = ap.parse_args()
    if sys.platform != 'darwin':
        sys.exit('build_moltenvk: macOS only')
    if not a.skip_build:
        fetch(a.src)
        build(a.src)
        stage(a.src, a.out)
        print(f'build_moltenvk: {a.out}')
    if a.install:
        if not (a.out / 'libMoltenVK.dylib').is_file():
            sys.exit(f'build_moltenvk: nothing built in {a.out}')
        dst = a.install / 'moltenvk'
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(a.out, dst)
        print(f'build_moltenvk: installed into {dst}')


if __name__ == '__main__':
    main()
