"""Build the game's own MoltenVK for macOS: toolchain/moltenvk/ (MoltenVK at a pinned commit, merged with a pinned pull
request) and install it beside a game.

The system's MoltenVK (Homebrew) has no VK_EXT_mesh_shader, and the game draws its foliage, particles and glows with
mesh shaders: without them those draws are skipped. KhronosGroup/MoltenVK pull request 2842 adds the extension on Metal's
own mesh pipelines (task shaders only from Apple9 GPUs on; PATCHES below give them to Apple7 and Apple8 too), and
SPIRV_CROSS_PATCHES fix SPIRV-Cross's mesh stage.
This fetches MoltenVK at MVK_MAIN into toolchain/src/MoltenVK (local changes and build output there are discarded),
merges the pull request's commit MVK_PR, pins SPIRV-Cross at SPIRV_CROSS (its mesh shader fixes, KhronosGroup/SPIRV-Cross
pull request 2686, came after the revision MoltenVK pins), applies PATCHES, and builds with MoltenVK's own fetchDependencies and Makefile
(Xcode). The build fetches MoltenVK's dependencies (SPIRV-Cross, SPIRV-Tools, SPIRV-Headers, Vulkan-Headers, cereal)
from GitHub.

The build defines MVK_USE_METAL_PRIVATE_API=1 (MAKE_ARGS): MoltenVK then also calls Metal methods that are not in
Apple's public headers, which an App Store application may not do and which Apple may change in any macOS release (each
call is guarded by respondsToSelector, so a missing one is skipped). This game's renderer needs one of them:
VK_EXT_provoking_vertex's provokingVertexLast (the encoder's setProvokingVertexMode:), because the game takes a flat
fragment input from its primitive's last vertex and Metal's public API only from the first. The switch also brings:
logicOp (the pipeline descriptor's logic operation; the back end uses it when the game asks for one), wideLines and
the dynamic line width (the back end always draws lines 1 pixel wide), the static sample mask on the pipeline
descriptor instead of in the fragment shader (the back end sets none), primitive restart that can be turned off and
that follows the index type (VK_EXT_primitive_topology_list_restart; MoltenVK then also puts every render pass
descriptor into Metal's "OpenGL mode", which unlocks that call), byte indices widened without their restart value
rewritten, VK_EXT_non_seamless_cube_map and VK_EXT_legacy_dithering (the back end uses neither).

libMoltenVK.dylib, its ICD manifest (MoltenVK_icd.json, naming the library by a relative path) and
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
MAKE_ARGS = ['MVK_USE_METAL_PRIVATE_API=1']   # see the docstring: provokingVertexLast and what else it turns on

# Changes made to the merged source, as (file, text, replacement); each text must occur once.
#
# Mesh draws without a task shader, in blocks of 1024 mesh threadgroups. Metal launches a mesh grid from an object
# threadgroup, and one object threadgroup launches at most 1024 mesh threadgroups; a larger grid failed the command
# buffer ("Insufficient Memory") and the device was lost. The game's foliage draws up to about 1,800 workgroups without a
# task shader. A pipeline with a mesh shader and no task shader gets MoltenVK's own object function on every Apple GPU
# (not only those without indirect mesh draws). A draw launches MESH_BLOCKS object threadgroups (a direct draw: as many
# as its count needs); threadgroup k launches workgroups [1024 k, 1024 k + 1024) of the draw's count and writes 1024 k into
# its payload, and the game's mesh shaders, as our translator writes them for macOS (tools/xlate/MeshSpirv.cs), add that
# base to their workgroup ID. A grid of more than one row is one block, clamped to 1024.
SHADERS = 'MoltenVK/MoltenVK/Commands/MVKCommandPipelineStateFactoryShaderSource.h'
PIPELINE = 'MoltenVK/MoltenVK/GPUObjects/MVKPipeline.mm'
DRAW = 'MoltenVK/MoltenVK/Commands/MVKCmdDraw.mm'
MESH_BLOCKS = 8   # indirect draws: up to 8192 workgroups
MSL_VERSION = 4   # one more with each change to SPIRV_CROSS_PATCHES: 1 mesh positions flipped, 2 outputs per thread,
                  # 3 unwritten fragment inputs, 4 mesh positions' depth range
PATCHES = [
    (SHADERS,
     '[[object]] void objCmdDrawMeshTasks(const device MTLDispatchThreadgroupsIndirectArguments& args [[buffer(0)]],\n'
     '                                    mesh_grid_properties grid) {\n'
     '\tgrid.set_threadgroups_per_grid(uint3(args.threadgroupsPerGrid[0], args.threadgroupsPerGrid[1], '
     'args.threadgroupsPerGrid[2]));\n'
     '}',
     '[[object]] void objCmdDrawMeshTasks(const device MTLDispatchThreadgroupsIndirectArguments& args [[buffer(0)]],\n'
     '                                    object_data uint& base [[payload]],\n'
     '                                    uint block [[threadgroup_position_in_grid]],\n'
     '                                    mesh_grid_properties grid) {\n'
     '\tuint3 g = uint3(args.threadgroupsPerGrid[0], args.threadgroupsPerGrid[1], args.threadgroupsPerGrid[2]);\n'
     '\tif (g.y * g.z != 1u) {\n'
     '\t\tbase = 0u;\n'
     '\t\tif (block != 0u) { g = uint3(0u); }\n'
     '\t\telse if (g.x * g.y * g.z > 1024u) { g = uint3(min(g.x, 1024u), min(g.y, 1024u / max(min(g.x, 1024u), 1u)), 1u); }\n'
     '\t\tgrid.set_threadgroups_per_grid(g);\n'
     '\t\treturn;\n'
     '\t}\n'
     '\tuint first = block * 1024u;\n'
     '\tbase = first;\n'
     '\tgrid.set_threadgroups_per_grid(uint3(g.x > first ? min(g.x - first, 1024u) : 0u, 1u, 1u));\n'
     '}'),
    (PIPELINE,
     '\t} else if ( !getPhysicalDevice()->getMTLDeviceCapabilities().supportsApple9 ) {\n'
     '\t\t// without indirect mesh draws, an object function reads each draw\'s mesh grid from a buffer\n'
     '\t\tplDesc.objectFunction = [factory->newFunctionNamed("objCmdDrawMeshTasks") autorelease];\n'
     '\t\t_hasMeshGridObject = true;\n'
     '\t}',
     '\t} else {\n'
     '\t\t// an object function reads each draw\'s mesh grid from a buffer and launches it in blocks of 1024 mesh\n'
     '\t\t// threadgroups, the block\'s first workgroup in its payload\n'
     '\t\tplDesc.objectFunction = [factory->newFunctionNamed("objCmdDrawMeshTasks") autorelease];\n'
     '\t\tplDesc.payloadMemoryLength = 16;\n'
     '\t\t_hasMeshGridObject = true;\n'
     '\t}'),
    (DRAW,
     '\t\tcmdEncoder->getMtlGraphics().bindObjectBytes(cmdEncoder->_mtlRenderEncoder, grid, sizeof(grid), 0);\n'
     '\t\tgroupCount = MTLSizeMake(1, 1, 1);',
     '\t\tcmdEncoder->getMtlGraphics().bindObjectBytes(cmdEncoder->_mtlRenderEncoder, grid, sizeof(grid), 0);\n'
     '\t\tgroupCount = MTLSizeMake(grid[1] * grid[2] == 1 ? (grid[0] + 1023) / 1024 : 1, 1, 1);'),
    (DRAW,
     '\t\t\tcmdEncoder->getMtlGraphics().bindObjectBuffer(cmdEncoder->_mtlRenderEncoder, indirectBuffer, '
     'indirectBufferOffset + drawIdx * indirectBufferStride, 0);\n'
     '\t\t\t[cmdEncoder->_mtlRenderEncoder drawMeshThreadgroups: MTLSizeMake(1, 1, 1)',
     '\t\t\tcmdEncoder->getMtlGraphics().bindObjectBuffer(cmdEncoder->_mtlRenderEncoder, indirectBuffer, '
     'indirectBufferOffset + drawIdx * indirectBufferStride, 0);\n'
     f'\t\t\t[cmdEncoder->_mtlRenderEncoder drawMeshThreadgroups: MTLSizeMake({MESH_BLOCKS}, 1, 1)'),
    # Task shaders on Apple7 and Apple8 GPUs (the M1 and M2). MoltenVK offers them from Apple9 on only, on the grounds
    # that the older GPUs have no indirect mesh draw. They have one for an object grid: on an M1 Max (Apple7),
    # drawMeshThreadgroupsWithIndirectBuffer with an object function launched exactly the grid in the buffer (0, 1, 64,
    # 1024, 3880, 70000 threadgroups and a 300x4x2 grid, counted by the object and mesh functions; 2026-11-10), which
    # is how MoltenVK draws an indirect task draw. The mesh grid each task workgroup launches stays within 1024: the
    # game's task shaders, as our translator writes them, clamp their mesh count to their thread count.
    ('MoltenVK/MoltenVK/GPUObjects/MVKDevice.mm',
     '\t\t\t\t// without indirect mesh draws, indirect task draws have no stage left to read their grid\n'
     '\t\t\t\tmeshFeatures->taskShader = _gpuCapabilities.supportsApple9;\n',
     '\t\t\t\t// Apple7 and Apple8 launch an object grid read from a buffer too (drawMeshThreadgroupsWithIndirectBuffer)\n'
     '\t\t\t\tmeshFeatures->taskShader = _gpuCapabilities.supportsApple7;\n'),
    # The mesh stage's shared memory limit, with its outputs out of threadgroup memory (SPIRV_CROSS_PATCHES): all of
    # Metal's 32768 bytes but the 8 of the vertex and primitive counts, instead of half of what a worst case of outputs
    # left (12284). A shader whose outputs stay in threadgroup memory (an output block) and that goes over fails at
    # pipeline creation.
    ('MoltenVK/MoltenVK/GPUObjects/MVKDevice.mm',
     '\t\t\t\tuint32_t meshShared = (lim.maxComputeSharedMemorySize - sizeof(uint32_t[2]) - meshPrims * sizeof(uint32_t[4])) / 2;\n',
     '\t\t\t\tuint32_t meshShared = lim.maxComputeSharedMemorySize - sizeof(uint32_t[2]);\n'),
    # SPIRV_CROSS_PATCHES change the MSL of shaders the pipeline cache may hold from an earlier build, keyed by their
    # SPIR-V and options only: another cache UUID makes the driver refuse such a cache.
    ('MoltenVK/MoltenVK/GPUObjects/MVKDevice.mm',
     '\tmtlFeatures |= _isUsingMetalArgumentBuffers << 0;\n',
     '\tmtlFeatures |= _isUsingMetalArgumentBuffers << 0;\n'
     f'\tmtlFeatures |= {MSL_VERSION}u << 9;\t// the MSL the patched SPIRV-Cross writes\n'),
    # The cull mode and the fill mode of a mesh pipeline. Metal culls and fills triangles only, so MoltenVK sets both
    # on the encoder only for a pipeline whose primitive type is a triangle one, and a mesh pipeline, which has no
    # input assembly state, was given the point type: a mesh draw kept the cull mode and fill mode of the last
    # triangle draw of its render pass before it (none and fill at the start of a pass), whatever its own state asked
    # for. A mesh pipeline now counts as drawing triangles there; Metal ignores both for the lines or points a mesh
    # shader may emit instead.
    (PIPELINE,
     '\t_vkPrimitiveTopology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;\n',
     '\t// a mesh pipeline: triangles, so that its cull mode and fill mode reach the encoder\n'
     '\t_vkPrimitiveTopology = _isMeshPipeline ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : VK_PRIMITIVE_TOPOLOGY_POINT_LIST;\n'),
    # fetchDependencies checks out each dependency at its revision with --force, which would drop SPIRV_CROSS_PATCHES:
    # a repository already at its revision is left as it is (build() resets SPIRV-Cross itself before patching it).
    ('fetchDependencies',
     '\t\tgit cat-file -e $3 || git fetch --all\n'
     '\t\tgit checkout --force $3\n',
     '\t\tgit cat-file -e $3 || git fetch --all\n'
     '\t\t[ "$(git rev-parse HEAD)" = "$(git rev-parse $3^{commit})" ] || git checkout --force $3\n'),
    # Buffers read through device addresses. With a residency set, every buffer's memory is in it from its creation
    # (makeResident), and the queue uses the set for each command buffer. MoltenVK also added every buffer made with
    # device address usage to each encoder's useResources, whether the submission reads it or not: over 4,000 of the
    # game's pool buffers each time, unbound ones as nil entries, and a buffer destroyed while such a command buffer
    # ran failed it ("Invalid Resource"). With a residency set that list is not encoded.
    ('MoltenVK/MoltenVK/GPUObjects/MVKDevice.mm',
     'void MVKDevice::encodeGPUAddressableBuffers(MVKUseResourceHelper& resources, MVKResourceUsageStages stage) {\n'
     '\tlock_guard<mutex> lock(_rezLock);\n',
     'void MVKDevice::encodeGPUAddressableBuffers(MVKUseResourceHelper& resources, MVKResourceUsageStages stage) {\n'
     '\tif (hasResidencySet()) { return; }\n'
     '\tlock_guard<mutex> lock(_rezLock);\n'),
]


# Changes made to SPIRV-Cross (External/SPIRV-Cross, checked out at SPIRV_CROSS by fetchDependencies), as PATCHES; build()
# applies them between the checkout and the build.
#
# Mesh shader positions flipped for Metal. MoltenVK asks SPIRV-Cross to negate each vertex's y (Vulkan's clip space
# points y down, Metal's up), and the mesh stage did so only for a position that is a member of an output block. Our
# translator declares the mesh stage's gl_Position as an array variable of its own (tools/xlate/MeshSpirv.cs), so every
# vertex of the game's mesh shaders came out mirrored top to bottom, with its winding reversed: foliage hanging from
# the sky, rocks in the air, a door culled inside out.
SPIRV_CROSS_PATCHES = [
    ('spirv_msl.cpp',
     '\t\t\tBuiltIn builtin = BuiltInMax;\n'
     '\t\t\tstd::string access;\n'
     '\t\t\tif (orig_type.basetype == SPIRType::Struct)\n',
     '\t\t\tBuiltIn builtin = BuiltInMax;\n'
     '\t\t\tstd::string access;\n'
     '\t\t\t// a position declared as a variable of its own, not as a member of a block, is flipped too\n'
     '\t\t\tif (orig_type.basetype != SPIRType::Struct && has_decoration(orig_var, DecorationBuiltIn) &&\n'
     '\t\t\t    BuiltIn(get_decoration(orig_var, DecorationBuiltIn)) == BuiltInPosition)\n'
     '\t\t\t\tbuiltin = BuiltInPosition;\n'
     '\t\t\tif (orig_type.basetype == SPIRType::Struct)\n'),
    # Mesh shader outputs kept by each thread instead of in threadgroup memory. Metal allows a mesh function 32768 bytes
    # of threadgroup memory in all (a pipeline over it fails at its creation: "Threadgroup memory size (N) exceeds the
    # maximum threadgroup memory allowed (32768)"; measured on an M1 Max, the sum of the function's threadgroup
    # variables that its fragment function leaves alive). SPIRV-Cross keeps every output array there, 16 bytes per
    # vector, vertex and primitive, beside the shader's own shared memory: the game's larger foliage shaders (160
    # vertices, nine vectors, 13008 to 17040 bytes of shared memory) came to 34520 to 47256 bytes and were not drawn.
    # An output array that is not a block, in a shader with a literal workgroup size of T threads, becomes a
    # spvMeshThreadArray: each thread holds elements t, t + T, t + 2T, ..., the ones the entry point's last loops hand
    # to the mesh from that thread. Our translator's mesh entry writes each vertex and primitive from that same
    # thread (tools/xlate/MeshSpirv.cs: "every thread copies vertices i, i + threads, ..."); a shader that writes
    # another thread's element would lose it. The whole array as a local of each thread was seven times slower (21 ms
    # against 3 for 8192 mesh threadgroups); the per-thread part is as fast as threadgroup memory.
    ('spirv_msl.cpp',
     '\t\t\t\tfunc.add_parameter(type_id, next_id, true);\n'
     '\t\t\t\tauto &v = set<SPIRVariable>(next_id, type_id, StorageClassFunction, 0, arg_id);\n'
     '\t\t\t\tv.storage = StorageClassWorkgroup;\n'
     '\n'
     '\t\t\t\t// Ensure the existing variable has a valid name and the new variable has all the same meta info\n',
     '\t\t\t\tfunc.add_parameter(type_id, next_id, true);\n'
     '\t\t\t\tauto &v = set<SPIRVariable>(next_id, type_id, StorageClassFunction, 0, arg_id);\n'
     '\t\t\t\tif (mesh_output_thread_step(arg_id) == 0)\n'
     '\t\t\t\t\tv.storage = StorageClassWorkgroup;\n'
     '\n'
     '\t\t\t\t// Ensure the existing variable has a valid name and the new variable has all the same meta info\n'),
    ('spirv_msl.cpp',
     '\t\t{\n'
     '\t\tcase SPVFuncImplUnsafeArray:\n'
     '\t\t\tstatement("template<typename T, size_t Num>");\n'
     '\t\t\tstatement("struct spvUnsafeArray");\n',
     '\t\t{\n'
     '\t\tcase SPVFuncImplUnsafeArray:\n'
     '\t\t\tif (get_execution_model() == ExecutionModelMeshEXT)\n'
     '\t\t\t{\n'
     "\t\t\t\t// One thread's part of an output array: elements t, t + Step, t + 2 * Step, ... of Num.\n"
     '\t\t\t\tstatement("template<typename T, size_t Num, size_t Step>");\n'
     '\t\t\t\tstatement("struct spvMeshThreadArray");\n'
     '\t\t\t\tbegin_scope();\n'
     '\t\t\t\tstatement("T elements[(Num + Step - 1) / Step];");\n'
     '\t\t\t\tstatement("");\n'
     '\t\t\t\tstatement("thread T& operator [] (size_t pos) thread");\n'
     '\t\t\t\tbegin_scope();\n'
     '\t\t\t\tstatement("return elements[pos / Step];");\n'
     '\t\t\t\tend_scope();\n'
     '\t\t\t\tstatement("constexpr const thread T& operator [] (size_t pos) const thread");\n'
     '\t\t\t\tbegin_scope();\n'
     '\t\t\t\tstatement("return elements[pos / Step];");\n'
     '\t\t\t\tend_scope();\n'
     '\t\t\t\tend_scope_decl();\n'
     '\t\t\t\tstatement("");\n'
     '\t\t\t}\n'
     '\t\t\tstatement("template<typename T, size_t Num>");\n'
     '\t\t\tstatement("struct spvUnsafeArray");\n'),
    ('spirv_msl.cpp',
     '}\n'
     '\n'
     '// Metal leaves these undefined when a mesh shader omits them, while Vulkan defines them when they are not written.\n'
     'bool CompilerMSL::is_mesh_output_with_default(BuiltIn builtin, StorageClass storage) const\n',
     '}\n'
     '\n'
     "// A mesh shader's output array that is not a block is kept by each thread for itself, in a spvMeshThreadArray: thread t\n"
     '// of T holds elements t, t + T, t + 2T, ... (the vertices and primitives this thread hands to the mesh at the end of\n'
     '// the entry point), so the array takes no threadgroup memory. The shader must write and read an element from the thread\n'
     '// that owns it. Returns T for such a variable (or a function parameter standing for it), else 0.\n'
     'uint32_t CompilerMSL::mesh_output_thread_step(uint32_t id) const\n'
     '{\n'
     '\tif (!is_mesh_shader())\n'
     '\t\treturn 0;\n'
     '\tauto *var = maybe_get<SPIRVariable>(id);\n'
     '\twhile (var && var->basevariable && var->basevariable != var->self)\n'
     '\t\tvar = maybe_get<SPIRVariable>(var->basevariable);\n'
     "\t// the pointer type tells: a declaration as a function local changes the variable's own storage class for a while\n"
     '\tif (!var || get<SPIRType>(var->basetype).storage != StorageClassOutput)\n'
     '\t\treturn 0;\n'
     '\tauto &type = get_variable_data_type(*var);\n'
     '\tif (type.array.size() != 1 || type.basetype == SPIRType::Struct)\n'
     '\t\treturn 0;\n'
     '\tauto &execution = get_entry_point();\n'
     '\t// only a literal workgroup size: the step is written into the type\n'
     '\tif (execution.flags.get(ExecutionModeLocalSizeId) || !execution.flags.get(ExecutionModeLocalSize))\n'
     '\t\treturn 0;\n'
     '\treturn execution.workgroup_size.x * execution.workgroup_size.y * execution.workgroup_size.z;\n'
     '}\n'
     '\n'
     '// Metal leaves these undefined when a mesh shader omits them, while Vulkan defines them when they are not written.\n'
     'bool CompilerMSL::is_mesh_output_with_default(BuiltIn builtin, StorageClass storage) const\n'),
    ('spirv_msl.cpp',
     '\n'
     '\t\tif (is_mesh_shader())\n'
     '\t\t\taddr_space = "threadgroup";\n'
     '\t\tbreak;\n'
     '\n',
     '\n'
     '\t\tif (is_mesh_shader())\n'
     '\t\t\taddr_space = mesh_output_thread_step(id) ? "thread" : "threadgroup";\n'
     '\t\tbreak;\n'
     '\n'),
    ('spirv_msl.cpp',
     '\t\tstring sizes;\n'
     '\n'
     '\t\tfor (uint32_t i = 0; i < uint32_t(type.array.size()); i++)\n'
     '\t\t{\n'
     '\t\t\tres += "spvUnsafeArray<";\n'
     '\t\t\tsizes += ", ";\n'
     '\t\t\tsizes += to_array_size(type, i);\n'
     '\t\t\tsizes += ">";\n'
     '\t\t}\n',
     '\t\tstring sizes;\n'
     '\n'
     '\t\tuint32_t thread_step = mesh_output_thread_step(id);\n'
     '\t\tfor (uint32_t i = 0; i < uint32_t(type.array.size()); i++)\n'
     '\t\t{\n'
     '\t\t\tres += thread_step ? "spvMeshThreadArray<" : "spvUnsafeArray<";\n'
     '\t\t\tsizes += ", ";\n'
     '\t\t\tsizes += to_array_size(type, i);\n'
     '\t\t\tif (thread_step)\n'
     '\t\t\t\tsizes += join(", ", thread_step);\n'
     '\t\t\tsizes += ">";\n'
     '\t\t}\n'),
    ('spirv_msl.cpp',
     '\n'
     '\t\tif (is_mesh_shader())\n'
     '\t\t\treturn variable.storage == StorageClassOutput;\n'
     '\n'
     '\t\treturn variable.storage == StorageClassOutput && is_tesc_shader() && is_stage_output_variable_masked(variable);\n',
     '\n'
     '\t\tif (is_mesh_shader())\n'
     '\t\t\treturn variable.storage == StorageClassOutput && mesh_output_thread_step(variable.self) == 0;\n'
     '\n'
     '\t\treturn variable.storage == StorageClassOutput && is_tesc_shader() && is_stage_output_variable_masked(variable);\n'),
    ('spirv_msl.hpp',
     '\tbool is_tese_shader() const;\n'
     '\tbool is_mesh_shader() const;\n'
     '\tbool is_mesh_output_with_default(BuiltIn builtin, StorageClass storage) const;\n'
     '\tbool is_64bit_integer_interface(const SPIRType &type) const;\n',
     '\tbool is_tese_shader() const;\n'
     '\tbool is_mesh_shader() const;\n'
     '\tuint32_t mesh_output_thread_step(uint32_t id) const;\n'
     '\tbool is_mesh_output_with_default(BuiltIn builtin, StorageClass storage) const;\n'
     '\tbool is_64bit_integer_interface(const SPIRType &type) const;\n'),
    # Fragment inputs the stage before does not write read as zero. The game draws one mesh program (position and one
    # vector) with a fragment program that also reads three flat vectors: Metal's compiler ended on a null pointer
    # building that pipeline (MTLCompilerService in AGCLLVMUserMeshShaderSW::getBasePointer, reported to the game as
    # XPC_ERROR_CONNECTION_INTERRUPTED), and with a vertex function Metal refuses such a pipeline. MoltenVK tells the
    # fragment conversion which locations the stage before writes; an input at another location is left out of the
    # stage-in struct and becomes a zero local, which is what an unwritten varying reads on the console's GPU.
    ('spirv_msl.cpp',
     '\t}\n'
     '\n'
     '\t// Holds SetMeshOutputsEXT information. Threadgroup since first thread wins.\n'
     '\tif (processing_entry_point && is_mesh_shader())\n',
     '\t}\n'
     '\n'
     '\tstd::sort(unfed_frag_inputs.begin(), unfed_frag_inputs.end());\n'
     '\tfor (uint32_t var_id : unfed_frag_inputs)\n'
     '\t{\n'
     '\t\tauto &var = get<SPIRVariable>(var_id);\n'
     '\t\tadd_local_variable_name(var_id);\n'
     '\t\tstatement(CompilerGLSL::variable_decl(var), " = {};");\n'
     '\t\tvar.deferred_declaration = false;\n'
     '\t}\n'
     '\n'
     '\t// Holds SetMeshOutputsEXT information. Threadgroup since first thread wins.\n'
     '\tif (processing_entry_point && is_mesh_shader())\n'),
    ('spirv_msl.cpp',
     '\t\t}\n'
     '\n'
     '\t\t// Barycentric inputs must be emitted in stage-in, because they can have interpolation arguments.\n'
     '\t\tif (is_active && (bi_type == BuiltInBaryCoordKHR || bi_type == BuiltInBaryCoordNoPerspKHR))\n',
     '\t\t}\n'
     '\n'
     '\t\t// A fragment input at a location the stage before does not write, where the pipeline told which it writes: it is\n'
     "\t\t// left out of the stage-in struct and reads as zero. Metal's compiler otherwise fails the pipeline, or with a\n"
     '\t\t// mesh function before it ends on a null pointer (AGX, macOS 27).\n'
     '\t\tif (get_execution_model() == ExecutionModelFragment && storage == StorageClassInput && !patch && !is_builtin &&\n'
     '\t\t    !is_block && is_active && !hidden && !inputs_by_location.empty() &&\n'
     '\t\t    has_decoration(var_id, DecorationLocation) && type.basetype != SPIRType::Struct && type.array.empty() &&\n'
     '\t\t    type.columns == 1)\n'
     '\t\t{\n'
     '\t\t\tbool fed = false;\n'
     '\t\t\tfor (auto &input : inputs_by_location)\n'
     '\t\t\t\tfed = fed || input.first.location == location;\n'
     '\t\t\tif (!fed)\n'
     '\t\t\t{\n'
     '\t\t\t\thidden = true;\n'
     '\t\t\t\tunfed_frag_inputs.push_back(var_id);\n'
     '\t\t\t}\n'
     '\t\t}\n'
     '\n'
     '\t\t// Barycentric inputs must be emitted in stage-in, because they can have interpolation arguments.\n'
     '\t\tif (is_active && (bi_type == BuiltInBaryCoordKHR || bi_type == BuiltInBaryCoordNoPerspKHR))\n'),
    ('spirv_msl.hpp',
     '\n'
     '\tSmallVector<uint32_t> disabled_frag_outputs;\n'
     '\n'
     '\tstd::unordered_set<SetBindingPair, InternalHasher> inline_uniform_blocks;\n',
     '\n'
     '\tSmallVector<uint32_t> disabled_frag_outputs;\n'
     '\tSmallVector<uint32_t> unfed_frag_inputs;\n'
     '\n'
     '\tstd::unordered_set<SetBindingPair, InternalHasher> inline_uniform_blocks;\n'),
    # Mesh shader positions in the game's depth range. The game's clip space has z from -w to w, which the back end asks
    # for per pipeline (VK_EXT_depth_clip_control, negativeOneToOne); Metal's has z from 0 to w, so MoltenVK has
    # SPIRV-Cross write z = (z + w) / 2 at the end of the last stage before the rasterizer. SPIRV-Cross did that for
    # vertex, tessellation and geometry functions only: a mesh function's vertices kept the game's z, so Metal stored
    # 1 - 2 near / w for them where every other draw stores 1 - near / w, the depth of a point half as far away, and
    # clipped them within twice the near distance. Grass, foliage, particles and the ground mist (all mesh draws) then
    # won the depth test against anything up to twice as far: a character standing in grass or mist was covered by what
    # lies behind and around him, and passes that rebuild a position from the depth lit and fogged foliage as if it
    # were at half its distance. The mesh stage now makes the same adjustment where it hands each vertex to the mesh.
    ('spirv_msl.cpp',
     '\t\t\tstatement("spvV.", to_member_name(type_vert, index), " = ", value, ";");\n',
     '\t\t\tstatement("spvV.", to_member_name(type_vert, index), " = ", value, ";");\n'
     '\t\t\tif (options.vertex.fixup_clipspace && builtin == BuiltInPosition)\n'
     '\t\t\t{\n'
     '\t\t\t\tstatement("spvV.", to_member_name(type_vert, index), ".z = (spvV.", to_member_name(type_vert, index),\n'
     '\t\t\t\t          ".z + spvV.", to_member_name(type_vert, index), ".w) * 0.5;       // Adjust clip-space for Metal");\n'
     '\t\t\t}\n'),
]

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
    apply(src, PATCHES)


def apply(root, patches):
    for name, text, new in patches:
        f = root / name
        body = f.read_text()
        if body.count(text) != 1:
            sys.exit(f'build_moltenvk: {name}: the text to patch occurs {body.count(text)} times')
        f.write_text(body.replace(text, new))


def run(src, *cmd):
    print('build_moltenvk:', ' '.join(cmd), flush=True)
    if subprocess.run(cmd, cwd=src).returncode:
        sys.exit(f'build_moltenvk: {" ".join(cmd)} failed')


def build(src):
    # SPIRV-Cross is patched between its checkout (fetchDependencies --none fetches only) and its build (the patched
    # fetchDependencies leaves a repository at its revision as it is).
    sc = src / 'External' / 'SPIRV-Cross'
    if (sc / '.git').is_dir():
        git(sc, 'checkout', '-q', '--force', SPIRV_CROSS)   # drops an earlier build's patches
    run(src, './fetchDependencies', '--none')
    apply(sc, SPIRV_CROSS_PATCHES)
    run(src, './fetchDependencies', '--macos')
    run(src, 'make', 'macos', *MAKE_ARGS)
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
    (out / 'VERSION.txt').write_text(f'MoltenVK {MVK_MAIN} + pull request 2842 {MVK_PR}; SPIRV-Cross {SPIRV_CROSS}; '
                                     f'{len(PATCHES)} local patches, {len(SPIRV_CROSS_PATCHES)} to SPIRV-Cross; built with '
                                     f'{" ".join(MAKE_ARGS)} (private Metal API calls)\n')


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
