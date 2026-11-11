#version 450
// tests/mesh_parity.c: the fragment shader of both pipelines of a case. Four float targets take what the rasterizer
// hands over: the smooth, flat and noperspective vectors and the fragment's facing, depth and 1/w.
// -DBIG: eight generic vectors as the game's largest foliage shaders have. -DPRIM: per-primitive values (-DPRIM_MESH:
// from a mesh shader, where the per-primitive colour is a perprimitiveEXT input), with -DLAYER the layer and the
// viewport index too.
#ifdef PRIM_MESH
#extension GL_EXT_mesh_shader : require
#endif
layout(location = 0) in vec4 g0;
layout(location = 1) flat in vec4 g1;
layout(location = 2) noperspective in vec4 g2;
#ifdef BIG
layout(location = 3) in vec4 g3;
layout(location = 4) in vec4 g4;
layout(location = 5) in vec4 g5;
layout(location = 6) in vec4 g6;
layout(location = 7) in vec4 g7;
#endif
#ifdef PRIM
#ifdef PRIM_MESH
layout(location = 3) perprimitiveEXT flat in vec4 prim_colour;
#else
layout(location = 3) flat in vec4 prim_colour;
#endif
#endif
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;

void main() {
    o0 = g0;
    o1 = g1;
    o2 = g2;
    o3 = vec4(gl_FrontFacing ? 1.0 : 0.25, gl_FragCoord.z, gl_FragCoord.w, 1.0);
#ifdef BIG
    o0 = g0 + g3;
    o2 = g2 + vec4(g4.xyz, g5.x);
    o3.w = g6.x + g7.x;
#endif
#ifdef PRIM
    o0 = prim_colour;
#ifdef LAYER
    o2 = vec4(float(gl_PrimitiveID), float(gl_Layer), float(gl_ViewportIndex), 1.0);
#else
    o2 = vec4(float(gl_PrimitiveID), 0.0, 0.0, 1.0);
#endif
#endif
}
