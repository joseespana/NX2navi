#version 450
// tests/mesh_parity.c: the vertex pipeline's side of a case. Vertex i of the draw is record i of the geometry buffer
// (STRIDE vectors: the position, eight generic vectors, then point size, clip distance, cull distance and a spare).
// tests/mesh_parity.py defines IPP (vertices per primitive) and what the variant has: BIG, PSIZE, CLIP, PRIM, LAYER,
// GRID (the draw's vertices are the corners of the triangles of 8 x 8 vertex grids, 64 records and 98 triangles each,
// as tests/mesh_parity/mesh.comp indexes them).
#extension GL_ARB_shader_viewport_layer_array : enable
#define STRIDE 10
layout(std430, set = 0, binding = 0) readonly buffer Geo { uint hdr[16]; vec4 data[]; } geo;
layout(location = 0) out vec4 g0;
layout(location = 1) flat out vec4 g1;
layout(location = 2) noperspective out vec4 g2;
#ifdef BIG
layout(location = 3) out vec4 g3;
layout(location = 4) out vec4 g4;
layout(location = 5) out vec4 g5;
layout(location = 6) out vec4 g6;
layout(location = 7) out vec4 g7;
#endif
#ifdef PRIM
layout(location = 3) flat out vec4 prim_colour;
#endif

#ifdef GRID
uint record(uint n) {
    uint p = n / 3u, j = n % 3u, cell = (p % 98u) / 2u, x = cell % 7u, y = cell / 7u;
    uint corner = (p & 1u) == 0u ? (j == 0u ? 0u : j == 1u ? 1u : 8u) : (j == 0u ? 1u : j == 1u ? 9u : 8u);
    return p / 98u * 64u + y * 8u + x + corner;
}
#else
uint record(uint n) { return n; }
#endif

void main() {
    uint n = uint(gl_VertexIndex), i = record(n), b = i * STRIDE;
    // hdr[1] bit 0: a flat value comes from its primitive's last vertex (what the game expects) instead of this one
    uint last = record(n - n % IPP + (IPP - 1));
    uint f = (geo.hdr[1] & 1u) != 0u ? last : i;
    gl_Position = geo.data[b];
    g0 = geo.data[b + 1];
    g1 = geo.data[f * STRIDE + 2];
    g2 = geo.data[b + 3];
#ifdef BIG
    g3 = geo.data[b + 4];
    g4 = geo.data[b + 5];
    g5 = geo.data[b + 6];
    g6 = geo.data[b + 7];
    g7 = geo.data[b + 8];
#endif
#ifdef PSIZE
    gl_PointSize = geo.data[b + 9].x;
#endif
#ifdef CLIP
    gl_ClipDistance[0] = geo.data[b + 9].y;
#endif
#ifdef PRIM
    // what a mesh shader writes once per primitive: the same on each of the primitive's vertices
#ifdef LAYER
    uint p = n / IPP;
    gl_Layer = int(p & 1u);
    gl_ViewportIndex = int((p >> 1) & 1u);
#endif
    prim_colour = geo.data[last * STRIDE + 4];
#endif
}
