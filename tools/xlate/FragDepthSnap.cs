// macOS only (Program.cs Translate): a fragment shader that compares gl_FragCoord.z with a value it reads from a texture
// gets the comparison's z operand replaced by that texture value when the two are within one float step.
//
// Why: a game draws a surface depth-only, then draws the same surface again with a fragment shader that reads the stored
// depth and discards where gl_FragCoord.z is above it (Ocarina's ground mist: its own surface must pass, anything in
// front of the stored depth too). On the console the depth a pass stores is exactly the z its fragment shader reads, so
// a fragment never rejects its own surface. Apple GPUs before the M6 store a depth that is, at a few percent of a
// triangle's pixels, one float step below the z the fragment shader is given for the same triangle and pixel (never
// above), also with invariant positions. The second draw then discards its own surface at those pixels: isolated dark
// pixels all over the mist.
//
// The rewrite, for `cmp(z, d)` or `cmp(d, z)` with any float comparison:
//   lo = bitcast<float>(bitcast<uint>(z) - 1) - 2^-20 * fwidth(z)      one step below z, plus a sliver of the slope
//   z' = (d < z && d >= lo) ? d : z
// and the comparison uses z'. A stored depth of the fragment's own surface compares equal, as on the console; a surface
// outside that window (2^-24 at z near 1) compares as before. z = 0 gives a NaN bound and leaves z alone.
//
// Recognised: z is a load of gl_FragCoord's third component (through an access chain or an extract of the whole vector),
// d is a texture sample, fetch or gather result or a component of one; either may pass through a function variable that
// is stored exactly once. Comparisons of scaled values (z * k against d * k) are left as they are.
//
// Opcode and enum numbers are the SPIR-V 1.6 specification's.
using System;
using System.Collections.Generic;

static class FragDepthSnap
{
    const int OpTypeInt = 21, OpTypeFloat = 22, OpConstant = 43, OpFunction = 54, OpVariable = 59, OpLoad = 61, OpStore = 62,
        OpAccessChain = 65, OpInBoundsAccessChain = 66, OpDecorate = 71, OpCompositeExtract = 81,
        OpImageSampleImplicitLod = 87, OpImageDrefGather = 97, OpBitcast = 124, OpISub = 130, OpFSub = 131, OpFMul = 133,
        OpLogicalAnd = 167, OpSelect = 169, OpFOrdEqual = 180, OpFOrdLessThan = 184, OpFOrdGreaterThanEqual = 190,
        OpFUnordGreaterThanEqual = 191, OpFwidth = 209;
    const int DecBuiltIn = 11, BiFragCoord = 15, ScFunction = 7;

    // Returns the module with every such comparison rewritten (the same array when there is none); count: how many.
    public static byte[] Apply(byte[] spirv, out int count)
    {
        count = 0;
        if (spirv.Length < 20 || BitConverter.ToUInt32(spirv, 0) != 0x07230203) return spirv;
        var w = new uint[spirv.Length / 4];
        Buffer.BlockCopy(spirv, 0, w, 0, w.Length * 4);
        var at = new List<int>();   // word offset of each instruction
        for (int p = 5; p < w.Length;)
        {
            int n = (int)(w[p] >> 16);
            if (n == 0 || p + n > w.Length) return spirv;
            at.Add(p);
            p += n;
        }
        int Op(int p) => (int)(w[p] & 0xffff);
        int Len(int p) => (int)(w[p] >> 16);

        uint fragCoord = 0, tFloat = 0, tUint = 0;
        int firstFunction = -1;
        var def = new Dictionary<uint, int>();          // result id -> offset, for the value instructions read below
        var intConst = new Dictionary<uint, uint>();    // 32-bit constants by id
        var stores = new Dictionary<uint, (int n, uint value)>();   // pointer -> number of stores, the last value
        var functionVars = new HashSet<uint>();
        foreach (int p in at)
        {
            int op = Op(p);
            if (op == OpDecorate && Len(p) == 4 && w[p + 2] == DecBuiltIn && w[p + 3] == BiFragCoord) fragCoord = w[p + 1];
            else if (op == OpTypeFloat && w[p + 2] == 32) tFloat = w[p + 1];
            else if (op == OpTypeInt && w[p + 2] == 32 && w[p + 3] == 0) tUint = w[p + 1];
            else if (op == OpConstant && Len(p) == 4) intConst[w[p + 2]] = w[p + 3];
            else if (op == OpFunction && firstFunction < 0) firstFunction = p;
            else if (op == OpVariable) { if (w[p + 3] == ScFunction) functionVars.Add(w[p + 2]); }
            else if (op == OpStore) stores[w[p + 1]] = (stores.TryGetValue(w[p + 1], out var s) ? s.n + 1 : 1, w[p + 2]);
            else if (op is OpLoad or OpAccessChain or OpInBoundsAccessChain or OpCompositeExtract
                     || (op >= OpImageSampleImplicitLod && op <= OpImageDrefGather))
                def[w[p + 2]] = p;
        }
        if (fragCoord == 0 || tFloat == 0 || firstFunction < 0) return spirv;

        // The value an operand stands for: a load of a function variable stored once is the value stored there.
        uint Resolve(uint id)
        {
            for (int hop = 0; hop < 8; hop++)
            {
                if (!def.TryGetValue(id, out int p) || Op(p) != OpLoad || !functionVars.Contains(w[p + 3])) break;
                if (!stores.TryGetValue(w[p + 3], out var s) || s.n != 1) break;
                id = s.value;
            }
            return id;
        }
        bool IsZ(uint id)
        {
            if (!def.TryGetValue(id, out int p) || w[p + 1] != tFloat) return false;
            if (Op(p) == OpLoad)   // a load through an access chain to component 2
                return def.TryGetValue(w[p + 3], out int c) && Op(c) is OpAccessChain or OpInBoundsAccessChain && Len(c) == 5
                       && w[c + 3] == fragCoord && intConst.TryGetValue(w[c + 4], out uint k) && k == 2;
            if (Op(p) == OpCompositeExtract && Len(p) == 5 && w[p + 4] == 2)   // component 2 of the whole vector
                return def.TryGetValue(w[p + 3], out int l) && Op(l) == OpLoad && w[l + 3] == fragCoord;
            return false;
        }
        bool IsSample(int p) => Op(p) >= OpImageSampleImplicitLod && Op(p) <= OpImageDrefGather;
        bool IsTexel(uint id)
        {
            if (!def.TryGetValue(id, out int p) || w[p + 1] != tFloat) return false;
            if (IsSample(p)) return true;   // a depth-comparison sample returns a scalar
            return Op(p) == OpCompositeExtract && def.TryGetValue(w[p + 3], out int s) && IsSample(s);
        }

        uint bound = w[3];
        var types = new List<uint>();   // new types and constants, placed before the first function
        void Emit(List<uint> o, int op, params uint[] operands)
        {
            o.Add((uint)((operands.Length + 1) << 16 | op));
            o.AddRange(operands);
        }
        uint one = 0, sliver = 0;
        var before = new Dictionary<int, List<uint>>();   // comparison offset -> instructions placed before it
        foreach (int p in at)
        {
            int op = Op(p);
            if (op < OpFOrdEqual || op > OpFUnordGreaterThanEqual || Len(p) != 5) continue;
            int zAt = IsZ(Resolve(w[p + 3])) && IsTexel(Resolve(w[p + 4])) ? 3
                      : IsZ(Resolve(w[p + 4])) && IsTexel(Resolve(w[p + 3])) ? 4 : 0;
            if (zAt == 0) continue;
            uint z = w[p + zAt], d = w[p + 7 - zAt], tBool = w[p + 1];
            if (one == 0)
            {
                if (tUint == 0) Emit(types, OpTypeInt, tUint = bound++, 32, 0);
                Emit(types, OpConstant, tUint, one = bound++, 1);
                Emit(types, OpConstant, tFloat, sliver = bound++, 0x35800000);   // 2^-20
            }
            var o = new List<uint>();
            uint fw = bound++, zi = bound++, zm = bound++, zp = bound++, sl = bound++, lo = bound++, c1 = bound++,
                 c2 = bound++, c = bound++, snapped = bound++;
            Emit(o, OpFwidth, tFloat, fw, z);
            Emit(o, OpBitcast, tUint, zi, z);
            Emit(o, OpISub, tUint, zm, zi, one);
            Emit(o, OpBitcast, tFloat, zp, zm);
            Emit(o, OpFMul, tFloat, sl, fw, sliver);
            Emit(o, OpFSub, tFloat, lo, zp, sl);
            Emit(o, OpFOrdLessThan, tBool, c1, d, z);
            Emit(o, OpFOrdGreaterThanEqual, tBool, c2, d, lo);
            Emit(o, OpLogicalAnd, tBool, c, c1, c2);
            Emit(o, OpSelect, tFloat, snapped, c, d, z);
            w[p + zAt] = snapped;
            before[p] = o;
            count++;
        }
        if (count == 0) return spirv;

        var res = new List<uint>(w.Length + types.Count + count * 48);
        for (int i = 0; i < 5; i++) res.Add(w[i]);
        res[3] = bound;
        foreach (int p in at)
        {
            if (p == firstFunction) res.AddRange(types);
            if (before.TryGetValue(p, out var o)) res.AddRange(o);
            for (int i = 0; i < Len(p); i++) res.Add(w[p + i]);
        }
        var bytes = new byte[res.Count * 4];
        Buffer.BlockCopy(res.ToArray(), 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
