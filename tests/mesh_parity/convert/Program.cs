// tests/mesh_parity.py: turns compute modules into task and mesh modules with the translator's own MeshSpirv.Convert,
// so the test draws with the module shape the game's shaders have. One argument: a JSON list of jobs
//   {"in": compute.spv, "out": mesh.spv, "stage": 6 | 7, "threads", "maxv", "maxp", "topo", "payload", "task": bool,
//    "lines": [output line addresses], "attr", "map", "pay", "shared": the layout the GLSL was compiled for,
//    "flat_first": bool (the translator's job option of that name: triangles handed over last vertex first)}
// The layout comes from AmpereLowering.MeshInfo.Layout on a program header built from the lines; a job whose own
// numbers differ from it fails.
using System;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;

static class Program
{
    static int Main(string[] args)
    {
        foreach (JsonObject j in JsonNode.Parse(File.ReadAllText(args[0])).AsArray())
        {
            var m = new AmpereLowering.MeshInfo
            {
                Stage = (int)j["stage"], Threads = (int)j["threads"], MaxVertices = (int)j["maxv"], MaxPrimitives = (int)j["maxp"],
                Topology = (int)j["topo"], PayloadBytes = (int)j["payload"], HasTask = (bool)j["task"],
                FlatFirst = j["flat_first"] is JsonNode f && (bool)f,
            };
            int[] lines = j["lines"].AsArray().Select(x => (int)x).ToArray();
            var sph = new byte[0x50];
            foreach (int a in lines)
            {
                if (a < 0x80) sph[0x35] |= (byte)(1 << ((a - 0x60) / 4));
                else { int g = (a - 0x80) / 16, c = (a - 0x80) % 16 / 4; sph[0x36 + g / 2] |= (byte)(1 << (4 * (g & 1) + c)); }
            }
            m.Layout(sph);
            if (m.Stage == 7 && !m.Lines.SequenceEqual(lines)) throw new Exception($"{(string)j["out"]}: lines out of order");
            if (m.Attr != (int)j["attr"] || m.Map != (int)j["map"] || m.Pay != (int)j["pay"] || m.SharedBytes != (int)j["shared"])
                throw new Exception($"{(string)j["out"]}: layout attr {m.Attr} map {m.Map} pay {m.Pay} shared {m.SharedBytes}");
            File.WriteAllBytes((string)j["out"], MeshSpirv.Convert(File.ReadAllBytes((string)j["in"]), m));
        }
        return 0;
    }
}
