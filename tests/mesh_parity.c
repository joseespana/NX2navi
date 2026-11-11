// The same geometry drawn by a vertex shader pipeline and by a mesh shader pipeline, into offscreen float targets,
// compared pixel by pixel: four colour targets (smooth, flat and noperspective inputs, facing / depth / 1/w) and the
// depth buffer. The mesh and task modules have the shape tools/xlate/MeshSpirv.cs gives the game's (tests/mesh_parity.py
// builds them with that file). Built and run by tests/mesh_parity.py against the game's own MoltenVK; macOS only.
//
//   mesh_parity <spv dir> <first|last> [filter]
//
// One line per case: "pass" or "FAIL" with the number of differing pixels per target and the first of them. A case
// marked as a known difference prints "known" and does not fail the run. "last": a flat input of a mesh pipeline is
// expected from its triangle's last vertex (what the game expects, and what tools/xlate/MeshSpirv.cs arranges on
// macOS; a line's still from its first); "first": from the first (Vulkan's default).
// Exits 1 when a case failed.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

enum { W = 64, H = 64, LAYERS = 2, NCOL = 4, STRIDE = 10, MAXVERT = 16384, NCNT = 32768, PIX = W * H * LAYERS };
#define COL_BYTES ((size_t)PIX * 16)
#define READ_BYTES (COL_BYTES * NCOL + (size_t)PIX * 4)
#define OK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { printf("FAIL: %s = %d (line %d)\n", #call, (int)r_, __LINE__); exit(2); } } while (0)

// The shader sets tests/mesh_parity.py builds (its VARIANTS): vertices per primitive, primitives per mesh workgroup.
typedef struct { const char *name; int ipp, maxp, task, viewports; VkShaderModule vs, fs, mfs, ms, ts; } Variant;
static Variant variants[] = {
    {"tri", 3, 32}, {"line", 2, 32}, {"point", 1, 32}, {"point_nosize", 1, 32}, {"count", 1, 1}, {"tcount", 1, 1, 1},
    {"ttri", 3, 32, 1}, {"big", 3, 53}, {"tbig", 3, 53, 1}, {"bigx", 3, 53}, {"clip", 3, 32}, {"prim", 3, 32},
    {"layer", 3, 32, 0, 2}, {"grid", 3, 98}, {"tgrid", 3, 98, 1},
};
enum { V_TRI, V_LINE, V_POINT, V_POINT_NOSIZE, V_COUNT, V_TCOUNT, V_TTRI, V_BIG, V_TBIG, V_BIGX, V_CLIP, V_PRIM, V_LAYER,
       V_GRID, V_TGRID, NVARIANT };

typedef struct {
    int variant;
    VkCullModeFlags cull;
    int ccw, minus_w, clamp;
    int wire;           // polygon mode: lines
    VkViewport vp;
    int dtest, dwrite;
    VkCompareOp dcmp;
    int bias;
    float bias_const, bias_clamp, bias_slope;
    float clear_depth;
    int prepass;        // a depth-only draw (LESS, written) of the same geometry before the draw
    int nprims;
    int indirect;       // the mesh draw reads its count from a buffer
    int per_task;       // mesh workgroups each task workgroup launches
    int no_image;       // more primitives than pixels: only the workgroup counters are checked
    int known;          // a difference that is expected and reported, not a failure
    int may_be_empty;   // nothing drawn by either pipeline is a pass
} State;

static VkInstance inst;
static VkPhysicalDevice phys;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkCommandBuffer cb;
static VkRenderPass pass;
static VkFramebuffer fb;
static VkImage col[NCOL], depth;
static VkDescriptorSetLayout dsl;
static VkPipelineLayout layout;
static VkDescriptorSet dset;
static VkBuffer geo_buf, cnt_buf, read_buf, ind_buf;
static uint32_t *geo_hdr, *cnt, *ind;
static float *geo;      // vertex records: STRIDE vectors each
static uint8_t *readback;
static int provoking_last, fails, cases;
static const char *filter = "";
static PFN_vkCmdDrawMeshTasksEXT draw_mesh;                   // extension commands: not exported by the loader
static PFN_vkCmdDrawMeshTasksIndirectEXT draw_mesh_indirect;

// ---- Vulkan objects
static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    printf("FAIL: no memory type\n");
    exit(2);
}

static VkBuffer buffer(VkDeviceSize size, VkBufferUsageFlags usage, void **map) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
    VkBuffer b;
    OK(vkCreateBuffer(dev, &bi, NULL, &b));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, b, &mr);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
        .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VkDeviceMemory m;
    OK(vkAllocateMemory(dev, &ai, NULL, &m));
    OK(vkBindBufferMemory(dev, b, m, 0));
    OK(vkMapMemory(dev, m, 0, VK_WHOLE_SIZE, 0, map));
    return b;
}

static VkImage image(VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkImageView *view) {
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
        .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = LAYERS, .samples = VK_SAMPLE_COUNT_1_BIT,
        .usage = usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    VkImage im;
    OK(vkCreateImage(dev, &ii, NULL, &im));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev, im, &mr);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
        .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VkDeviceMemory m;
    OK(vkAllocateMemory(dev, &ai, NULL, &m));
    OK(vkBindImageMemory(dev, im, m, 0));
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im,
        .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY, .format = fmt, .subresourceRange = {aspect, 0, 1, 0, LAYERS}};
    OK(vkCreateImageView(dev, &vi, NULL, view));
    return im;
}

static VkShaderModule module(const char *dir, const char *name, const char *kind) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.%s.spv", dir, name, kind);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL: %s not found\n", path); exit(2); }
    static uint32_t code[1 << 18];
    size_t n = fread(code, 1, sizeof code, f);
    fclose(f);
    VkShaderModuleCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = n, .pCode = code};
    VkShaderModule m;
    OK(vkCreateShaderModule(dev, &si, NULL, &m));
    return m;
}

static void init(const char *dir) {
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
    const char *iext[] = {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
        .flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR, .enabledExtensionCount = 1, .ppEnabledExtensionNames = iext};
    OK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 1;
    vkEnumeratePhysicalDevices(inst, &n, &phys);
    if (!n) { printf("FAIL: no Vulkan device\n"); exit(2); }
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(phys, &pp);

    // every feature the driver has, but the two mesh features that need features of other extensions
    VkPhysicalDeviceDepthClipControlFeaturesEXT dcc = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, .pNext = &dcc};
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &mesh};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(phys, &f2);
    VkPhysicalDeviceMeshShaderPropertiesEXT mprop = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &mprop};
    vkGetPhysicalDeviceProperties2(phys, &p2);
    printf("device: %s; mesh %d task %d; depth clip control %d; depth clamp %d; clip/cull distance %d/%d; multiViewport %d; "
           "output layer/viewport %d/%d; largePoints %d\n", pp.deviceName, mesh.meshShader, mesh.taskShader,
           dcc.depthClipControl, f2.features.depthClamp, f2.features.shaderClipDistance, f2.features.shaderCullDistance,
           f2.features.multiViewport, f12.shaderOutputLayer, f12.shaderOutputViewportIndex, f2.features.largePoints);
    printf("limits: mesh shared memory %u, output memory %u, payload and output %u, vertices %u, primitives %u, "
           "mesh workgroups %u, task workgroups %u, task payload %u\n", mprop.maxMeshSharedMemorySize,
           mprop.maxMeshOutputMemorySize, mprop.maxMeshPayloadAndOutputMemorySize, mprop.maxMeshOutputVertices,
           mprop.maxMeshOutputPrimitives, mprop.maxMeshWorkGroupTotalCount, mprop.maxTaskWorkGroupTotalCount,
           mprop.maxTaskPayloadSize);
    if (!mesh.meshShader || !mesh.taskShader || !dcc.depthClipControl) { printf("FAIL: no mesh or task shaders or depth clip control\n"); exit(2); }
    mesh.multiviewMeshShader = mesh.primitiveFragmentShadingRateMeshShader = VK_FALSE;

    uint32_t nq = 0, qf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
    VkQueueFamilyProperties qp[16];
    if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp);
    for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qf = i; break; }
    float prio = 1;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio};
    const char *dext[] = {VK_EXT_MESH_SHADER_EXTENSION_NAME, VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME, "VK_KHR_portability_subset"};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2, .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qi, .enabledExtensionCount = 3, .ppEnabledExtensionNames = dext};
    OK(vkCreateDevice(phys, &di, NULL, &dev));
    vkGetDeviceQueue(dev, qf, 0, &queue);
    draw_mesh = (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT");
    draw_mesh_indirect = (PFN_vkCmdDrawMeshTasksIndirectEXT)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectEXT");
    VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
    OK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
    VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .commandBufferCount = 1};
    OK(vkAllocateCommandBuffers(dev, &cai, &cb));

    // targets
    VkImageView views[NCOL + 1];
    VkAttachmentDescription att[NCOL + 1];
    VkAttachmentReference refs[NCOL + 1];
    for (int i = 0; i <= NCOL; i++) {
        int d = i == NCOL;
        VkFormat fmt = d ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
        VkImage im = image(fmt, d ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                           d ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, &views[i]);
        if (d) depth = im; else col[i] = im;
        att[i] = (VkAttachmentDescription){.format = fmt, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
        refs[i] = (VkAttachmentReference){(uint32_t)i, d ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    }
    VkSubpassDescription sub = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = NCOL,
        .pColorAttachments = refs, .pDepthStencilAttachment = &refs[NCOL]};
    VkRenderPassCreateInfo rpi = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = NCOL + 1,
        .pAttachments = att, .subpassCount = 1, .pSubpasses = &sub};
    OK(vkCreateRenderPass(dev, &rpi, NULL, &pass));
    VkFramebufferCreateInfo fbi = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = pass,
        .attachmentCount = NCOL + 1, .pAttachments = views, .width = W, .height = H, .layers = LAYERS};
    OK(vkCreateFramebuffer(dev, &fbi, NULL, &fb));

    // buffers: the geometry (a header of 16 words, then the vertex records), the workgroup counters, the readback
    void *p;
    geo_buf = buffer(64 + (VkDeviceSize)MAXVERT * STRIDE * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &p);
    geo_hdr = p, geo = (float *)((uint8_t *)p + 64);
    cnt_buf = buffer(NCNT * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &p);
    cnt = p;
    read_buf = buffer(READ_BYTES, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &p);
    readback = p;
    ind_buf = buffer(256, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, &p);
    ind = p;

    VkShaderStageFlags stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_TASK_BIT_EXT;
    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stages}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stages}};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = b};
    OK(vkCreateDescriptorSetLayout(dev, &li, NULL, &dsl));
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl};
    OK(vkCreatePipelineLayout(dev, &pli, NULL, &layout));
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VkDescriptorPool dp;
    OK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
    VkDescriptorSetAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp,
        .descriptorSetCount = 1, .pSetLayouts = &dsl};
    OK(vkAllocateDescriptorSets(dev, &dai, &dset));
    VkDescriptorBufferInfo bi[2] = {{geo_buf, 0, VK_WHOLE_SIZE}, {cnt_buf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[2];
    for (int i = 0; i < 2; i++)
        wr[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset, .dstBinding = (uint32_t)i,
            .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(dev, 2, wr, 0, NULL);

    for (int i = 0; i < NVARIANT; i++) {
        Variant *v = &variants[i];
        v->vs = module(dir, v->name, "vert"), v->fs = module(dir, v->name, "frag"), v->mfs = module(dir, v->name, "mfrag");
        v->ms = module(dir, v->name, "mesh");
        if (v->task) v->ts = module(dir, v->name, "task");
        if (!v->viewports) v->viewports = 1;
    }
}

// ---- pipelines: one per shader set, stage kind and the state that is not dynamic
typedef struct { int variant, mesh, minus_w, clamp, wire, nocolour; VkPipeline p; } Pipe;
static Pipe pipes[512];
static int npipes;

static VkPipeline pipeline(const State *s, int mesh, int nocolour) {
    for (int i = 0; i < npipes; i++)
        if (pipes[i].variant == s->variant && pipes[i].mesh == mesh && pipes[i].minus_w == s->minus_w &&
            pipes[i].clamp == s->clamp && pipes[i].wire == s->wire && pipes[i].nocolour == nocolour) return pipes[i].p;
    const Variant *v = &variants[s->variant];
    VkPipelineShaderStageCreateInfo st[3];
    uint32_t nst = 0;
#define STAGE(bit, mod) st[nst++] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = bit, .module = mod, .pName = "main"}
    if (mesh) {
        if (v->task) STAGE(VK_SHADER_STAGE_TASK_BIT_EXT, v->ts);
        STAGE(VK_SHADER_STAGE_MESH_BIT_EXT, v->ms);
        STAGE(VK_SHADER_STAGE_FRAGMENT_BIT, v->mfs);
    } else {
        STAGE(VK_SHADER_STAGE_VERTEX_BIT, v->vs);
        STAGE(VK_SHADER_STAGE_FRAGMENT_BIT, v->fs);
    }
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = v->ipp == 3 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : v->ipp == 2 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_POINT_LIST};
    VkPipelineViewportDepthClipControlCreateInfoEXT cc = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
        .negativeOneToOne = s->minus_w};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .pNext = &cc,
        .viewportCount = (uint32_t)v->viewports, .scissorCount = (uint32_t)v->viewports};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = s->clamp, .polygonMode = s->wire ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL, .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState ba[NCOL];
    for (int i = 0; i < NCOL; i++) ba[i] = (VkPipelineColorBlendAttachmentState){.colorWriteMask = nocolour ? 0 : 0xf};
    VkPipelineColorBlendStateCreateInfo bs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = NCOL, .pAttachments = ba};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = sizeof dyn / sizeof *dyn, .pDynamicStates = dyn};
    VkGraphicsPipelineCreateInfo gp = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = nst, .pStages = st,
        .pVertexInputState = mesh ? NULL : &vi, .pInputAssemblyState = mesh ? NULL : &ia, .pViewportState = &vp,
        .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &bs,
        .pDynamicState = &dy, .layout = layout, .renderPass = pass};
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, NULL, &p);
    if (r != VK_SUCCESS) p = VK_NULL_HANDLE;
    pipes[npipes++] = (Pipe){s->variant, mesh, s->minus_w, s->clamp, s->wire, nocolour, p};
    return p;
}

// ---- one render: the pre-pass (if any) and the draw, each by a vertex ('V') or a mesh ('M') pipeline
typedef struct { uint8_t px[READ_BYTES]; uint32_t counts[NCNT]; int no_pipeline; } Shot;

static void draw(const State *s, int mesh, int nocolour, Shot *out) {
    const Variant *v = &variants[s->variant];
    VkPipeline p = pipeline(s, mesh, nocolour);
    if (!p) { out->no_pipeline = 1; return; }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    if (!mesh) { vkCmdDraw(cb, (uint32_t)(s->nprims * v->ipp), 1, 0, 0); return; }
    uint32_t groups = (uint32_t)((s->nprims + v->maxp - 1) / v->maxp);
    if (v->task) groups = (groups + (uint32_t)s->per_task - 1) / (uint32_t)s->per_task;
    if (s->indirect) {
        ind[0] = ind[1] = ind[2] = 0xdeadbeef;   // the command starts at byte 16
        ind[4] = groups, ind[5] = 1, ind[6] = 1;
        draw_mesh_indirect(cb, ind_buf, 16, 1, sizeof(VkDrawMeshTasksIndirectCommandEXT));
    } else draw_mesh(cb, groups, 1, 1);
}

static void render(const State *s, const char *kinds, Shot *out) {
    const Variant *v = &variants[s->variant];
    memset(out, 0, sizeof *out);
    memset(cnt, 0, NCNT * 4);
    geo_hdr[0] = (uint32_t)s->nprims, geo_hdr[1] = (uint32_t)(provoking_last && v->ipp == 3), geo_hdr[2] = (uint32_t)s->per_task, geo_hdr[3] = NCNT / 2;
    OK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    OK(vkBeginCommandBuffer(cb, &bi));
    VkClearValue clear[NCOL + 1];
    for (int i = 0; i < NCOL; i++) clear[i].color = (VkClearColorValue){.float32 = {-1, -1, -1, -1}};
    clear[NCOL].depthStencil = (VkClearDepthStencilValue){s->clear_depth, 0};
    VkRenderPassBeginInfo rb = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = pass, .framebuffer = fb,
        .renderArea = {{0, 0}, {W, H}}, .clearValueCount = NCOL + 1, .pClearValues = clear};
    vkCmdBeginRenderPass(cb, &rb, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &dset, 0, NULL);
    // a second viewport (the per-primitive case) is the first moved right and down
    VkViewport vps[2] = {s->vp, s->vp};
    vps[1].x += 6, vps[1].y += 3;
    VkRect2D sc[2] = {{{0, 0}, {W, H}}, {{0, 0}, {W, H}}};
    vkCmdSetViewport(cb, 0, (uint32_t)v->viewports, vps);
    vkCmdSetScissor(cb, 0, (uint32_t)v->viewports, sc);
    vkCmdSetCullMode(cb, s->cull);
    vkCmdSetFrontFace(cb, s->ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE);
    vkCmdSetDepthBiasEnable(cb, s->bias);
    vkCmdSetDepthBias(cb, s->bias_const, s->bias_clamp, s->bias_slope);
    if (s->prepass) {
        vkCmdSetDepthTestEnable(cb, 1), vkCmdSetDepthWriteEnable(cb, 1), vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_LESS);
        draw(s, kinds[0] == 'M', 1, out);
    }
    vkCmdSetDepthTestEnable(cb, s->dtest), vkCmdSetDepthWriteEnable(cb, s->dwrite), vkCmdSetDepthCompareOp(cb, s->dcmp);
    draw(s, kinds[s->prepass ? 1 : 0] == 'M', 0, out);
    vkCmdEndRenderPass(cb);
    for (int i = 0; i <= NCOL; i++) {
        VkBufferImageCopy r = {.bufferOffset = COL_BYTES * (size_t)i,
            .imageSubresource = {i == NCOL ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, LAYERS}, .imageExtent = {W, H, 1}};
        vkCmdCopyImageToBuffer(cb, i == NCOL ? depth : col[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, read_buf, 1, &r);
    }
    OK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    OK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    OK(vkQueueWaitIdle(queue));
    memcpy(out->px, readback, READ_BYTES);
    memcpy(out->counts, cnt, NCNT * 4);
}

// ---- geometry
static uint32_t rng_state;
static float rnd(void) {   // 0..1
    rng_state ^= rng_state << 13, rng_state ^= rng_state >> 17, rng_state ^= rng_state << 5;
    return (float)(rng_state >> 8) / 16777216.0f;
}
static float range(float a, float b) { return a + (b - a) * rnd(); }
static float *vert(int i) { return geo + (size_t)i * STRIDE * 4; }

// Primitives scattered over the view and a little beyond it, each vertex with its own w (perspective), depth between
// zlo and zhi of the depth range (below 0 or above 1: beyond the near or far plane), random generic vectors, a point
// size of 1 to 4, a clip distance that cuts some primitives and a cull distance that drops a few.
static void scatter(uint32_t seed, int nprims, int ipp, int minus_w, float zlo, float zhi, float size) {
    rng_state = seed * 2654435761u + 12345u;
    for (int p = 0; p < nprims; p++) {
        float cx = range(-0.95f, 0.95f), cy = range(-0.95f, 0.95f), cz = range(zlo, zhi);
        int culled = rnd() < 0.1f;
        for (int j = 0; j < ipp; j++) {
            float *v = vert(p * ipp + j);
            float w = range(0.5f, 2.0f), z = cz + range(-0.1f, 0.1f) * (zhi - zlo);
            if (minus_w) z = z * 2 - 1;
            v[0] = (cx + range(-size, size)) * w, v[1] = (cy + range(-size, size)) * w, v[2] = z * w, v[3] = w;
            for (int k = 4; k < 36; k++) v[k] = rnd();
            v[36] = (float)(1 + (int)(rnd() * 4)), v[37] = range(-0.3f, 1.0f), v[38] = culled ? -1.0f : 1.0f, v[39] = 0;
        }
    }
}

// Groups of 64 vertices, each an 8 x 8 grid over a patch of the view, bent in depth and w, for the shader sets whose
// triangles share their vertices.
static void patches(uint32_t seed, int groups, int minus_w) {
    rng_state = seed * 2654435761u + 777u;
    for (int g = 0; g < groups; g++) {
        float cx = range(-0.7f, 0.7f), cy = range(-0.7f, 0.7f), cz = range(0.2f, 0.8f), size = range(0.3f, 0.6f);
        float turn = range(0.0f, 6.28f), tilt = range(-0.15f, 0.15f), mirror = (g & 1) ? -1.0f : 1.0f;   // every other one faces away
        for (int i = 0; i < 64; i++) {
            float *v = vert(g * 64 + i);
            float u = mirror * ((float)(i % 8) / 7 - 0.5f + range(-0.03f, 0.03f)), t = (float)(i / 8) / 7 - 0.5f + range(-0.03f, 0.03f);
            float x = cx + size * (u * cosf(turn) - t * sinf(turn)), y = cy + size * (u * sinf(turn) + t * cosf(turn));
            float w = 1.0f + 0.6f * u + 0.3f * t, z = cz + tilt * (u + t) + range(-0.01f, 0.01f);
            if (minus_w) z = z * 2 - 1;
            v[0] = x * w, v[1] = y * w, v[2] = z * w, v[3] = w;
            for (int k = 4; k < 36; k++) v[k] = rnd();
            v[36] = 1, v[37] = 1, v[38] = 1, v[39] = 0;
        }
    }
}

// Point i at the centre of pixel i (row by row), one pixel large, with a colour of its own.
static void grid_points(int n) {
    rng_state = 99;
    for (int i = 0; i < n; i++) {
        float *v = vert(i);
        v[0] = ((float)(i % W) + 0.5f) / W * 2 - 1, v[1] = ((float)(i / W % H) + 0.5f) / H * 2 - 1, v[2] = 0.5f, v[3] = 1;
        for (int k = 4; k < 36; k++) v[k] = rnd();
        v[36] = 1, v[37] = 1, v[38] = 1, v[39] = 0;
    }
}

// ---- comparison
static const char *target_name[NCOL + 1] = {"smooth", "flat", "noperspective", "facing/z/w", "depth"};

static int covered(const Shot *s) {
    int n = 0;
    const float *c = (const float *)s->px;
    for (int i = 0; i < PIX; i++) n += c[i * 4] != -1.0f;
    return n;
}

// Workgroups that did not run exactly once: mesh workgroups [0, mesh), task workgroups [0, tasks).
static int count_errors(const Shot *s, int mesh, int tasks, char *why, size_t nwhy) {
    int bad = 0;
    for (int half = 0; half < 2; half++)
        for (int i = 0; i < NCNT / 2; i++) {
            uint32_t want = i < (half ? tasks : mesh), got = s->counts[half * (NCNT / 2) + i];
            if (got != want && !bad++) snprintf(why, nwhy, "%s workgroup %d ran %u times, not %u", half ? "task" : "mesh", i, got, want);
        }
    return bad;
}

static State base(int variant, int nprims) {
    State s = {.variant = variant, .cull = VK_CULL_MODE_NONE, .minus_w = 1, .vp = {0, H, W, -H, 0, 1}, .dtest = 1, .dwrite = 1,
               .dcmp = VK_COMPARE_OP_LESS, .clear_depth = 1.0f, .nprims = nprims, .per_task = 1};
    return s;
}

// Draws the case with vertex pipelines only and as kinds says, and compares. Returns 1 when they agree.
static int check(const char *name, const State *s, const char *kinds) {
    static Shot ref, got;
    if (!strstr(name, filter)) return 1;
    const Variant *v = &variants[s->variant];
    cases++;
    render(s, "VV", &ref);
    render(s, kinds, &got);
    char why[512] = "";
    int bad = 0, n_ref = covered(&ref);
    if (ref.no_pipeline || got.no_pipeline) {
        bad = 1;
        snprintf(why, sizeof why, "the %s pipeline was not created", ref.no_pipeline ? "vertex" : "mesh");
    }
    int mesh_draws = (kinds[0] == 'M') + (s->prepass && kinds[1] == 'M');
    if (!bad && mesh_draws) {
        int groups = (s->nprims + v->maxp - 1) / v->maxp, tasks = 0;
        if (v->task) tasks = (groups + s->per_task - 1) / s->per_task, groups = tasks * s->per_task;
        // two mesh draws count twice
        if (mesh_draws == 2)
            for (int i = 0; i < NCNT; i++) got.counts[i] = got.counts[i] & 1 ? got.counts[i] + NCNT : got.counts[i] / 2;
        bad = count_errors(&got, groups, tasks, why, sizeof why);
    }
    if (!bad && !s->no_image) {
        int per[NCOL + 1] = {0}, first = -1, first_t = 0;
        for (int t = 0; t <= NCOL; t++) {
            size_t sz = t == NCOL ? 4 : 16;
            for (int i = 0; i < PIX; i++)
                if (memcmp(ref.px + COL_BYTES * (size_t)t + sz * (size_t)i, got.px + COL_BYTES * (size_t)t + sz * (size_t)i, sz)) {
                    if (first < 0) first = i, first_t = t;
                    per[t]++, bad++;
                }
        }
        if (bad) {
            const float *a = (const float *)(ref.px + COL_BYTES * (size_t)first_t), *b = (const float *)(got.px + COL_BYTES * (size_t)first_t);
            int o = snprintf(why, sizeof why, "pixels differing:");
            for (int t = 0; t <= NCOL; t++) o += snprintf(why + o, sizeof why - (size_t)o, " %s %d", target_name[t], per[t]);
            o += snprintf(why + o, sizeof why - (size_t)o, "; first: %s (%d,%d) layer %d: vertex ", target_name[first_t],
                          first % W, first / W % H, first / (W * H));
            if (first_t == NCOL) snprintf(why + o, sizeof why - (size_t)o, "%.9g, mesh %.9g", a[first], b[first]);
            else snprintf(why + o, sizeof why - (size_t)o, "(%.6g %.6g %.6g %.6g), mesh (%.6g %.6g %.6g %.6g)", a[first * 4],
                          a[first * 4 + 1], a[first * 4 + 2], a[first * 4 + 3], b[first * 4], b[first * 4 + 1], b[first * 4 + 2], b[first * 4 + 3]);
        } else if (!n_ref && !s->may_be_empty) bad = 1, snprintf(why, sizeof why, "the vertex pipeline drew nothing: the case compares nothing");
    }
    if (!bad) printf("%-44s pass   (%d pixels drawn)\n", name, s->no_image ? covered(&got) : n_ref);
    else if (s->known) printf("%-44s known  %s\n", name, why);
    else printf("%-44s FAIL   %s\n", name, why), fails++;
    fflush(stdout);
    return !bad;
}

// Which vertex of a primitive a flat input comes from: 1, 2 or 3 (0: none of them).
static void provoking_info(void) {
    static Shot shot;
    static const float tri[3][2] = {{-0.9f, -0.9f}, {0.9f, -0.9f}, {0.0f, 0.9f}}, line[2][2] = {{-0.9f, 0.01f}, {0.9f, 0.01f}};
    int was = provoking_last;
    provoking_last = 0;
    for (int kind = 0; kind < 2; kind++) {
        int ipp = kind ? 2 : 3, at[2];
        memset(geo, 0, 3 * STRIDE * 16);
        for (int j = 0; j < ipp; j++) {
            float *v = vert(j);
            v[0] = kind ? line[j][0] : tri[j][0], v[1] = kind ? line[j][1] : tri[j][1], v[2] = 0.5f, v[3] = 1;
            v[8] = (float)(j + 1);   // the flat vector's x
            v[36] = 1, v[37] = 1, v[38] = 1;
        }
        State s = base(kind ? V_LINE : V_TRI, 1);
        for (int m = 0; m < 2; m++) {
            render(&s, m ? "M" : "V", &shot);
            const float *flat = (const float *)(shot.px + COL_BYTES);
            at[m] = 0;
            for (int i = 0; i < W * H; i++) if (flat[i * 4] > 0) at[m] = (int)flat[i * 4];
        }
        printf("info: a %s's flat input comes from vertex %d of %d in a vertex pipeline, vertex %d in a mesh pipeline\n",
               kind ? "line" : "triangle", at[0], ipp, at[1]);
    }
    provoking_last = was;
}

int main(int argc, char **argv) {
    if (argc < 3) { printf("usage: mesh_parity <spv dir> <first|last> [filter]\n"); return 2; }
    provoking_last = !strcmp(argv[2], "last");
    if (argc > 3) filter = argv[3];
    init(argv[1]);
    printf("a mesh pipeline's flat input is expected from its triangle's %s vertex\n", provoking_last ? "last" : "first");
    provoking_info();
    char name[128];
    static const VkViewport views[3] = {{0, 0, W, H, 0, 1}, {0, H, W, -H, 0, 1}, {8, 56, 40, -48, 0.25f, 0.75f}};
    static const char *view_name[3] = {"plain", "flipped", "flipped_inset_range"};

    // a. position: x, y, depth and w, in both clip spaces, through each viewport
    for (int mw = 0; mw < 2; mw++)
        for (int vi = 0; vi < 3; vi++) {
            State s = base(V_TRI, 40);
            s.minus_w = mw, s.vp = views[vi];
            scatter(1 + (uint32_t)vi, s.nprims, 3, mw, 0.05f, 0.95f, 0.35f);
            snprintf(name, sizeof name, "a.position.%s.%s", mw ? "z_minus_w_to_w" : "z_0_to_w", view_name[vi]);
            check(name, &s, "M");
        }
    // geometry through the near and far planes, clipped
    for (int mw = 0; mw < 2; mw++) {
        State s = base(V_TRI, 40);
        s.minus_w = mw;
        scatter(7, s.nprims, 3, mw, -0.4f, 1.4f, 0.35f);
        snprintf(name, sizeof name, "a.clip_near_far.%s", mw ? "z_minus_w_to_w" : "z_0_to_w");
        check(name, &s, "M");
    }

    // b. winding and culling, with the facing in the fourth target
    static const VkCullModeFlags culls[3] = {VK_CULL_MODE_NONE, VK_CULL_MODE_FRONT_BIT, VK_CULL_MODE_BACK_BIT};
    static const char *cull_name[3] = {"none", "front", "back"};
    for (int vi = 0; vi < 2; vi++)
        for (int c = 0; c < 3; c++)
            for (int ccw = 0; ccw < 2; ccw++) {
                State s = base(V_TRI, 40);
                s.cull = culls[c], s.ccw = ccw, s.vp = views[vi];
                scatter(11, s.nprims, 3, 1, 0.05f, 0.95f, 0.35f);
                snprintf(name, sizeof name, "b.cull_%s.front_%s.%s", cull_name[c], ccw ? "ccw" : "cw", view_name[vi]);
                check(name, &s, "M");
            }

    // lines instead of filled triangles
    {
        State s = base(V_TRI, 40);
        s.wire = 1, s.cull = VK_CULL_MODE_BACK_BIT;
        scatter(12, s.nprims, 3, 1, 0.05f, 0.95f, 0.35f);
        check("b.polygon_mode_line.cull_back", &s, "M");
    }
    // triangles that share their vertices, as the game's meshes do (64 vertices, up to 126 triangles a workgroup)
    for (int c = 0; c < 3; c++) {
        State s = base(V_GRID, 98 * 5 - 30);
        s.cull = culls[c];
        patches(13, 5, 1);
        snprintf(name, sizeof name, "b.shared_vertices.cull_%s", cull_name[c]);
        check(name, &s, "M");
    }

    // c. interpolation: every case compares the smooth, flat and noperspective targets; these name the flat source
    {
        State s = base(V_TRI, 64);
        scatter(21, s.nprims, 3, 1, 0.05f, 0.95f, 0.3f);
        check("c.interpolation.triangles", &s, "M");
        s = base(V_LINE, 64);
        scatter(22, s.nprims, 2, 1, 0.05f, 0.95f, 0.5f);
        check("c.interpolation.lines", &s, "M");
    }

    // d. depth state
    static const char *cmp_name[8] = {"never", "less", "equal", "less_or_equal", "greater", "not_equal", "greater_or_equal", "always"};
    for (int op = 0; op < 8; op++) {
        State s = base(V_TRI, 40);
        s.dcmp = (VkCompareOp)op, s.clear_depth = 0.5f;
        scatter(31, s.nprims, 3, 1, 0.05f, 0.95f, 0.35f);
        snprintf(name, sizeof name, "d.depth_compare.%s", cmp_name[op]);
        // NEVER, and EQUAL against the cleared depth: nothing passes, and both must leave the targets cleared
        s.may_be_empty = op == VK_COMPARE_OP_NEVER || op == VK_COMPARE_OP_EQUAL;
        check(name, &s, "M");
    }
    // a depth pre-pass and a colour pass that tests EQUAL, each by either kind of pipeline
    for (int mw = 0; mw < 2; mw++)
        for (int k = 0; k < 3; k++) {
            static const char *kinds[3] = {"VM", "MV", "MM"};
            State s = base(V_TRI, 64);
            s.minus_w = mw, s.prepass = 1, s.dcmp = VK_COMPARE_OP_EQUAL, s.dwrite = 0;
            scatter(41, s.nprims, 3, mw, 0.05f, 0.95f, 0.3f);
            snprintf(name, sizeof name, "d.equal.prepass_%s.colour_%s.%s", kinds[k][0] == 'M' ? "mesh" : "vertex",
                     kinds[k][1] == 'M' ? "mesh" : "vertex", mw ? "z_minus_w_to_w" : "z_0_to_w");
            check(name, &s, kinds[k]);
        }
    {
        State s = base(V_GRID, 98 * 6);
        s.prepass = 1, s.dcmp = VK_COMPARE_OP_EQUAL, s.dwrite = 0, s.cull = VK_CULL_MODE_BACK_BIT;
        patches(42, 6, 1);
        check("d.equal.shared_vertices.mesh_mesh", &s, "MM");
        check("d.equal.shared_vertices.vertex_mesh", &s, "VM");
        s = base(V_TGRID, 98 * 6);
        s.prepass = 1, s.dcmp = VK_COMPARE_OP_EQUAL, s.dwrite = 0, s.per_task = 4;
        patches(43, 6, 1);
        check("d.equal.shared_vertices.task_mesh", &s, "MM");
    }
    {
        State s = base(V_TRI, 40);
        s.bias = 1, s.bias_const = 4000.0f, s.bias_slope = 2.5f, s.dcmp = VK_COMPARE_OP_LESS_OR_EQUAL;
        scatter(51, s.nprims, 3, 1, 0.05f, 0.95f, 0.35f);
        check("d.depth_bias", &s, "M");
        s.bias_const = -4000.0f, s.bias_slope = -1.0f, s.bias_clamp = -0.01f;
        check("d.depth_bias.negative_clamped", &s, "M");
        // a biased colour pass over an unbiased pre-pass
        s.prepass = 1;
        check("d.depth_bias.over_prepass", &s, "MM");
    }
    for (int mw = 0; mw < 2; mw++)
        for (int clamp = 0; clamp < 2; clamp++) {
            State s = base(V_TRI, 40);
            s.minus_w = mw, s.clamp = clamp;
            scatter(61, s.nprims, 3, mw, -0.5f, 1.5f, 0.35f);
            snprintf(name, sizeof name, "d.depth_clamp_%s.%s", clamp ? "on" : "off", mw ? "z_minus_w_to_w" : "z_0_to_w");
            check(name, &s, "M");
        }
    // no depth test: the later primitive wins, across workgroups too
    {
        State s = base(V_TRI, 96);
        s.dtest = 0, s.dwrite = 0;
        scatter(71, s.nprims, 3, 1, 0.05f, 0.95f, 0.4f);
        check("d.no_depth_test.primitive_order", &s, "M");
    }

    // e. primitives
    {
        State s = base(V_LINE, 64);
        scatter(81, s.nprims, 2, 1, 0.05f, 0.95f, 0.5f);
        check("e.lines", &s, "M");
        s = base(V_POINT, 64);
        scatter(82, s.nprims, 1, 1, 0.05f, 0.95f, 0.0f);
        check("e.points.size_written", &s, "M");
        // no point size from the shader: one pixel from a vertex pipeline; the game's point programs all write it
        s = base(V_POINT_NOSIZE, 64);
        s.known = 1;
        scatter(82, s.nprims, 1, 1, 0.05f, 0.95f, 0.0f);
        check("e.points.size_not_written", &s, "M");
    }

    // f. outputs the translator does not write today
    {
        State s = base(V_CLIP, 40);
        s.known = 1;
        scatter(91, s.nprims, 3, 1, 0.05f, 0.95f, 0.35f);
        check("f.clip_distance", &s, "M");
        s = base(V_PRIM, 40);
        scatter(92, s.nprims, 3, 1, 0.05f, 0.95f, 0.3f);
        check("f.per_primitive.id_and_colour", &s, "M");
        s = base(V_LAYER, 40);
        s.known = 1;
        scatter(92, s.nprims, 3, 1, 0.05f, 0.95f, 0.3f);
        check("f.per_primitive.layer_and_viewport", &s, "M");
    }

    // g. workgroup counts: one pixel per workgroup, every workgroup counted
    static const int counts[] = {1, 1023, 1024, 1025, 3000, 9000};
    for (int ind_ = 0; ind_ < 2; ind_++)
        for (int c = 0; c < 6; c++) {
            State s = base(V_COUNT, counts[c]);
            s.indirect = ind_, s.no_image = counts[c] > W * H;
            s.known = ind_ && counts[c] > 8192;   // tools/build_moltenvk.py MESH_BLOCKS: an indirect draw launches 8 blocks of 1024
            grid_points(s.nprims);
            snprintf(name, sizeof name, "g.mesh_only.%s.%d", ind_ ? "indirect" : "direct", counts[c]);
            check(name, &s, "M");
        }
    for (int ind_ = 0; ind_ < 2; ind_++) {
        static const int tasks[][2] = {{1, 1}, {1023, 1}, {1024, 1}, {1025, 1}, {3000, 1}, {3000, 8}, {4096, 32}, {9000, 3}};
        for (int c = 0; c < 8; c++) {
            State s = base(V_TCOUNT, tasks[c][0]);
            s.indirect = ind_, s.per_task = tasks[c][1], s.no_image = tasks[c][0] > W * H;
            grid_points(s.nprims);
            snprintf(name, sizeof name, "g.task_mesh.%s.%d_tasks_x_%d", ind_ ? "indirect" : "direct",
                     (tasks[c][0] + tasks[c][1] - 1) / tasks[c][1], tasks[c][1]);
            check(name, &s, "M");
        }
        State s = base(V_TTRI, 200);
        s.indirect = ind_, s.per_task = 3;
        scatter(101, s.nprims, 3, 1, 0.05f, 0.95f, 0.2f);
        snprintf(name, sizeof name, "g.task_mesh.%s.triangles", ind_ ? "indirect" : "direct");
        check(name, &s, "M");
    }

    // h. outputs and shared memory as large as the game's largest foliage shaders
    {
        State s = base(V_BIG, 150);
        scatter(111, s.nprims, 3, 1, 0.05f, 0.95f, 0.2f);
        check("h.large_outputs.mesh_only", &s, "M");
        s.prepass = 1, s.dcmp = VK_COMPARE_OP_EQUAL, s.dwrite = 0;
        check("h.large_outputs.prepass_and_equal", &s, "MM");
        s = base(V_TBIG, 150);
        s.per_task = 2;
        scatter(112, s.nprims, 3, 1, 0.05f, 0.95f, 0.2f);
        check("h.large_outputs.task_mesh", &s, "M");
        // a thread that writes another thread's vertices and primitives: outside what the per-thread outputs of
        // tools/build_moltenvk.py allow
        s = base(V_BIGX, 150);
        s.known = 1;
        scatter(111, s.nprims, 3, 1, 0.05f, 0.95f, 0.2f);
        check("h.large_outputs.written_by_another_thread", &s, "M");
    }

    printf("mesh_parity: %d cases, %s\n", cases, fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
