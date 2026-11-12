"""Check the device format cache with concurrent callers and a mock physical device; no GPU required."""
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'runtime/vk_device.c').read_text()
start = source.index('// Entries include unsupported formats')
end = source.index('static PlatMutex g_lock', start)
cache = source[start:end]
# Use the actual teardown reset, including the lock, rather than a separate model of it.
start = source.index('    plat_mutex_lock(&format_lock);', source.index('static void destroy_all('))
end = source.index('    memset(&g_vk', start)
reset = source[start:end]
code = r'''
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
typedef int VkFormat;
typedef struct { uint32_t linearTilingFeatures, optimalTilingFeatures, bufferFeatures; } VkFormatProperties;
typedef pthread_mutex_t PlatMutex;
#define PLAT_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER
#define plat_mutex_lock pthread_mutex_lock
#define plat_mutex_unlock pthread_mutex_unlock
static struct { int phys; } g_vk = {1};
static unsigned queries;
static void vkGetPhysicalDeviceFormatProperties(int phys, VkFormat format, VkFormatProperties *p) {
    queries++;
    *p = format ? (VkFormatProperties){format, format + phys, format * 2} : (VkFormatProperties){0};
}
'''
code += cache
code += '\nstatic void reset_cache(void) {\n' + reset + '}\n'
code += r'''
static void *reader(void *unused) {
    (void)unused;
    for (int repeat = 0; repeat < 1000; repeat++) {
        for (int f = 0; f < 64; f++) {
            VkFormatProperties p;
            vk_format_properties(f, &p);
            assert(p.linearTilingFeatures == (unsigned)f);
            assert(p.optimalTilingFeatures == (unsigned)(f ? f + g_vk.phys : 0));
            assert(p.bufferFeatures == (unsigned)(f * 2));
        }
    }
    return 0;
}
int main(void) {
    pthread_t threads[8];
    for (int i = 0; i < 8; i++) assert(!pthread_create(&threads[i], 0, reader, 0));
    for (int i = 0; i < 8; i++) assert(!pthread_join(threads[i], 0));
    assert(queries == 64); /* Zero support is cached, and concurrent misses query once. */
    for (int f = 64; f < 300; f++) {
        VkFormatProperties p; vk_format_properties(f, &p);
        assert(p.optimalTilingFeatures == (unsigned)(f + 1));
    }
    assert(format_cache_count == 256 && queries == 300);
    VkFormatProperties p;
    vk_format_properties(299, &p); assert(queries == 301 && p.optimalTilingFeatures == 300);
    reset_cache(); g_vk.phys = 2;
    vk_format_properties(1, &p); assert(queries == 302 && p.optimalTilingFeatures == 3);
    return 0;
}
'''
work = ROOT / 'build/test/format_properties'
work.mkdir(parents=True, exist_ok=True)
c = work / 'check.c'
c.write_text(code)
exe = work / ('check.exe' if os.name == 'nt' else 'check')
subprocess.run([os.environ.get('CC') or shutil.which('clang') or 'cc', '-O1', '-Wall', '-Werror', '-pthread',
                str(c), '-o', str(exe)], check=True)
subprocess.run([str(exe)], check=True)
print('format_properties_check: passed (concurrency, unsupported formats, capacity, device reset)')
