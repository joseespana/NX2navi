"""Exercise the runtime's library wait with a deterministic clock and worker queue, without a GPU."""
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'runtime/vk_pipeline.c').read_text()
start = source.index('static void lib_wait(')
end = source.index('\n}\n', start) + 3
harness = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
typedef int VkPipeState;
typedef int GKey;
typedef int Layout;
typedef struct { int building, job; } Lib;
static struct { int lock; } P;
static struct { int timeouts, lib_workers, idle, stolen; } U;
static Lib libs[2];
static double clock_ms;
static int builds;
static double now_ms(void) { return clock_ms; }
static int lib_queue(const VkPipeState *s, const GKey *k, Layout *l, Lib **b) {
    (void)s; (void)k; (void)l;
    b[0] = &libs[0]; b[1] = &libs[1]; return 1;
}
static void lib_steal(Lib *b) { builds++; clock_ms += 1000; b->building = b->job = 0; }
static int done_cv;
static void plat_cond_wait_ms(int *cv, int *lock, uint32_t ms) {
    (void)cv; (void)lock; assert(ms > 0 && ms <= 501); clock_ms += ms;
}
'''
harness += source[start:end]
harness += r'''
int main(void) {
    /* With no idle worker, a bounded request must still return without compiling. */
    for (int idle = 0; idle <= 1; idle++) {
        U.idle = idle; U.lib_workers = 0; U.timeouts = 0;
        libs[0] = libs[1] = (Lib){1, 1}; clock_ms = 0; builds = 0;
        lib_wait(0, 0, 0, 6, 0);
        assert(builds == 0 && U.timeouts == 1 && clock_ms <= 7);
        assert(libs[0].job && libs[1].job);
    }
    /* An unlimited one-shot request retains the synchronous fallback. */
    clock_ms = 0; builds = 0; U.idle = 0;
    lib_wait(0, 0, 0, 10000, 1);
    assert(builds == 2 && !libs[0].building && !libs[1].building);
    /* Ready libraries return without sleeping. */
    clock_ms = 0; lib_wait(0, 0, 0, 6, 0); assert(clock_ms == 0);
    return 0;
}
'''
work = ROOT / 'build/test/pipeline_wait'
work.mkdir(parents=True, exist_ok=True)
c = work / 'check.c'
c.write_text(harness)
exe = work / ('check.exe' if os.name == 'nt' else 'check')
subprocess.run([os.environ.get('CC') or shutil.which('clang') or 'cc', '-Wall', '-Werror',
                str(c), '-o', str(exe)], check=True)
subprocess.run([str(exe)], check=True)
print('pipeline_wait_check: passed')
