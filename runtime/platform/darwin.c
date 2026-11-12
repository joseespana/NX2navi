// macOS implementation of platform.h: pthreads, the Mach clock, wait-on-address locks and condition variables, POSIX
// files, Mach and sysctl for the load counters, posix_spawn for child processes. For x86-64 and ARM64; macOS 14.4 or
// later (os_sync_wait_on_address).
#define _DARWIN_C_SOURCE
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <crt_externs.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <libproc.h>
#include <libunwind.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/mach_vm.h>
#include <netdb.h>
#include <netinet/in.h>
#include <os/os_sync_wait_on_address.h>
#include <poll.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <sched.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/event.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "platform.h"

extern char **environ;

// A function written in assembly: Mach-O names carry a leading underscore and have no .type or .size.
#define ASM_FN(name) ".globl _" name "\n.p2align 4\n_" name ":\n"

// The interrupted thread's instruction and stack pointers in a signal handler's context.
#if defined(__x86_64__)
#define UC_PC(u) ((u)->uc_mcontext->__ss.__rip)
#define UC_SP(u) ((u)->uc_mcontext->__ss.__rsp)
#else
#define UC_PC(u) ((u)->uc_mcontext->__ss.__pc)
#define UC_SP(u) ((u)->uc_mcontext->__ss.__sp)
#endif

static int pipe_cloexec(int p[2]) {
    if (pipe(p)) return -1;
    fcntl(p[0], F_SETFD, FD_CLOEXEC);
    fcntl(p[1], F_SETFD, FD_CLOEXEC);
    return 0;
}

static uint32_t sysctl_u32(const char *name) {
    uint32_t v = 0;
    size_t n = sizeof v;
    return sysctlbyname(name, &v, &n, NULL, 0) ? 0 : v;
}

// ---- threads

struct PlatThread {
    pthread_t t;
    PlatThreadFn fn;
    void *arg;
    volatile int32_t done;   // fn has returned (plat_thread_join_ns)
    char name[32];
    int home;                // QOS_HOME_* from the name (plat_thread_qos on), else QOS_HOME_NONE
    pthread_override_t boost;   // another thread's raise (plat_thread_set_priority, QoS on); NULL none
};

// ---- quality of service
// Apple silicon has performance and efficiency cores, and the scheduler picks a core by a thread's quality-of-service
// class: user-interactive and user-initiated threads are kept on the performance cores while those have room,
// utility threads go to whichever is free, background threads run on the efficiency cores only (and their file I/O
// is throttled). A thread with no class runs as the default class. Precedence importance (set_importance below) does
// not take part in that choice, and macOS drops a thread's class for good once a precedence policy is set on it
// (pthread_set_qos_class_self_np fails from then on), so with plat_thread_qos on the priority calls set classes and
// never a precedence policy. A thread's home class comes from its name at creation.
enum { QOS_HOME_NONE, QOS_HOME_CRIT, QOS_HOME_BG };
static int qos_on;
static __thread PlatThread *qos_self;   // the calling thread's handle when plat_thread_new made it

void plat_thread_qos(int on) { qos_on = on; }

static int name_is(const char *name, const char *const *list) {
    for (; *list; list++)
        if (!strncmp(name, *list, strlen(*list))) return 1;
    return 0;
}

// Prefixes of thread names (the plat_thread_new callers). Critical: what a frame waits for (the game's own threads,
// the replay, recorder, completion and present threads, audio, input). Background: work a frame does not wait for
// in play, or waits for only after raising the thread itself (the shader and pipeline workers, gfx_shader.c and
// vk_pipeline.c). Any other name (the ASTC pool, whose caller decodes alongside it; movies; the log) keeps the default.
static int qos_home(const char *name) {
    static const char *const crit[] = {"main", "guest", "vk replay", "vk recorder", "vk present", "vk completion",
                                       "vsync", "fg present", "audren", "pads", "motion", "window", NULL};
    static const char *const bg[] = {"xlate", "shload", "shader ", "use preload", "pipe ", "big compute", "lib ",
                                     "prewarm", "mesh fallback", "mem census", "flight dump", "vk headroom",
                                     "debug names", NULL};
    return !name || !*name ? QOS_HOME_NONE : name_is(name, crit) ? QOS_HOME_CRIT : name_is(name, bg) ? QOS_HOME_BG
                                                                                                 : QOS_HOME_NONE;
}

static qos_class_t qos_of_home(int home) {
    return home == QOS_HOME_CRIT ? QOS_CLASS_USER_INTERACTIVE : home == QOS_HOME_BG ? QOS_CLASS_UTILITY
                                                                                     : QOS_CLASS_DEFAULT;
}

// A priority level as a class and relative priority. Above normal: interactive. Normal: the thread's home class when
// that is interactive, else the default class, which is what normal priority was (a background worker raised to
// normal while it holds a lock a frame waits on). Below normal and lowest: utility. Idle: background.
static qos_class_t qos_of_prio(int prio, int home, int *rel) {
    *rel = 0;
    if (prio >= PLAT_PRIO_ABOVE_NORMAL) return QOS_CLASS_USER_INTERACTIVE;
    if (prio == PLAT_PRIO_NORMAL) return home == QOS_HOME_CRIT ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_DEFAULT;
    if (prio <= PLAT_PRIO_IDLE) return QOS_CLASS_BACKGROUND;
    if (prio == PLAT_PRIO_LOWEST) *rel = -4;
    return QOS_CLASS_UTILITY;
}

// A signal stack for the calling thread, so the fault and crash handlers still run when its own stack is exhausted.
// Never freed: threads that end are few, and a stack freed while a handler might run on it would be worse.
static void alt_stack(void) {
    stack_t cur;
    if (!sigaltstack(NULL, &cur) && !(cur.ss_flags & SS_DISABLE)) return;
    size_t size = 128 << 10;
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) return;
    stack_t ss = {.ss_sp = p, .ss_size = size};
    sigaltstack(&ss, NULL);
}

static void *thread_main(void *p) {
    PlatThread *t = p;
    qos_self = t;
    alt_stack();
    if (t->name[0]) pthread_setname_np(t->name);
    t->fn(t->arg);
    __atomic_store_n(&t->done, 1, __ATOMIC_RELEASE);
    return NULL;
}

PlatThread *plat_thread_new(PlatThreadFn fn, void *arg, const char *name) {
    return plat_thread_new_ex(fn, arg, name, 0);
}

PlatThread *plat_thread_new_ex(PlatThreadFn fn, void *arg, const char *name, size_t stack) {
    PlatThread *t = calloc(1, sizeof *t);
    if (!t) return NULL;
    t->fn = fn;
    t->arg = arg;
    if (name) snprintf(t->name, sizeof t->name, "%s", name);
    pthread_attr_t a;
    pthread_attr_init(&a);
    if (qos_on) {
        t->home = qos_home(name);
        pthread_attr_set_qos_class_np(&a, qos_of_home(t->home), 0);
    }
    // The stack is mapped lazily, as Windows reserves it: only the pages a thread touches cost memory.
    pthread_attr_setstacksize(&a, stack ? stack : PLAT_STACK_DEFAULT);
    int e = pthread_create(&t->t, &a, thread_main, t);
    pthread_attr_destroy(&a);
    if (e) { free(t); return NULL; }
    return t;
}

// macOS numbers the logical processors of one physical core next to each other, so the first of each core is every
// (logical / physical)th. Apple Silicon has one logical processor per core.
int plat_cpu_cores(uint64_t *all, uint64_t *first) {
    uint32_t lg = plat_cpu_count(), ph = sysctl_u32("hw.physicalcpu");
    if (lg > 64) lg = 64;
    if (!ph || ph > lg) ph = lg;
    uint32_t per = lg / ph;
    *all = lg == 64 ? ~0ull : (1ull << lg) - 1;
    *first = 0;
    for (uint32_t i = 0; i < lg; i += per) *first |= 1ull << i;
    return (int)ph;
}

// ---- hardware profile
// The GPU's core count is a property of its registry entry (the AGXAccelerator service), not in sysctl.
static uint32_t gpu_core_count(void) {
    uint32_t n = 0;
    io_iterator_t it = 0;
    if (IOServiceGetMatchingServices(0, IOServiceMatching("AGXAccelerator"), &it) != KERN_SUCCESS) return 0;
    for (io_object_t o; !n && (o = IOIteratorNext(it)); IOObjectRelease(o)) {
        CFTypeRef v = IORegistryEntryCreateCFProperty(o, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (!v) continue;
        int32_t k = 0;
        if (CFGetTypeID(v) == CFNumberGetTypeID() && CFNumberGetValue(v, kCFNumberSInt32Type, &k) && k > 0) n = (uint32_t)k;
        CFRelease(v);
    }
    IOObjectRelease(it);
    return n;
}

static PlatHw hw;
static PlatOnce hw_once = PLAT_ONCE_INIT;

static void hw_read(void *arg) {
    (void)arg;
    size_t n = sizeof hw.chip;
    if (sysctlbyname("machdep.cpu.brand_string", hw.chip, &n, NULL, 0)) snprintf(hw.chip, sizeof hw.chip, "unknown");
    // "Apple M1", "Apple M1 Pro", "Apple M3 Max", "Apple M2 Ultra"
    const char *m = strstr(hw.chip, "Apple M");
    hw.tier = -1;
    if (m && m[7] >= '1' && m[7] <= '9') {
        hw.gen = (int)strtol(m + 7, NULL, 10);
        hw.tier = strstr(m, " Ultra") ? 3 : strstr(m, " Max") ? 2 : strstr(m, " Pro") ? 1 : 0;
    }
    // Apple GPU families (Metal feature set tables): M1 Apple7, M2 Apple8, M3 and M4 Apple9, M5 Apple10, M6 Apple11.
    static const int family[] = {0, 7, 8, 9, 9, 10, 11};
    hw.metal_family = hw.gen > 0 && hw.gen < (int)(sizeof family / sizeof *family) ? family[hw.gen] : 0;
    // hw.perflevel0 is the fastest kind of core; a chip with one kind has no perflevel1.
    hw.pcores = sysctl_u32("hw.perflevel0.physicalcpu");
    hw.ecores = sysctl_u32("hw.nperflevels") > 1 ? sysctl_u32("hw.perflevel1.physicalcpu") : 0;
    if (!hw.pcores) hw.pcores = sysctl_u32("hw.physicalcpu");
    hw.gpu_cores = gpu_core_count();
    uint64_t mem = 0;
    n = sizeof mem;
    if (!sysctlbyname("hw.memsize", &mem, &n, NULL, 0)) hw.ram_mb = mem >> 20;
}

const PlatHw *plat_hw(void) {
    plat_once(&hw_once, hw_read, NULL);
    return &hw;
}

int plat_hw_text(char *out, size_t cap) {
    const PlatHw *h = plat_hw();
    static const char *const tiers[] = {"base", "Pro", "Max", "Ultra"};
    int k = snprintf(out, cap, "%s: %s%d %s, GPU family Apple%d, %u performance + %u efficiency cores, %u GPU cores, "
                     "%llu MiB of memory", h->chip, h->gen ? "M" : "generation ", h->gen,
                     h->tier >= 0 ? tiers[h->tier] : "tier unknown", h->metal_family, h->pcores, h->ecores,
                     h->gpu_cores, (unsigned long long)h->ram_mb);
    return k < 0 ? 0 : (size_t)k < cap ? k : (int)cap - 1;
}

// One L3 cache at most, shared by every core.
uint64_t plat_cpu_cache_die(uint32_t *l3_kb) {
    *l3_kb = 0;
    return 0;
}

// macOS has no processor affinity: the scheduler alone places threads, so these report failure and change nothing.
int plat_process_affinity(uint64_t mask) { (void)mask; return 0; }
int plat_cpu_reserve(char *which, size_t size) {
    if (which && size) which[0] = 0;
    return 0;
}
int plat_cpu_reserve_sweep(void) { return 0; }
uint64_t plat_thread_affinity(PlatThread *t, uint64_t mask) { (void)t, (void)mask; return 0; }

// tid is the thread's Mach port name (plat_thread_id).
uint64_t plat_thread_cpu_ns(uint32_t tid) {
    thread_basic_info_data_t bi;
    mach_msg_type_number_t n = THREAD_BASIC_INFO_COUNT;
    if (thread_info((thread_act_t)tid, THREAD_BASIC_INFO, (thread_info_t)&bi, &n) != KERN_SUCCESS) return 0;
    return ((uint64_t)bi.user_time.seconds + (uint64_t)bi.system_time.seconds) * 1000000000u +
           ((uint64_t)bi.user_time.microseconds + (uint64_t)bi.system_time.microseconds) * 1000u;
}

// ---- thread interruption: a signal handshake in place of Windows' SuspendThread / Get/SetThreadContext.
// The caller moves the handle from IDLE to REQ and signals the thread. The thread's handler stores where it was and
// moves REQ to STOPPED, then spins until the caller moves it on to GO (with the redirect in to), changes its own
// return context, and goes back to IDLE. A caller that sees no STOPPED within INTR_WAIT_NS (the thread is blocked in
// the kernel and runs no handler) takes REQ back to IDLE; a handler that then finds IDLE returns at once.
enum { INTR_IDLE, INTR_REQ, INTR_STOPPED, INTR_GO };
#define INTR_WAIT_NS 200000u

struct PlatIntr {
    pthread_t pt;
    volatile int32_t state;
    uint64_t rip, rsp, to;
};

static __thread PlatIntr *intr_me;
static int intr_sig;

static void intr_handler(int sig, siginfo_t *si, void *uc) {
    (void)sig, (void)si;
    PlatIntr *t = intr_me;
    ucontext_t *u = uc;
    if (!t) return;
    t->rip = (uint64_t)UC_PC(u);
    t->rsp = (uint64_t)UC_SP(u);
    int32_t req = INTR_REQ;
    if (!__atomic_compare_exchange_n(&t->state, &req, INTR_STOPPED, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return;
    while (__atomic_load_n(&t->state, __ATOMIC_ACQUIRE) != INTR_GO) plat_pause();
    if (t->to) UC_PC(u) = t->to;
    __atomic_store_n(&t->state, INTR_IDLE, __ATOMIC_RELEASE);
}

PlatIntr *plat_intr_self(void) {
    static PlatMutex install = PLAT_MUTEX_INIT;
    if (!__atomic_load_n(&intr_sig, __ATOMIC_ACQUIRE)) {
        plat_mutex_lock(&install);
        if (!intr_sig) {
            int sig = SIGUSR2;
            struct sigaction sa = {.sa_sigaction = intr_handler, .sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK};
            sigemptyset(&sa.sa_mask);
            if (!sigaction(sig, &sa, NULL)) __atomic_store_n(&intr_sig, sig, __ATOMIC_RELEASE);
        }
        plat_mutex_unlock(&install);
    }
    if (!intr_sig) return NULL;
    PlatIntr *t = calloc(1, sizeof *t);
    if (!t) return NULL;
    t->pt = pthread_self();
    intr_me = t;
    return t;
}

// Called by the thread itself as it ends, or by another once the thread no longer runs.
void plat_intr_free(PlatIntr *t) {
    if (!t) return;
    if (intr_me == t) intr_me = NULL;
    free(t);
}

int plat_intr_stop(PlatIntr *t, uint64_t *rip, uint64_t *rsp) {
    int32_t idle = INTR_IDLE;
    if (!__atomic_compare_exchange_n(&t->state, &idle, INTR_REQ, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return 0;
    if (pthread_kill(t->pt, intr_sig)) {
        __atomic_store_n(&t->state, INTR_IDLE, __ATOMIC_RELEASE);
        return 0;
    }
    uint64_t end = plat_now_ns() + INTR_WAIT_NS;
    while (__atomic_load_n(&t->state, __ATOMIC_ACQUIRE) != INTR_STOPPED) {
        if (plat_now_ns() < end) {
            plat_pause();
            continue;
        }
        int32_t req = INTR_REQ;
        if (__atomic_compare_exchange_n(&t->state, &req, INTR_IDLE, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return 0;
    }
    *rip = t->rip;
    *rsp = t->rsp;
    return 1;
}

int plat_intr_resume(PlatIntr *t, uint64_t to) {
    t->to = to;
    __atomic_store_n(&t->state, INTR_GO, __ATOMIC_RELEASE);
    return 1;
}

// ---- fibers: a stack switch of our own (swapcontext is deprecated here and makes a system call for the signal mask).
// fiber_swap saves the callee-saved registers and the floating-point control state on the current stack, stores the
// stack pointer in *save and loads the other one. A new fiber's stack is laid out as fiber_swap leaves one, returning
// into fiber_entry, which calls fn(arg).
struct PlatFiber {
    void *sp;
    void *stack;   // mapped stack (NULL for a thread's own context)
    size_t size;
};

void plat_fiber_swap(void **save, void *load);
void plat_fiber_entry(void);
#if defined(__x86_64__)
// fn and arg travel in r12 and r13.
__asm__(".text\n"
        ASM_FN("plat_fiber_swap")
        "    pushq %rbp\n"
        "    pushq %rbx\n"
        "    pushq %r12\n"
        "    pushq %r13\n"
        "    pushq %r14\n"
        "    pushq %r15\n"
        "    subq $16, %rsp\n"
        "    stmxcsr 8(%rsp)\n"
        "    fnstcw (%rsp)\n"
        "    movq %rsp, (%rdi)\n"
        "    movq %rsi, %rsp\n"
        "    fldcw (%rsp)\n"
        "    ldmxcsr 8(%rsp)\n"
        "    addq $16, %rsp\n"
        "    popq %r15\n"
        "    popq %r14\n"
        "    popq %r13\n"
        "    popq %r12\n"
        "    popq %rbx\n"
        "    popq %rbp\n"
        "    ret\n"
        ASM_FN("plat_fiber_entry")
        "    andq $-16, %rsp\n"   // fiber_swap's ret leaves rsp 8 off: the SysV ABI wants it aligned at the call
        "    movq %r13, %rdi\n"
        "    callq *%r12\n"
        "    ud2\n");
#define FIBER_WORDS 10   // fpu state (2 words), r15, r14, r13, r12, rbx, rbp, return address, padding

static void fiber_frame(uint64_t *sp, void (*fn)(void *arg), void *arg) {
    sp[0] = 0x037f;            // x87 control word: the default
    sp[1] = 0x1f80;            // MXCSR: the default
    sp[4] = (uint64_t)(uintptr_t)arg;   // r13
    sp[5] = (uint64_t)(uintptr_t)fn;    // r12
    sp[8] = (uint64_t)(uintptr_t)plat_fiber_entry;
}
#else
// The frame, from sp up: x19-x28, x29, x30, d8-d15, FPCR, padding (22 words). fn and arg travel in x19 and x20.
__asm__(".text\n"
        ASM_FN("plat_fiber_swap")
        "    sub sp, sp, #176\n"
        "    stp x19, x20, [sp, #0]\n"
        "    stp x21, x22, [sp, #16]\n"
        "    stp x23, x24, [sp, #32]\n"
        "    stp x25, x26, [sp, #48]\n"
        "    stp x27, x28, [sp, #64]\n"
        "    stp x29, x30, [sp, #80]\n"
        "    stp d8, d9, [sp, #96]\n"
        "    stp d10, d11, [sp, #112]\n"
        "    stp d12, d13, [sp, #128]\n"
        "    stp d14, d15, [sp, #144]\n"
        "    mrs x9, fpcr\n"
        "    str x9, [sp, #160]\n"
        "    mov x9, sp\n"
        "    str x9, [x0]\n"
        "    mov sp, x1\n"
        "    ldr x9, [sp, #160]\n"
        "    msr fpcr, x9\n"
        "    ldp x19, x20, [sp, #0]\n"
        "    ldp x21, x22, [sp, #16]\n"
        "    ldp x23, x24, [sp, #32]\n"
        "    ldp x25, x26, [sp, #48]\n"
        "    ldp x27, x28, [sp, #64]\n"
        "    ldp x29, x30, [sp, #80]\n"
        "    ldp d8, d9, [sp, #96]\n"
        "    ldp d10, d11, [sp, #112]\n"
        "    ldp d12, d13, [sp, #128]\n"
        "    ldp d14, d15, [sp, #144]\n"
        "    add sp, sp, #176\n"
        "    ret\n"
        ASM_FN("plat_fiber_entry")
        "    mov x0, x20\n"
        "    blr x19\n"
        "    brk #1\n");
#define FIBER_WORDS 22

static void fiber_frame(uint64_t *sp, void (*fn)(void *arg), void *arg) {
    sp[0] = (uint64_t)(uintptr_t)fn;    // x19
    sp[1] = (uint64_t)(uintptr_t)arg;   // x20
    sp[11] = (uint64_t)(uintptr_t)plat_fiber_entry;   // x30
}
#endif

static __thread PlatFiber *fiber_cur;
static __thread PlatFiber fiber_own;

PlatFiber *plat_fiber_self(void) {
    fiber_own = (PlatFiber){0};
    fiber_cur = &fiber_own;
    return &fiber_own;
}

void plat_fiber_unself(void) { fiber_cur = NULL; }

PlatFiber *plat_fiber_new(size_t stack, void (*fn)(void *arg), void *arg) {
    PlatFiber *f = calloc(1, sizeof *f);
    if (!f) return NULL;
    // Backed only as it is touched; the lowest page stays inaccessible, so an overflow faults instead of corrupting.
    void *m = mmap(NULL, stack, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (m == MAP_FAILED) {
        free(f);
        return NULL;
    }
    mprotect(m, (size_t)getpagesize(), PROT_NONE);
    f->stack = m, f->size = stack;
    uint64_t *top = (uint64_t *)(((uintptr_t)m + stack) & ~(uintptr_t)15);
    uint64_t *sp = top - FIBER_WORDS;
    memset(sp, 0, FIBER_WORDS * sizeof *sp);
    fiber_frame(sp, fn, arg);
    f->sp = sp;
    return f;
}

void plat_fiber_switch(PlatFiber *to) {
    PlatFiber *from = fiber_cur;
    fiber_cur = to;   // on this thread; the switch back into from sets it on whichever thread that is
    plat_fiber_swap(&from->sp, to->sp);
}

void plat_fiber_free(PlatFiber *f) {
    if (!f) return;
    if (f->stack) munmap(f->stack, f->size);
    free(f);
}

int plat_thread_spawn(PlatThreadFn fn, void *arg, const char *name, size_t stack) {
    PlatThread *t = plat_thread_new_ex(fn, arg, name, stack);
    if (t) plat_thread_detach(t);
    return t != NULL;
}

static void qos_unboost(PlatThread *t) {
    pthread_override_t o = __atomic_exchange_n(&t->boost, NULL, __ATOMIC_ACQ_REL);
    if (o) pthread_override_qos_class_end_np(o);
}

void plat_thread_join(PlatThread *t) {
    qos_unboost(t);
    pthread_join(t->t, NULL);
    free(t);
}

// There is no timed join here: the thread says when its function has returned, and the join that follows is short.
int plat_thread_join_ns(PlatThread *t, uint64_t ns) {
    uint64_t end = plat_now_ns() + ns;
    while (!__atomic_load_n(&t->done, __ATOMIC_ACQUIRE)) {
        uint64_t now = plat_now_ns();
        if (now >= end) return 0;
        plat_wait_ns(end - now < 1000000u ? end - now : 1000000u);
    }
    qos_unboost(t);
    pthread_join(t->t, NULL);
    free(t);
    return 1;
}

// The thread keeps using t->fn and t->arg, so the struct stays allocated (a few bytes per detached thread).
void plat_thread_detach(PlatThread *t) { pthread_detach(t->t); }

uint32_t plat_cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (uint32_t)n : 1;
}

uint32_t plat_cpu_usable(void) { return plat_cpu_count(); }

// Windows' levels map to the importance of Mach's precedence policy, an offset to the thread's base priority inside
// the process: IDLE -20, LOWEST -4, BELOW_NORMAL -2, NORMAL 0, ABOVE_NORMAL 2, HIGHEST 5, TIME_CRITICAL 10. Any
// thread may be set, and a lowered one raised again, without a privilege.
static int importance_of(int prio) {
    return prio <= PLAT_PRIO_IDLE ? -20 : prio == PLAT_PRIO_LOWEST ? -4 : prio == PLAT_PRIO_BELOW_NORMAL ? -2
         : prio == PLAT_PRIO_NORMAL ? 0 : prio == PLAT_PRIO_ABOVE_NORMAL ? 2 : prio == PLAT_PRIO_HIGHEST ? 5 : 10;
}

static void set_importance(pthread_t th, int prio) {
    thread_precedence_policy_data_t p = {importance_of(prio)};
    thread_policy_set(pthread_mach_thread_np(th), THREAD_PRECEDENCE_POLICY, (thread_policy_t)&p,
                      THREAD_PRECEDENCE_POLICY_COUNT);
}

// With QoS on: the calling thread's own class; a raise another thread started on it ends, as the thread now says
// for itself what it runs at.
static void qos_set_self(int prio) {
    PlatThread *t = qos_self;
    int rel;
    qos_class_t q = qos_of_prio(prio, t ? t->home : QOS_HOME_NONE, &rel);
    pthread_set_qos_class_self_np(q, rel);
    if (t) qos_unboost(t);
}

void plat_thread_priority(int prio) {
    if (qos_on) qos_set_self(prio);
    else set_importance(pthread_self(), prio);
}

// With QoS on, another thread's class cannot be set: a raise to normal or above starts an override of the class the
// level maps to (one per thread; the callers repeat their raise while they wait), and a level below normal ends it.
void plat_thread_set_priority(PlatThread *t, int prio) {
    if (!qos_on) {
        set_importance(t->t, prio);
        return;
    }
    if (pthread_equal(t->t, pthread_self())) {
        qos_set_self(prio);
        return;
    }
    if (prio < PLAT_PRIO_NORMAL) {
        qos_unboost(t);
        return;
    }
    if (__atomic_load_n(&t->boost, __ATOMIC_ACQUIRE)) return;
    int rel;
    qos_class_t q = qos_of_prio(prio, t->home, &rel);
    pthread_override_t o = pthread_override_qos_class_start_np(t->t, q, rel);
    pthread_override_t none = NULL;
    if (o && !__atomic_compare_exchange_n(&t->boost, &none, o, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        pthread_override_qos_class_end_np(o);   // another caller's raise got there first
}

int plat_thread_priority_get(void) {
    if (qos_on) {
        qos_class_t q = QOS_CLASS_UNSPECIFIED;
        int rel = 0;
        pthread_get_qos_class_np(pthread_self(), &q, &rel);
        return q == QOS_CLASS_USER_INTERACTIVE ? PLAT_PRIO_HIGHEST : q == QOS_CLASS_USER_INITIATED ? PLAT_PRIO_ABOVE_NORMAL
             : q == QOS_CLASS_UTILITY ? (rel < 0 ? PLAT_PRIO_LOWEST : PLAT_PRIO_BELOW_NORMAL)
             : q == QOS_CLASS_BACKGROUND ? PLAT_PRIO_IDLE : PLAT_PRIO_NORMAL;
    }
    thread_precedence_policy_data_t p = {0};
    mach_msg_type_number_t n = THREAD_PRECEDENCE_POLICY_COUNT;
    boolean_t def = 0;
    if (thread_policy_get(pthread_mach_thread_np(pthread_self()), THREAD_PRECEDENCE_POLICY, (thread_policy_t)&p, &n,
                          &def) != KERN_SUCCESS || def)
        return PLAT_PRIO_NORMAL;
    int i = p.importance;
    return i <= -20 ? PLAT_PRIO_IDLE : i <= -4 ? PLAT_PRIO_LOWEST : i <= -2 ? PLAT_PRIO_BELOW_NORMAL
         : i <= 0 ? PLAT_PRIO_NORMAL : i <= 2 ? PLAT_PRIO_ABOVE_NORMAL : i <= 5 ? PLAT_PRIO_HIGHEST
         : PLAT_PRIO_TIME_CRITICAL;
}

void plat_yield(void) { sched_yield(); }

// ---- time: CLOCK_UPTIME_RAW is mach_absolute_time in nanoseconds; it does not count while the machine sleeps.

uint64_t plat_now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

uint64_t plat_counter(void) { return plat_now_ns(); }
uint64_t plat_counter_hz(void) { return 1000000000u; }

void plat_timer_1ms(void) {}              // timed waits are finer than 1 ms as they are
int plat_power_full(void) { return 0; }

// mach_wait_until takes its deadline in the units of mach_absolute_time; the last PLAT_SPIN_NS are spent yielding so
// the deadline is met closely, as on Windows.
#define PLAT_SPIN_NS 100000u

void plat_sleep_ns(uint64_t ns) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    uint64_t end = plat_now_ns() + ns;
    if (ns > PLAT_SPIN_NS)
        mach_wait_until((uint64_t)((unsigned __int128)(end - PLAT_SPIN_NS) * tb.denom / tb.numer));
    while (plat_now_ns() < end) sched_yield();
}

void plat_wait_ns(uint64_t ns) {
    struct timespec ts = {(time_t)(ns / 1000000000u), (long)(ns % 1000000000u)};
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
}

void plat_sleep_ms(uint32_t ms) {
    struct timespec ts = {(time_t)(ms / 1000), (long)(ms % 1000) * 1000000};
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
}

// ---- locks: a reader-writer lock in one 32-bit word, zero when free.
// bit 31: a writer holds it; bit 30: some thread sleeps on the word; bits 0-29: readers holding it.
// A thread that cannot take the lock sets the sleeper bit and waits on the futex; every release that finds the bit
// clears it and wakes all sleepers, which retry (and set it again if they still cannot get in). Readers do not enter
// while the sleeper bit is set, so a waiting writer is not starved by a stream of readers.

#define LK_W 0x80000000u
#define LK_S 0x40000000u
#define LK_R 0x3fffffffu
#define LK_SPINS 64

// The two futex calls the locks below are written against, on the system's wait-on-address: FUTEX_WAIT sleeps while
// the word still holds val (at most ts when given; -1 with errno ETIMEDOUT when that passes), FUTEX_WAKE wakes one
// sleeper for a val of 1 and all of them otherwise.
enum { FUTEX_WAIT, FUTEX_WAKE };

static long futex(volatile uint32_t *w, int op, uint32_t val, const struct timespec *ts) {
    void *a = (void *)(uintptr_t)w;
    if (op == FUTEX_WAKE)
        return val == 1 ? os_sync_wake_by_address_any(a, 4, OS_SYNC_WAKE_BY_ADDRESS_NONE)
                        : os_sync_wake_by_address_all(a, 4, OS_SYNC_WAKE_BY_ADDRESS_NONE);
    if (!ts) return os_sync_wait_on_address(a, val, 4, OS_SYNC_WAIT_ON_ADDRESS_NONE);
    uint64_t ns = (uint64_t)ts->tv_sec * 1000000000u + (uint64_t)ts->tv_nsec;
    if (!ns) {
        errno = ETIMEDOUT;
        return -1;
    }
    return os_sync_wait_on_address_with_timeout(a, val, 4, OS_SYNC_WAIT_ON_ADDRESS_NONE, OS_CLOCK_MACH_ABSOLUTE_TIME,
                                                ns);
}

static inline volatile uint32_t *lk(PlatMutex *m) { return (volatile uint32_t *)&m->p; }

// Sets the sleeper bit on s and sleeps while the word still equals the result.
static void lk_sleep(volatile uint32_t *w, uint32_t s) {
    if (!(s & LK_S) && !__atomic_compare_exchange_n(w, &s, s | LK_S, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return;
    futex(w, FUTEX_WAIT, s | LK_S, NULL);
}

static void lk_wake(volatile uint32_t *w) {
    __atomic_fetch_and(w, ~LK_S, __ATOMIC_RELAXED);
    futex(w, FUTEX_WAKE, INT_MAX, NULL);
}

void plat_mutex_init(PlatMutex *m) { m->p = 0; }

int plat_mutex_trylock(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
    while (!(s & (LK_W | LK_R)))
        if (__atomic_compare_exchange_n(w, &s, s | LK_W, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return 1;
    return 0;
}

// Lock-wait timing is Windows only (platform.h): here the note is kept and never called.
void plat_lock_timing(PlatLockNote fn) { (void)fn; }
void plat_mutex_lock_at(PlatMutex *m, void *site) { (void)site; plat_mutex_lock(m); }

void plat_mutex_lock(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    for (int spin = 0;; spin++) {
        uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
        if (!(s & (LK_W | LK_R))) {
            if (__atomic_compare_exchange_n(w, &s, s | LK_W, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return;
            continue;
        }
        if (spin < LK_SPINS) plat_pause();
        else lk_sleep(w, s);
    }
}

void plat_mutex_unlock(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_fetch_and(w, ~LK_W, __ATOMIC_RELEASE);
    if (s & LK_S) lk_wake(w);
}

int plat_mutex_trylock_shared(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
    while (!(s & (LK_W | LK_S)))
        if (__atomic_compare_exchange_n(w, &s, s + 1, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return 1;
    return 0;
}

void plat_mutex_lock_shared(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    for (int spin = 0;; spin++) {
        uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
        if (!(s & (LK_W | LK_S))) {
            if (__atomic_compare_exchange_n(w, &s, s + 1, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return;
            continue;
        }
        if (spin < LK_SPINS) plat_pause();
        else lk_sleep(w, s);
    }
}

void plat_mutex_unlock_shared(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_sub_fetch(w, 1, __ATOMIC_RELEASE);
    if (!(s & LK_R) && (s & LK_S)) lk_wake(w);
}

// ---- condition variables: a sequence number in the first 32 bits. A waiter reads it before releasing the lock and
// sleeps only while it is unchanged, so a signal between the release and the sleep is not lost.

static inline volatile uint32_t *cv(PlatCond *c) { return (volatile uint32_t *)&c->p; }

void plat_cond_init(PlatCond *c) { c->p = 0; }

void plat_cond_wait(PlatCond *c, PlatMutex *m) {
    uint32_t seq = __atomic_load_n(cv(c), __ATOMIC_RELAXED);
    plat_mutex_unlock(m);
    futex(cv(c), FUTEX_WAIT, seq, NULL);
    plat_mutex_lock(m);
}

int plat_cond_wait_ns(PlatCond *c, PlatMutex *m, uint64_t ns) {
    uint32_t seq = __atomic_load_n(cv(c), __ATOMIC_RELAXED);
    struct timespec ts = {(time_t)(ns / 1000000000u), (long)(ns % 1000000000u)};
    plat_mutex_unlock(m);
    long r = futex(cv(c), FUTEX_WAIT, seq, &ts);
    int timed_out = r == -1 && errno == ETIMEDOUT;
    plat_mutex_lock(m);
    return !timed_out;
}

void plat_cond_signal(PlatCond *c) {
    __atomic_add_fetch(cv(c), 1, __ATOMIC_RELEASE);
    futex(cv(c), FUTEX_WAKE, 1, NULL);
}

void plat_cond_broadcast(PlatCond *c) {
    __atomic_add_fetch(cv(c), 1, __ATOMIC_RELEASE);
    futex(cv(c), FUTEX_WAKE, INT_MAX, NULL);
}

// ---- files

int plat_file_exists(const char *path) {
    struct stat st;
    return !stat(path, &st) && !S_ISDIR(st.st_mode);
}

int plat_dir_exists(const char *path) {
    struct stat st;
    return !stat(path, &st) && S_ISDIR(st.st_mode);
}

int plat_full_path(const char *path, char *out, size_t cap) {
    if (path[0] == '/') return snprintf(out, cap, "%s", path) < (int)cap;
    char cwd[PATH_MAX];
    if (!getcwd(cwd, sizeof cwd)) return 0;
    return snprintf(out, cap, "%s/%s", cwd, path) < (int)cap;
}

void plat_file_mkdir_p(const char *path) {
    char tmp[PATH_MAX];
    if (snprintf(tmp, sizeof tmp, "%s", path) >= (int)sizeof tmp) return;
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            char c = *p;
            *p = 0;
            mkdir(tmp, 0755);
            *p = c;
        }
    }
    mkdir(tmp, 0755);
}

static int write_all(int fd, const void *data, size_t n) {
    const uint8_t *p = data;
    while (n) {
        ssize_t k = write(fd, p, n);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return 0;
        p += k, n -= (size_t)k;
    }
    return 1;
}

uint8_t *plat_file_read(const char *path, uint64_t *size) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return NULL;
    struct stat st;
    uint8_t *buf = NULL;
    if (!fstat(fd, &st) && S_ISREG(st.st_mode) && st.st_size > 0 && st.st_size < (1ll << 30)) {
        buf = malloc((size_t)st.st_size + 1);
        size_t got = 0;
        while (buf && got < (size_t)st.st_size) {
            ssize_t k = read(fd, buf + got, (size_t)st.st_size - got);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) break;
            got += (size_t)k;
        }
        if (buf && got != (size_t)st.st_size) {
            free(buf);
            buf = NULL;
        }
        if (buf) {
            buf[st.st_size] = 0;
            *size = (uint64_t)st.st_size;
        }
    }
    close(fd);
    return buf;
}

int plat_file_write(const char *path, const void *data, size_t n) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return 0;
    int ok = write_all(fd, data, n);
    return close(fd) == 0 && ok;
}

// O_APPEND makes each write land whole at the end of the file, from any thread.
PlatAppend *plat_append_open(const char *path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC, 0644);
    return fd < 0 ? NULL : (PlatAppend *)(intptr_t)(fd + 1);
}

int plat_append_write(PlatAppend *f, const void *data, size_t n) {
    int fd = (int)(intptr_t)f - 1;
    ssize_t k;
    do k = write(fd, data, n);
    while (k < 0 && errno == EINTR);
    return k == (ssize_t)n;
}

// Flushes the directory holding path, so a rename into it survives a crash.
static void sync_dir_of(const char *path) {
    char dir[PATH_MAX];
    if (snprintf(dir, sizeof dir, "%s", path) >= (int)sizeof dir) return;
    char *slash = strrchr(dir, '/');
    if (slash == dir) slash[1] = 0;
    else if (slash) *slash = 0;
    else strcpy(dir, ".");
    int fd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
}

int plat_file_replace(const char *from, const char *to, int durable) {
    if (rename(from, to)) return 0;
    if (durable) sync_dir_of(to);
    return 1;
}

static char lock_held[4][1100];
static int lock_fd[4];
static int lock_n;
static PlatMutex lock_lock = PLAT_MUTEX_INIT;

int plat_lock_file(const char *path) {
    plat_mutex_lock(&lock_lock);
    int r = 1, nheld = lock_n;
    for (int i = 0; i < nheld; i++)
        if (!strcmp(lock_held[i], path)) goto done;
    int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0644);   // open until plat_unlock_file or the process ends
    if (fd < 0) {
        r = -1;
        goto done;
    }
    if (flock(fd, LOCK_EX | LOCK_NB)) {
        r = errno == EWOULDBLOCK ? 0 : -1;
        close(fd);
        goto done;
    }
    if (nheld < 4) {
        snprintf(lock_held[nheld], sizeof lock_held[0], "%s", path);
        lock_fd[nheld] = fd;
        lock_n = nheld + 1;
    }
done:
    plat_mutex_unlock(&lock_lock);
    return r;
}

void plat_unlock_file(const char *path) {
    plat_mutex_lock(&lock_lock);
    for (int i = 0; i < lock_n; i++)
        if (!strcmp(lock_held[i], path)) {
            close(lock_fd[i]);
            lock_n--;
            memmove(&lock_held[i], &lock_held[i + 1], sizeof lock_held[0] * (size_t)(lock_n - i));
            memmove(&lock_fd[i], &lock_fd[i + 1], sizeof lock_fd[0] * (size_t)(lock_n - i));
            break;
        }
    plat_mutex_unlock(&lock_lock);
}

// path.<pid>.<tid>.tmp renamed over path; durable: the file and then its folder are synced to the disk.
static int write_swap(const char *path, const void *data, size_t n, int durable) {
    // A rename replaces a read-only file here (only the folder's permission counts) but not on Windows: a file the
    // player made read-only stays as it is on both.
    if (access(path, W_OK) && errno != ENOENT) return 0;
    char tmp[PATH_MAX + 48];
    int len = snprintf(tmp, sizeof tmp, "%s.%u.%u.tmp", path, plat_process_id(), plat_thread_id());
    if (len < 0 || len >= (int)sizeof tmp) return 0;
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return 0;
    int ok = write_all(fd, data, n) && (!durable || fsync(fd) == 0);
    ok = close(fd) == 0 && ok;
    if (ok && !rename(tmp, path)) {
        if (durable) sync_dir_of(path);
        return 1;
    }
    unlink(tmp);
    return 0;
}

int plat_file_write_atomic(const char *path, const void *data, size_t n) { return write_swap(path, data, n, 1); }

int plat_file_write_cached(const char *path, const void *data, size_t n) { return write_swap(path, data, n, 0); }

int plat_file_flush(const char *path) {
    int fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    int ok = fsync(fd) == 0;
    close(fd);
    return ok;
}

int plat_file_sync(FILE *f) { return !fflush(f) && !fsync(fileno(f)); }

void plat_file_delete(const char *path) { unlink(path); }

int plat_file_truncate(const char *path, uint64_t size) { return truncate(path, (off_t)size) == 0; }

int64_t plat_file_size(const char *path) {
    struct stat st;
    return stat(path, &st) ? -1 : (int64_t)st.st_size;
}

// The owner's write permission stands for Windows' read-only attribute.
int plat_file_set_readonly(const char *path, int on) {
    struct stat st;
    if (stat(path, &st)) return 0;
    mode_t m = on ? st.st_mode & ~(mode_t)0222 : st.st_mode | S_IWUSR;
    return chmod(path, m & 07777) == 0;
}

int plat_file_private(const char *path) {
    struct stat st;
    if (stat(path, &st)) return 0;
    return chmod(path, S_ISDIR(st.st_mode) ? 0700 : 0600) == 0;
}

int plat_file_list(const char *dir, const char *ext, int (*cb)(const char *name, void *ctx), void *ctx) {
    DIR *d = opendir(dir);
    if (!d) return 0;
    size_t el = strlen(ext);
    for (struct dirent *e; (e = readdir(d));) {
        size_t n = strlen(e->d_name);
        if (n < el || strcmp(e->d_name + n - el, ext) || e->d_name[0] == '.') continue;
        if (!cb(e->d_name, ctx)) break;
    }
    closedir(d);
    return 1;
}

// ---- virtual memory. Reservations are inaccessible anonymous mappings; commit is mprotect read-write, and decommit
// maps fresh inaccessible pages over the range (they read zero when committed again). The system keeps no
// committed/reserved distinction, so a byte per guest page records it (pstate, mapped lazily: only the pages touched
// cost memory).
//
// Guest pages are 4 KiB and the host's are 16 KiB on ARM64. There a host page gets the most permissive protection of
// the guest pages in it: every access the guest's protections allow works, and an access they forbid faults only
// when the whole host page forbids it. A guest page decommitted beside one still committed is cleared by hand.
//
// Write watch: the system has no record of written pages to read and reset, so a watched reservation keeps one bit per
// guest page (dirty). A read-write host page is writable only while the bits of all its guest pages are set; otherwise
// it is mapped read-only, the first write to it faults, and the fault handler sets those bits and makes the page
// writable before the write runs again (watch_fault). plat_mem_written reports the guest pages whose bit is set and,
// with reset, clears the bits it reported and maps their host pages read-only again. A page just
// committed has no write on record, as on the other systems. The kernel does not fault on such a page, it refuses to
// write into it, so plat_file_pread marks its buffer written first (watch_touch). Writes are recorded per host page
// (plat_mem_watch_grain): on ARM64 a written host page reports its four guest pages.
// NX2NAVI_WRITE_WATCH=0 turns the tracking off: plat_mem_written then reports every range as untracked and the back
// end uploads it in full.

#define PG 4096ull
#define PS_LIMIT (1ull << 39)
enum { PS_FREE, PS_RESERVED, PS_NONE, PS_READ, PS_RW };   // PS_NONE.. are committed: PLAT_PROT_* + PS_NONE
static uint8_t *pstate;
static uint64_t hpg;      // the host's page size, once pstate is mapped
static unsigned hshift;   // its log2
static uint64_t *dirty;   // a bit per guest page below PS_LIMIT: written since the last reset (watched reservations)
static struct { uint64_t lo, hi; } watch[8];   // the watched reservations; entries below nwatch never change
static volatile int32_t nwatch;
static int watch_off = -1;                 // NX2NAVI_WRITE_WATCH=0; -1 before the first reservation
static PlatMutex vm_lock = PLAT_MUTEX_INIT;
// Held around every change of a dirty bit together with the protection that goes with it. The fault handler takes it,
// so it is a spin lock, and no code holding it writes guest memory.
static volatile int32_t watch_lock;

static void watch_take(void) {
    while (__atomic_exchange_n(&watch_lock, 1, __ATOMIC_ACQUIRE)) plat_pause();
}

static void watch_give(void) { __atomic_store_n(&watch_lock, 0, __ATOMIC_RELEASE); }

static int watched(uint64_t a) {
    for (int i = 0, n = __atomic_load_n(&nwatch, __ATOMIC_ACQUIRE); i < n; i++)
        if (a >= watch[i].lo && a < watch[i].hi) return 1;
    return 0;
}

// The dirty bits of the guest pages in the host page at h, as a mask in their word of the map (a host page's guest
// pages are a power of two at most 64, aligned alike, so they share a word).
static uint64_t dirty_mask(uint64_t h, uint64_t **word) {
    uint64_t i = h / PG, per = hpg / PG;
    *word = &dirty[i >> 6];
    return (per == 64 ? ~0ull : (1ull << per) - 1) << (i & 63);
}

// Every guest page of the host page at h has a write on record.
static int dirty_all(uint64_t h) {
    uint64_t *w, m = dirty_mask(h, &w);
    return (__atomic_load_n(w, __ATOMIC_RELAXED) & m) == m;
}

// Sets or clears the bits of every guest page in the host pages of [a, a + n).
static void dirty_put(uint64_t a, uint64_t n, int on) {
    for (uint64_t h = a & ~(hpg - 1), e = a + n; h < e; h += hpg) {
        uint64_t *w, m = dirty_mask(h, &w);
        if (on) __atomic_fetch_or(w, m, __ATOMIC_RELAXED);
        else __atomic_fetch_and(w, ~m, __ATOMIC_RELAXED);
    }
}

static void ps_set(uint64_t addr, uint64_t n, uint8_t v) {
    if (addr < PS_LIMIT) memset(pstate + addr / PG, v, (size_t)((n + PG - 1) / PG));
}

// vm_lock held.
static int vm_init(void) {
    if (pstate) return 1;
    hpg = (uint64_t)getpagesize();
    hshift = (unsigned)__builtin_ctzll(hpg);
    void *p = mmap(NULL, PS_LIMIT / PG, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    void *d = mmap(NULL, PS_LIMIT / PG / 8, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED || d == MAP_FAILED) return 0;
    dirty = d;
    pstate = p;
    return 1;
}

// The most permissive state of the guest pages in the host page at h.
static uint8_t host_state(uint64_t h) {
    uint8_t best = PS_FREE;
    for (uint64_t g = h; g < h + hpg; g += PG)
        if (pstate[g / PG] > best) best = pstate[g / PG];
    return best;
}

// The range's protections follow pstate through host_apply: its host pages are larger than the guest's, or watched.
static int vm_managed(uint64_t addr, uint64_t n) {
    return pstate && addr + n <= PS_LIMIT && (hpg > PG || watched(addr));
}

// Gives each host page of [addr, addr + n) the protection its guest pages ask for, read-only in place of read-write
// where a watched page has no write on record. vm_lock held.
static int host_apply(uint64_t addr, uint64_t n) {
    uint64_t lo = addr & ~(hpg - 1), hi = (addr + n + hpg - 1) & ~(hpg - 1), run = lo;
    int run_prot = -1, ok = 1, w = watched(lo);
    watch_take();
    for (uint64_t h = lo; h < hi; h += hpg) {
        uint8_t st = host_state(h);
        int p = st == PS_RW ? (w && !dirty_all(h) ? PROT_READ : PROT_READ | PROT_WRITE) : st == PS_READ ? PROT_READ
              : PROT_NONE;
        if (p == run_prot) continue;
        if (run_prot >= 0 && mprotect((void *)(uintptr_t)run, h - run, run_prot)) ok = 0;
        run = h, run_prot = p;
    }
    if (run_prot >= 0 && mprotect((void *)(uintptr_t)run, hi - run, run_prot)) ok = 0;
    watch_give();
    return ok;
}

// A write faulted at addr: 1 when the page is a watched read-write one, now on record as written and writable. The
// fault handler calls it, so it takes no lock but the spin lock.
static int watch_fault(uint64_t addr) {
    if (!pstate || addr >= PS_LIMIT || !watched(addr)) return 0;
    uint64_t h = addr & ~(hpg - 1);
    if (host_state(h) != PS_RW) return 0;
    watch_take();
    dirty_put(h, hpg, 1);
    int ok = !mprotect((void *)(uintptr_t)h, hpg, PROT_READ | PROT_WRITE);
    watch_give();
    return ok;
}

// The kernel is about to write [addr, addr + n): its watched read-write pages go on record as written and become
// writable, as a write of our own would make them.
static void watch_touch(uint64_t addr, uint64_t n) {
    if (!n || !pstate || addr >= PS_LIMIT || !watched(addr)) return;
    for (uint64_t h = addr & ~(hpg - 1), e = addr + n < PS_LIMIT ? addr + n : PS_LIMIT; h < e; h += hpg)
        if (!dirty_all(h)) watch_fault(h);
}

static void fault_install(void);

// Without MAP_FIXED the address is where the search for free space starts: the mapping lands there exactly when the
// range is free, and elsewhere (undone below) when any of it is taken.
int plat_mem_reserve(uint64_t addr, uint64_t n, int watch_it) {
    plat_mutex_lock(&vm_lock);
    if (watch_off < 0) {
        const char *e = getenv("NX2NAVI_WRITE_WATCH");
        watch_off = e && !strcmp(e, "0");
    }
    int ok = vm_init() && addr + n <= PS_LIMIT && !((addr | n) & (hpg - 1));
    if (ok) {
        void *p = mmap((void *)(uintptr_t)addr, n, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
        ok = p == (void *)(uintptr_t)addr;
        if (p != MAP_FAILED && !ok) munmap(p, n);
    }
    if (ok) {
        ps_set(addr, n, PS_RESERVED);
        int k = nwatch;
        if (watch_it && !watch_off && k < (int)(sizeof watch / sizeof *watch)) {
            fault_install();   // the watch works through the fault handler
            watch[k].lo = addr, watch[k].hi = addr + n;
            __atomic_store_n(&nwatch, k + 1, __ATOMIC_RELEASE);
        }
    }
    plat_mutex_unlock(&vm_lock);
    return ok;
}

// Recorded as committed where pstate reaches, so a protection change beside it (host_apply) counts it as in use.
void *plat_mem_alloc(size_t n) {
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) return NULL;
    plat_mutex_lock(&vm_lock);
    if (vm_init() && (uint64_t)(uintptr_t)p + n <= PS_LIMIT) ps_set((uint64_t)(uintptr_t)p, n, PS_RW);
    plat_mutex_unlock(&vm_lock);
    return p;
}

int plat_mem_commit(uint64_t addr, size_t n) {
    if (!vm_managed(addr, n)) {
        if (mprotect((void *)(uintptr_t)addr, n, PROT_READ | PROT_WRITE)) return 0;
        ps_set(addr, n, PS_RW);
        return 1;
    }
    plat_mutex_lock(&vm_lock);
    ps_set(addr, n, PS_RW);
    int ok = host_apply(addr, n);
    plat_mutex_unlock(&vm_lock);
    return ok;
}

// vm_lock held.
static int map_fresh(uint64_t addr, uint64_t n) {
    if (mmap((void *)(uintptr_t)addr, n, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) == MAP_FAILED) return 0;
    if (pstate && addr + n <= PS_LIMIT) {
        watch_take();
        dirty_put(addr, n, 0);
        watch_give();
    }
    return 1;
}

// The guest pages [a, e) of one host page: cleared in place when the page keeps other committed guest pages, else
// mapped fresh. vm_lock held.
static int clear_part(uint64_t a, uint64_t e) {
    uint64_t h = a & ~(hpg - 1);
    if (host_state(h) < PS_NONE) return map_fresh(h, hpg);
    if (mprotect((void *)(uintptr_t)h, hpg, PROT_READ | PROT_WRITE)) return 0;
    memset((void *)(uintptr_t)a, 0, e - a);
    return host_apply(h, hpg);
}

int plat_mem_decommit(uint64_t addr, uint64_t n) {
    plat_mutex_lock(&vm_lock);
    int ok = 1;
    if (!pstate || addr + n > PS_LIMIT) {
        ok = map_fresh(addr, n);
    } else {
        ps_set(addr, n, PS_RESERVED);
        uint64_t end = addr + n, lo = (addr + hpg - 1) & ~(hpg - 1), hi = end & ~(hpg - 1);
        if (lo > hi) {   // inside one host page
            ok = clear_part(addr, end);
        } else {
            if (addr < lo) ok &= clear_part(addr, lo);
            if (lo < hi) ok &= map_fresh(lo, hi - lo);
            if (hi < end) ok &= clear_part(hi, end);
        }
    }
    plat_mutex_unlock(&vm_lock);
    return ok;
}

int plat_mem_protect(uint64_t addr, uint64_t n, int prot) {
    if (!vm_managed(addr, n)) {
        int p = prot == PLAT_PROT_RW ? PROT_READ | PROT_WRITE : prot == PLAT_PROT_READ ? PROT_READ : PROT_NONE;
        if (mprotect((void *)(uintptr_t)addr, n, p)) return 0;
        if (pstate) ps_set(addr, n, (uint8_t)(PS_NONE + prot));
        return 1;
    }
    plat_mutex_lock(&vm_lock);
    ps_set(addr, n, (uint8_t)(PS_NONE + prot));
    int ok = host_apply(addr, n);
    plat_mutex_unlock(&vm_lock);
    return ok;
}

int plat_mem_state(uint64_t addr, uint64_t limit, uint64_t *end) {
    if (!pstate || addr >= PS_LIMIT) {
        *end = limit;
        return PLAT_MEM_FREE;
    }
    uint64_t lim = limit < PS_LIMIT ? limit : PS_LIMIT, i = addr / PG;
    uint8_t v = pstate[i];
    while ((i + 1) * PG < lim && pstate[i + 1] == v) i++;
    *end = (i + 1) * PG < lim ? (i + 1) * PG : lim;
    return v == PS_FREE ? PLAT_MEM_FREE : v == PS_RESERVED ? PLAT_MEM_RESERVED : v - PS_NONE;
}

int plat_mem_census(uint64_t *priv, uint64_t *wc, uint64_t *wc_allocs) {
    *priv = *wc = *wc_allocs = 0;
    return 0;
}

uint64_t plat_mem_watch_grain(void) { return (uint64_t)getpagesize(); }

// The watch protects pages here, so a host write over many of them faults page by page unless they are marked first.
int plat_mem_prepare_write(uint64_t addr, uint64_t n) {
    if (!n || !pstate || addr >= PS_LIMIT || !watched(addr)) return 0;
    watch_touch(addr, n);
    return 1;
}

// A Metal buffer made over host memory gives the GPU a read-only view of pages that are read-only at that moment, and
// drops what shaders write there without an error (measured on an M1 Max; protecting the pages after the buffer is
// made is fine: the GPU's writes land and the CPU sees them). The watch keeps unwritten pages read-only, so the back
// end brackets a host memory import with these two: open makes the range's unwritten pages writable without putting
// them on record, close protects them again. A page written in between has no fault to record it, so open keeps a sum
// of each such page and close puts those whose sum changed on record. vm_lock is held from open to close: nothing
// else changes a protection or reports written pages meanwhile, and the import makes no call into this file.
static uint64_t page_sum(uint64_t h) {
    const uint64_t *w = (const uint64_t *)(uintptr_t)h;
    uint64_t s = 0x9e3779b97f4a7c15ull;
    for (uint64_t i = 0; i < hpg / 8; i++) s = (s ^ w[i]) * 0x100000001b3ull + (s >> 29);
    return s | 1;   // never 0: 0 marks a page left alone
}

void plat_mem_import_open(uint64_t addr, uint64_t n, void **state) {
    *state = NULL;
    plat_mutex_lock(&vm_lock);
    if (!n || !pstate || addr + n > PS_LIMIT || !watched(addr)) return;
    uint64_t lo = addr & ~(hpg - 1), hi = (addr + n + hpg - 1) & ~(hpg - 1);
    uint64_t *sum = calloc((size_t)((hi - lo) / hpg), sizeof *sum);
    for (uint64_t h = lo; h < hi; h += hpg) {
        watch_take();
        if (host_state(h) == PS_RW && !dirty_all(h)) {
            if (sum) sum[(h - lo) / hpg] = page_sum(h);
            else dirty_put(h, hpg, 1);   // no memory for the sums: the page counts as written
            mprotect((void *)(uintptr_t)h, hpg, PROT_READ | PROT_WRITE);
        }
        watch_give();
    }
    *state = sum;
}

void plat_mem_import_close(uint64_t addr, uint64_t n, void *state) {
    uint64_t *sum = state;
    if (sum) {
        uint64_t lo = addr & ~(hpg - 1), hi = (addr + n + hpg - 1) & ~(hpg - 1);
        for (uint64_t h = lo; h < hi; h += hpg) {
            uint64_t was = sum[(h - lo) / hpg];
            if (!was) continue;
            watch_take();
            if (host_state(h) == PS_RW && !dirty_all(h)) {
                mprotect((void *)(uintptr_t)h, hpg, PROT_READ);   // a write from here on faults and is recorded
                if (page_sum(h) != was) {
                    dirty_put(h, hpg, 1);
                    mprotect((void *)(uintptr_t)h, hpg, PROT_READ | PROT_WRITE);
                }
            }
            watch_give();
        }
        free(sum);
    }
    plat_mutex_unlock(&vm_lock);
}

// The read-write guest pages of the range whose bit is set, in address order. With reset the bits reported are cleared
// and their host pages mapped read-only again, both under the spin lock, so a write lands either before that (and is
// in the memory the caller reads next) or after it (and faults, setting the bits again). When pages is full the scan
// stops and the pages not reported keep their bits: callers ask again while a call fills their list.
uint64_t plat_mem_written(uint64_t addr, uint64_t n, uint64_t *pages, uint64_t max, int reset) {
    if (!max || !n) return 0;
    if (!pstate || addr + n > PS_LIMIT || !watched(addr) || !watched(addr + n - 1)) return UINT64_MAX;
    uint64_t got = 0, end = addr + n;
    plat_mutex_lock(&vm_lock);
    for (uint64_t g = addr & ~(PG - 1); g < end && got < max;) {
        uint64_t i = g / PG, word = __atomic_load_n(&dirty[i >> 6], __ATOMIC_RELAXED) >> (i & 63);
        if (!word) {   // nothing written up to the end of this word of the map
            g = ((i | 63) + 1) * PG;
            continue;
        }
        if (!(word & 1)) {
            g += (uint64_t)__builtin_ctzll(word) * PG;
            continue;
        }
        // the set bits of g's host page inside the range, as far as pages has room
        uint64_t h = g & ~(hpg - 1), he = h + hpg < end ? h + hpg : end, taken = 0;
        for (; g < he && got < max; g += PG) {
            uint64_t k = g / PG, bit = 1ull << (k & 63);
            if (!(__atomic_load_n(&dirty[k >> 6], __ATOMIC_RELAXED) & bit)) continue;
            if (pstate[k] == PS_RW) pages[got++] = g;
            taken |= bit;   // a page that is not read-write any more only loses its bit
        }
        if (reset && taken) {
            watch_take();
            __atomic_fetch_and(&dirty[(h / PG) >> 6], ~taken, __ATOMIC_RELAXED);
            if (host_state(h) == PS_RW) mprotect((void *)(uintptr_t)h, hpg, PROT_READ);
            watch_give();
        }
    }
    plat_mutex_unlock(&vm_lock);
    return got;
}

// One handler for the synchronous faults. An access to a mapped page its protection forbids arrives as SIGBUS and one
// to an unmapped address as SIGSEGV. A write to a watched page is the write watch's own (watch_fault); otherwise both
// go to the access hook, and a write to the write-fault hook (memory.c thaws a frozen region). Any other fault, in this executable or in a library, then goes to the crash hook (main.c), which
// reports it and ends the process. Without a crash hook the fault gets the default action. Runs on the thread's
// signal stack.
static int (*fault_hook)(uint64_t addr);
static int (*access_hook)(uint64_t addr, int write, uint64_t ip);   // every read or write fault, before fault_hook
static void (*crash_fn)(const PlatCrash *c);

// The access that faulted, from the processor's fault record: 0 read, 1 write, 8 execute, -1 not a memory access.
static int fault_access(const ucontext_t *u) {
#if defined(__x86_64__)
    uint64_t err = u->uc_mcontext->__es.__err;   // the page-fault error code: bit 1 write, bit 4 instruction fetch
    return (err & 16) ? 8 : (err & 2) ? 1 : 0;
#else
    uint32_t esr = u->uc_mcontext->__es.__esr, ec = esr >> 26;   // the exception class; WnR is bit 6 of a data abort
    return ec == 0x20 || ec == 0x21 ? 8 : ec == 0x24 || ec == 0x25 ? (int)(esr >> 6 & 1) : -1;
#endif
}

static void fault_handler(int sig, siginfo_t *si, void *uc) {
    ucontext_t *u = uc;
    int mem = sig == SIGSEGV || sig == SIGBUS, access = mem ? fault_access(u) : -1;
    uint64_t addr = (uint64_t)(uintptr_t)si->si_addr;
    uintptr_t rip = (uintptr_t)UC_PC(u);
    if (mem && access == 1 && watch_fault(addr)) return;
    if (mem && access_hook && (access == 0 || access == 1) && access_hook(addr, access, rip)) return;
    if (mem && fault_hook && access == 1 && fault_hook(addr)) return;
    Dl_info at, self;
    if (crash_fn) {
        int found = dladdr((void *)rip, &at) && at.dli_fname;
        PlatCrash c = {sig == SIGSEGV ? "access violation" : sig == SIGBUS ? "access violation (protection)"
                       : sig == SIGILL ? "illegal instruction" : "arithmetic exception",
                       (uint32_t)sig, rip, found ? rip - (uintptr_t)at.dli_fbase : rip, -1, 0};
        if (!found) {
            c.module = "";
        } else if (!dladdr((void *)plat_crash_hook, &self) || at.dli_fbase != self.dli_fbase) {
            const char *n = strrchr(at.dli_fname, '/');
            c.module = n ? n + 1 : at.dli_fname;
        }
        if (mem && access >= 0) c.access = access, c.addr = addr;
        crash_fn(&c);
    }
    signal(sig, SIG_DFL);   // returns into the faulting instruction, which faults again with the default action
}

static void fault_install(void) {
    static int done;
    if (done) return;
    done = 1;
    alt_stack();
    struct sigaction sa = {.sa_sigaction = fault_handler, .sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER};
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
}

void plat_mem_fault_hook(int (*fn)(uint64_t addr)) {
    fault_hook = fn;
    fault_install();
}

void plat_mem_access_hook(int (*fn)(uint64_t addr, int write, uint64_t ip)) {
    access_hook = fn;
    fault_install();
}

void plat_crash_context(const char *s) { (void)s; }   // the Windows crash notes only
int plat_crash_filter_reassert(char *who, size_t n) { (void)who, (void)n; return 0; }   // Windows only

// plat_jmp_save / plat_jmp_go (platform.h): leaf code that does not move the stack pointer before it leaves.
#if defined(__x86_64__)
_Static_assert(sizeof(PlatJmp) == 72 && __builtin_offsetof(PlatJmp, mxcsr) == 64, "plat_jmp_save layout");
__asm__(".text\n"
        ASM_FN("plat_jmp_save")
        "    movq %rbx, 0(%rdi)\n"
        "    movq %rbp, 8(%rdi)\n"
        "    movq %r12, 16(%rdi)\n"
        "    movq %r13, 24(%rdi)\n"
        "    movq %r14, 32(%rdi)\n"
        "    movq %r15, 40(%rdi)\n"
        "    leaq 8(%rsp), %rdx\n"   // the caller's rsp after this call returns
        "    movq %rdx, 48(%rdi)\n"
        "    movq (%rsp), %rdx\n"
        "    movq %rdx, 56(%rdi)\n"
        "    stmxcsr 64(%rdi)\n"
        "    fnstcw 68(%rdi)\n"
        "    xorl %eax, %eax\n"
        "    ret\n"
        ASM_FN("plat_jmp_go")
        "    movq 0(%rdi), %rbx\n"
        "    movq 8(%rdi), %rbp\n"
        "    movq 16(%rdi), %r12\n"
        "    movq 24(%rdi), %r13\n"
        "    movq 32(%rdi), %r14\n"
        "    movq 40(%rdi), %r15\n"
        "    ldmxcsr 64(%rdi)\n"
        "    fldcw 68(%rdi)\n"
        "    movq 48(%rdi), %rsp\n"
        "    movl $1, %eax\n"
        "    jmpq *56(%rdi)\n");
#else
_Static_assert(sizeof(PlatJmp) == 176 && __builtin_offsetof(PlatJmp, fpcr) == 104 &&
               __builtin_offsetof(PlatJmp, d) == 112, "plat_jmp_save layout");
__asm__(".text\n"
        ASM_FN("plat_jmp_save")
        "    stp x19, x20, [x0, #0]\n"
        "    stp x21, x22, [x0, #16]\n"
        "    stp x23, x24, [x0, #32]\n"
        "    stp x25, x26, [x0, #48]\n"
        "    stp x27, x28, [x0, #64]\n"
        "    mov x9, sp\n"           // the caller's sp: a call moves nothing here
        "    stp x29, x9, [x0, #80]\n"
        "    mrs x9, fpcr\n"
        "    stp x30, x9, [x0, #96]\n"
        "    stp d8, d9, [x0, #112]\n"
        "    stp d10, d11, [x0, #128]\n"
        "    stp d12, d13, [x0, #144]\n"
        "    stp d14, d15, [x0, #160]\n"
        "    mov w0, #0\n"
        "    ret\n"
        ASM_FN("plat_jmp_go")
        "    ldp x19, x20, [x0, #0]\n"
        "    ldp x21, x22, [x0, #16]\n"
        "    ldp x23, x24, [x0, #32]\n"
        "    ldp x25, x26, [x0, #48]\n"
        "    ldp x27, x28, [x0, #64]\n"
        "    ldp x29, x9, [x0, #80]\n"
        "    mov sp, x9\n"
        "    ldp x30, x9, [x0, #96]\n"
        "    msr fpcr, x9\n"
        "    ldp d8, d9, [x0, #112]\n"
        "    ldp d10, d11, [x0, #128]\n"
        "    ldp d12, d13, [x0, #144]\n"
        "    ldp d14, d15, [x0, #160]\n"
        "    mov w0, #1\n"
        "    ret\n");
#endif

// The system unwinder, frame by frame from plat_frame_capture's caller. After a step the cursor is a caller's frame as
// it will be when the call it is making returns: the IP is the return address, the stack pointer is the one it then
// has, and the callee-saved registers are as that frame sees them.
__attribute__((noinline)) int plat_frame_capture(PlatJmp *out, int (*want)(uintptr_t rip, void *arg), void *arg) {
    unw_context_t uc;
    unw_cursor_t cur;
    if (unw_getcontext(&uc) || unw_init_local(&cur, &uc)) return 0;
    for (int n = 0; n < 256 && unw_step(&cur) > 0; n++) {
        unw_word_t ip = 0, sp = 0, v = 0;
        unw_get_reg(&cur, UNW_REG_IP, &ip);
        unw_get_reg(&cur, UNW_REG_SP, &sp);
        if (!ip) return 0;
        if (!want((uintptr_t)ip, arg)) continue;
#if defined(__x86_64__)
        static const int regs[6] = {UNW_X86_64_RBX, UNW_X86_64_RBP, UNW_X86_64_R12, UNW_X86_64_R13, UNW_X86_64_R14,
                                    UNW_X86_64_R15};
        for (int r = 0; r < 6; r++) unw_get_reg(&cur, regs[r], &v), out->gpr[r] = v;
        out->gpr[6] = sp, out->gpr[7] = ip;
        __asm__ volatile("stmxcsr %0\n\tfnstcw %1" : "=m"(out->mxcsr), "=m"(out->fpcw));
#else
        for (int r = 0; r < 10; r++) unw_get_reg(&cur, UNW_ARM64_X19 + r, &v), out->gpr[r] = v;
        unw_get_reg(&cur, UNW_ARM64_FP, &v);
        out->gpr[10] = v, out->gpr[11] = sp, out->gpr[12] = ip;
        __asm__ volatile("mrs %0, fpcr" : "=r"(out->fpcr));
        for (int r = 0; r < 8; r++) {
            unw_fpreg_t d = 0;
            unw_get_fpreg(&cur, UNW_ARM64_D8 + r, &d);
            out->d[r] = d;
        }
#endif
        return 1;
    }
    return 0;
}

void plat_crash_hook(void (*fn)(const PlatCrash *c)) {
    crash_fn = fn;
    fault_install();
}

void plat_reexec(void) {
    char exe[PATH_MAX];
    fflush(stdout);
    fflush(stderr);
    if (plat_exe_path(exe, sizeof exe)) execv(exe, *_NSGetArgv());
    fprintf(stderr, "FATAL: re-running the process failed (%s)\n", strerror(errno));
    _exit(70);
}

// The kernel copies and reports an unmapped or inaccessible page as an error.
int plat_mem_read(uint64_t addr, void *out, size_t n) {
    mach_vm_size_t got = 0;
    return mach_vm_read_overwrite(mach_task_self(), addr, n, (mach_vm_address_t)(uintptr_t)out, &got) ==
               KERN_SUCCESS && got == n;
}

int plat_mem_readable(uint64_t addr, uint64_t n) {
    uint8_t b;
    uint64_t end = addr + (n ? n : 1);
    for (uint64_t a = addr; a < end; a = (a | 4095) + 1)
        if (!plat_mem_read(a, &b, 1)) return 0;
    return 1;
}

void *plat_aligned_alloc(size_t size, size_t align) {
    void *p = NULL;
    return posix_memalign(&p, align < sizeof(void *) ? sizeof(void *) : align, size) ? NULL : p;
}

void plat_aligned_free(void *p) { free(p); }

// A PlatFile is the descriptor + 1, so descriptor 0 is not NULL.
PlatFile *plat_file_open(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    return fd < 0 ? NULL : (PlatFile *)(intptr_t)(fd + 1);
}

// F_NOCACHE keeps the reads out of the file cache; they need no alignment here.
PlatFile *plat_file_open_unbuffered(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd >= 0) fcntl(fd, F_NOCACHE, 1);
    return fd < 0 ? NULL : (PlatFile *)(intptr_t)(fd + 1);
}

static int fd_of(PlatFile *f) { return (int)(intptr_t)f - 1; }

void plat_file_close(PlatFile *f) {
    if (f) close(fd_of(f));
}

int64_t plat_file_pread(PlatFile *f, void *buf, uint32_t n, uint64_t off) {
    ssize_t k;
    watch_touch((uint64_t)(uintptr_t)buf, n);
    do k = pread(fd_of(f), buf, n, (off_t)off);
    while (k < 0 && errno == EINTR);
    return k >= 0 ? k : errno == EFAULT ? -2 : -1;
}

int plat_file_id(PlatFile *f, PlatFileId *out) {
    struct stat st;
    if (fstat(fd_of(f), &st)) return 0;
    *out = (PlatFileId){(uint64_t)st.st_dev, (uint64_t)st.st_ino, (uint64_t)st.st_size,
                        (uint64_t)st.st_mtimespec.tv_sec * 1000000000u + (uint64_t)st.st_mtimespec.tv_nsec,
                        (uint64_t)st.st_ctimespec.tv_sec * 1000000000u + (uint64_t)st.st_ctimespec.tv_nsec};
    return 1;
}

const void *plat_file_map(PlatFile *f) {
    struct stat st;
    if (fstat(fd_of(f), &st) || !st.st_size) return NULL;
    void *v = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd_of(f), 0);
    return v == MAP_FAILED ? NULL : v;
}

int plat_fs_os_error(void) { return errno; }

static int fs_err(int e) {
    switch (e) {
    case ENOENT: case ENOTDIR: return PLAT_FS_NOT_FOUND;
    case EEXIST: return PLAT_FS_EXISTS;
    case ENOTEMPTY: return PLAT_FS_NOT_EMPTY;
    case EBUSY: case ETXTBSY: return PLAT_FS_BUSY;
    default: return PLAT_FS_ERROR;
    }
}

int plat_fs_kind(const char *path) {
    struct stat st;
    if (lstat(path, &st)) return PLAT_FS_MISSING;
    return S_ISLNK(st.st_mode) ? PLAT_FS_LINK : S_ISDIR(st.st_mode) ? PLAT_FS_DIR : PLAT_FS_FILE;
}

int plat_fs_mkdir(const char *path) { return mkdir(path, 0755) ? fs_err(errno) : PLAT_FS_OK; }

int plat_fs_rmdir(const char *path) {
    struct stat st;
    if (!lstat(path, &st) && S_ISLNK(st.st_mode)) return unlink(path) ? fs_err(errno) : PLAT_FS_OK;
    return rmdir(path) ? fs_err(errno) : PLAT_FS_OK;
}

// Deleting needs write access to the directory, not to the file: a read-only file goes as any other.
int plat_fs_unlink(const char *path) { return unlink(path) ? fs_err(errno) : PLAT_FS_OK; }

// RENAME_EXCL keeps Windows' rule that a rename never replaces; a file system without it (ENOTSUP) gets a check
// first, which another process could race but the save directories are ours.
int plat_fs_rename(const char *from, const char *to, int durable) {
    if (renamex_np(from, to, RENAME_EXCL)) {
        if (errno != ENOTSUP && errno != EINVAL) return fs_err(errno);
        if (plat_fs_kind(to) != PLAT_FS_MISSING) return PLAT_FS_EXISTS;
        if (rename(from, to)) return fs_err(errno);
    }
    if (durable) {
        sync_dir_of(from);
        sync_dir_of(to);
    }
    return PLAT_FS_OK;
}

int plat_fs_link(const char *existing, const char *path) { return link(existing, path) ? fs_err(errno) : PLAT_FS_OK; }

int plat_fs_links(const char *path) {
    struct stat st;
    return stat(path, &st) ? 0 : (int)st.st_nlink;
}

void plat_fs_sync_dir(const char *path) {
    int fd = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
}

// Open files never stop a rename here.
void plat_fs_busy_reason(const char *dir, char *out, size_t cap) {
    (void)dir;
    if (cap) *out = 0;
}

int plat_fs_list(const char *dir, int (*cb)(const char *name, int kind, uint64_t size, void *ctx), void *ctx) {
    DIR *d = opendir(dir);
    if (!d) return fs_err(errno);
    for (struct dirent *e; (e = readdir(d));) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        struct stat st;
        if (fstatat(dirfd(d), e->d_name, &st, AT_SYMLINK_NOFOLLOW)) continue;
        int kind = S_ISLNK(st.st_mode) ? PLAT_FS_LINK : S_ISDIR(st.st_mode) ? PLAT_FS_DIR : PLAT_FS_FILE;
        if (!cb(e->d_name, kind, kind == PLAT_FS_FILE ? (uint64_t)st.st_size : 0, ctx)) break;
    }
    closedir(d);
    return PLAT_FS_OK;
}

PlatFile *plat_fs_open(const char *path, int write, int *err) {
    struct stat st;
    int fd = open(path, (write ? O_RDWR : O_RDONLY) | O_CLOEXEC);
    if (fd >= 0 && (fstat(fd, &st) || !S_ISREG(st.st_mode))) {
        close(fd);
        fd = -1, errno = ENOENT;
    }
    if (fd < 0 && err) *err = fs_err(errno);
    return fd < 0 ? NULL : (PlatFile *)(intptr_t)(fd + 1);
}

PlatFile *plat_fs_create(const char *path, int *err) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0 && err) *err = fs_err(errno);
    return fd < 0 ? NULL : (PlatFile *)(intptr_t)(fd + 1);
}

int64_t plat_file_pwrite(PlatFile *f, const void *buf, uint32_t n, uint64_t off) {
    ssize_t k;
    do k = pwrite(fd_of(f), buf, n, (off_t)off);
    while (k < 0 && errno == EINTR);
    return k;
}

int plat_file_set_size(PlatFile *f, uint64_t size) { return ftruncate(fd_of(f), (off_t)size) ? fs_err(errno) : 0; }

int plat_file_fsync(PlatFile *f) { return fsync(fd_of(f)) == 0; }

int plat_temp_dir(char *out, size_t cap) {
    const char *t = getenv("TMPDIR");
    if (!t || !*t) t = "/tmp";
    size_t n = strlen(t);
    while (n > 1 && t[n - 1] == '/') n--;
    return snprintf(out, cap, "%.*s", (int)n, t) < (int)cap;
}

uintptr_t plat_exe_base(void) {
    Dl_info d;
    return dladdr((void *)plat_exe_base, &d) ? (uintptr_t)d.dli_fbase : 0;
}

int plat_module_of(uintptr_t addr, char *name, size_t cap, uintptr_t *base) {
    Dl_info d;
    if (!dladdr((void *)addr, &d) || !d.dli_fname) return 0;
    const char *n = strrchr(d.dli_fname, '/');
    *base = (uintptr_t)d.dli_fbase;
    return snprintf(name, cap, "%s", n ? n + 1 : d.dli_fname) < (int)cap;
}

int plat_exe_path(char *out, size_t cap) {
    char raw[PATH_MAX], real[PATH_MAX];
    uint32_t n = sizeof raw;
    if (_NSGetExecutablePath(raw, &n) || !realpath(raw, real)) return 0;
    return snprintf(out, cap, "%s", real) < (int)cap;
}

size_t plat_getenv(const char *name, char *out, size_t cap) {
    const char *v = getenv(name);
    size_t n = v ? strlen(v) : 0;
    if (!n || n >= cap) return 0;
    memcpy(out, v, n + 1);
    return n;
}

// The thread's Mach port name, cached per thread: plat_rmutex_lock asks on every acquire, hundreds of thousands of
// times a second in play. noinline: a fiber may resume on another thread (platform.h), and a caller that inlined this
// could keep the address of tid it computed before a plat_fiber_switch; as a call it reads the slot of the thread it
// runs on.
__attribute__((noinline)) uint32_t plat_thread_id(void) {
    static __thread uint32_t tid;
    if (!tid) tid = (uint32_t)pthread_mach_thread_np(pthread_self());
    return tid;
}

// A CoreFoundation string ("es_ES", "zh-Hans-CN", "en_US@rg=eszzzz") as a BCP-47 tag into tag; "" when s is NULL.
static void cf_tag(CFStringRef s, char *tag, size_t cap) {
    if (!s || !CFStringGetCString(s, tag, (CFIndex)cap, kCFStringEncodingUTF8)) tag[0] = 0;
    for (char *p = tag; *p; p++) {
        if (*p == '_') *p = '-';
        if (*p == '@') *p = 0;
    }
}

const char *plat_ui_language(void) {
    static char tag[32];
    static int done;
    if (!done) {
        CFArrayRef a = CFLocaleCopyPreferredLanguages();
        cf_tag(a && CFArrayGetCount(a) ? CFArrayGetValueAtIndex(a, 0) : NULL, tag, sizeof tag);
        if (a) CFRelease(a);
        done = 1;
    }
    return tag;
}

const char *plat_user_locale(void) {
    static char tag[32];
    static int done;
    if (!done) {
        CFLocaleRef l = CFLocaleCopyCurrent();
        cf_tag(l ? CFLocaleGetIdentifier(l) : NULL, tag, sizeof tag);
        if (l) CFRelease(l);
        done = 1;
    }
    return tag;
}

// The system's own CJK fonts; which of them a release installs has changed over the years, so each language has
// several to try.
int plat_font_path(const char *lang, char *out, size_t cap) {
    static const char *const ja[] = {"ヒラギノ角ゴシック W3.ttc", "Hiragino Sans GB.ttc", NULL};
    static const char *const ko[] = {"AppleSDGothicNeo.ttc", "Supplemental/AppleGothic.ttf", NULL};
    static const char *const hans[] = {"PingFang.ttc", "STHeiti Light.ttc", "Hiragino Sans GB.ttc", NULL};
    static const char *const hant[] = {"PingFang.ttc", "STHeiti Medium.ttc", "STHeiti Light.ttc", NULL};
    const char *const *files = !strcmp(lang, "ja") ? ja : !strcmp(lang, "ko") ? ko : !strcmp(lang, "zh-Hans") ? hans
                             : !strcmp(lang, "zh-Hant") ? hant : NULL;
    for (; files && *files; files++) {
        char path[PATH_MAX];
        snprintf(path, sizeof path, "/System/Library/Fonts/%s", *files);
        if (plat_file_exists(path)) return snprintf(out, cap, "%s", path) < (int)cap;
    }
    return 0;
}

// "macOS 15.5 (Darwin 24.5.0)".
int plat_os_version(char *out, size_t cap) {
    char ver[64] = "", rel[64] = "";
    size_t n = sizeof ver;
    sysctlbyname("kern.osproductversion", ver, &n, NULL, 0);
    n = sizeof rel;
    sysctlbyname("kern.osrelease", rel, &n, NULL, 0);
    if (!ver[0]) return 0;
    return snprintf(out, cap, "macOS %s (Darwin %s)", ver, rel) < (int)cap;
}

int plat_is_wine(void) { return 0; }

int plat_stderr_path(char *out, size_t cap) {
    struct stat st;
    char path[PATH_MAX];
    if (fstat(2, &st) || !S_ISREG(st.st_mode) || fcntl(2, F_GETPATH, path) == -1) return 0;
    return snprintf(out, cap, "%s", path) < (int)cap;
}

// ---- process

#ifdef NX2NAVI_PGO_GEN
// compiler-rt's profile runtime (an instrumented build, tools/build_game.py --pgo gen). It writes its counters from an
// atexit handler, which _exit skips; __llvm_profile_dump writes them once and makes later writes no-ops.
int __llvm_profile_dump(void);
#endif

void (*plat_exit_hook)(void);

void plat_exit(int code) {
    if (plat_exit_hook) plat_exit_hook();
#ifdef NX2NAVI_PGO_GEN
    __llvm_profile_dump();
#endif
    fflush(stdout);
    fflush(stderr);
    _exit(code);
}

int plat_setenv(const char *name, const char *value) {
    return (value ? setenv(name, value, 1) : unsetenv(name)) == 0;
}

uint32_t plat_process_id(void) { return (uint32_t)getpid(); }

uint64_t plat_process_cpu_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t)) return 0;
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

// The start time of process pid in microseconds since 1970, 0 when it is gone.
static uint64_t process_start(int pid) {
    struct proc_bsdinfo bi;
    if (proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &bi, sizeof bi) != (int)sizeof bi) return 0;
    return bi.pbi_start_tvsec * 1000000u + bi.pbi_start_tvusec;
}

uint64_t plat_process_uptime_ns(void) {
    uint64_t at = process_start((int)getpid()) * 1000u, now = plat_unix_time_ns();
    return at && now > at ? now - at : 0;
}

uint64_t plat_unix_time_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

void plat_local_time(struct tm *out) {
    time_t t = time(NULL);
    localtime_r(&t, out);
}

// The range of the Windows CRT's localtime_s and gmtime_s (1970 to 3000), which the C library here exceeds: the game's time
// service reports times outside it as invalid on both hosts.
#define PLAT_TIME_MAX 32535215999ll

int plat_localtime(int64_t unix_s, struct tm *out) {
    time_t t = (time_t)unix_s;
    return unix_s >= 0 && unix_s <= PLAT_TIME_MAX && localtime_r(&t, out) != NULL;
}

int plat_gmtime(int64_t unix_s, struct tm *out) {
    time_t t = (time_t)unix_s;
    return unix_s >= 0 && unix_s <= PLAT_TIME_MAX && gmtime_r(&t, out) != NULL;
}

long plat_tz_west(void) {
    tzset();
    return timezone;
}

int plat_run_self(const char *env_name, const char *env_value, char *out, size_t cap, uint32_t timeout_ms) {
    char **argv = *_NSGetArgv();
    int argc = *_NSGetArgc();
    char exe[PATH_MAX];
    if (!argc || !plat_exe_path(exe, sizeof exe)) return -1;
    int p[2];
    if (pipe_cloexec(p)) return -1;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, p[1], 1);
    posix_spawn_file_actions_adddup2(&fa, p[1], 2);
    setenv(env_name, env_value, 1);
    pid_t pid;
    int e = posix_spawn(&pid, exe, &fa, NULL, argv, environ);
    unsetenv(env_name);
    posix_spawn_file_actions_destroy(&fa);
    close(p[1]);
    if (e) {
        close(p[0]);
        return -1;
    }
    size_t len = 0;
    char sink[512];
    ssize_t k;
    while (len + 1 < cap && (k = read(p[0], out + len, cap - 1 - len)) != 0)
        if (k > 0) len += (size_t)k;
        else if (errno != EINTR) break;
    if (cap) out[len] = 0;
    while ((k = read(p[0], sink, sizeof sink)) > 0 || (k < 0 && errno == EINTR)) {}
    close(p[0]);
    int status = 0;
    uint64_t end = plat_now_ns() + (uint64_t)timeout_ms * 1000000u;
    while (waitpid(pid, &status, timeout_ms ? WNOHANG : 0) == 0) {
        if (plat_now_ns() >= end) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            break;
        }
        plat_sleep_ms(10);
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
}

int plat_random(void *out, size_t n) {
    arc4random_buf(out, n);
    return 1;
}

// ---- desktop

// pbcopy puts what it reads on the clipboard.
int plat_clipboard_set(const char *utf8) {
    int p[2];
    if (pipe_cloexec(p)) return 0;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, p[0], 0);
    char *argv[] = {"pbcopy", NULL};
    pid_t pid;
    int e = posix_spawn(&pid, "/usr/bin/pbcopy", &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(p[0]);
    int ok = !e && write_all(p[1], utf8, strlen(utf8));
    close(p[1]);
    int status = 0;
    if (!e) waitpid(pid, &status, 0);
    return ok && WIFEXITED(status) && !WEXITSTATUS(status);
}

// pbpaste writes the clipboard's text. Any thread: no window system call is made here.
int plat_clipboard_get(char *out, size_t cap) {
    if (!cap) return 0;
    out[0] = 0;
    int p[2];
    if (pipe_cloexec(p)) return 0;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, p[1], 1);
    char *argv[] = {"pbpaste", NULL};
    pid_t pid;
    int e = posix_spawn(&pid, "/usr/bin/pbpaste", &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(p[1]);
    size_t n = 0;
    int more = 0;   // the text is longer than out holds
    while (!e) {
        char rest[256];
        ssize_t k = n < cap - 1 ? read(p[0], out + n, cap - 1 - n) : read(p[0], rest, sizeof rest);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) break;
        if (n < cap - 1) n += (size_t)k;
        else more = 1;
    }
    close(p[0]);
    int status = 0;
    if (!e) waitpid(pid, &status, 0);
    if (e || !WIFEXITED(status) || WEXITSTATUS(status)) n = 0;
    if (more)   // cut where a character starts, not inside one
        while (n && ((unsigned char)out[n] & 0xc0) == 0x80) n--;
    out[n] = 0;
    return n != 0;
}

static void reap_main(void *arg) {
    waitpid((pid_t)(intptr_t)arg, NULL, 0);
}

// open shows the folder in the Finder; it runs detached, so a slow answer does not hold up the caller, and a thread
// of its own reaps it so it does not stay a zombie.
int plat_open_folder(const char *path) {
    struct stat st;
    if (stat(path, &st) || !S_ISDIR(st.st_mode)) return 0;
    posix_spawnattr_t a;
    posix_spawnattr_init(&a);
    posix_spawnattr_setflags(&a, POSIX_SPAWN_SETSID);
    char *argv[] = {"open", (char *)path, NULL};
    pid_t pid;
    int e = posix_spawn(&pid, "/usr/bin/open", NULL, &a, argv, environ);
    posix_spawnattr_destroy(&a);
    if (e) return 0;
    plat_thread_spawn(reap_main, (void *)(intptr_t)pid, "reap", 64 << 10);
    return 1;
}

// ---- dynamic libraries

void *plat_lib_open(const char *name) { return dlopen(name, RTLD_NOW | RTLD_LOCAL); }
void *plat_lib_sym(void *lib, const char *name) { return lib ? dlsym(lib, name) : NULL; }

// ---- machine load: the Mach host's processor ticks and page counts, the task's own counters. The system gives no
// temperature and no per-adapter GPU counters without private interfaces: those stay unknown.

struct PlatLoad {
    uint64_t busy, total;   // the previous call's ticks; total 0 before the first
    uint64_t proc_ns, at_ns;
};

static mach_port_t host_port(void) {
    static mach_port_t h;
    if (!h) h = mach_host_self();
    return h;
}

PlatLoad *plat_load_open(const uint8_t *gpu_luid, int pci_bus) {
    (void)gpu_luid, (void)pci_bus;
    return calloc(1, sizeof(PlatLoad));
}

void plat_load_close(PlatLoad *l) { free(l); }

void plat_cpu_load(PlatLoad *l, float *total_pct, float *proc_pct, float *temp_c) {
    *total_pct = *proc_pct = *temp_c = -1;
    if (!l) return;
    host_cpu_load_info_data_t ci;
    mach_msg_type_number_t n = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(host_port(), HOST_CPU_LOAD_INFO, (host_info_t)&ci, &n) != KERN_SUCCESS) return;
    uint64_t total = 0;
    for (int i = 0; i < CPU_STATE_MAX; i++) total += ci.cpu_ticks[i];
    uint64_t busy = total - ci.cpu_ticks[CPU_STATE_IDLE], pns = plat_process_cpu_ns(), now = plat_now_ns();
    if (l->total && total > l->total && now > l->at_ns) {
        *total_pct = 100.0f * (float)(busy - l->busy) / (float)(total - l->total);
        // the process time against the wall time of every logical processor
        *proc_pct = (float)(100.0 * (double)(pns - l->proc_ns) / ((double)(now - l->at_ns) * plat_cpu_count()));
    }
    l->busy = busy, l->total = total, l->proc_ns = pns, l->at_ns = now;
}

void plat_gpu_load(PlatLoad *l, float *load_pct, float *vram_used_mb) {
    (void)l;
    *load_pct = *vram_used_mb = -1;
}

void plat_process_memory(uint64_t *resident, uint64_t *private_bytes) {
    *resident = *private_bytes = 0;
    task_vm_info_data_t vi;
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &n) != KERN_SUCCESS) return;
    *resident = vi.resident_size;
    *private_bytes = vi.phys_footprint;   // what Activity Monitor shows as the process's memory
}

// In use: the active, wired and compressed pages, as Activity Monitor counts "Memory Used".
void plat_mem_info(float *used_mb, float *total_mb, float *proc_mb) {
    *used_mb = *total_mb = *proc_mb = -1;
    uint64_t total = 0, resident, private_bytes;
    size_t len = sizeof total;
    if (!sysctlbyname("hw.memsize", &total, &len, NULL, 0)) *total_mb = (float)(total >> 20);
    vm_statistics64_data_t vs;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host_port(), HOST_VM_INFO64, (host_info64_t)&vs, &n) == KERN_SUCCESS)
        *used_mb = (float)(((uint64_t)vs.active_count + vs.wire_count + vs.compressor_page_count) *
                           (uint64_t)vm_kernel_page_size >> 20);
    plat_process_memory(&resident, &private_bytes);
    if (resident) *proc_mb = (float)(resident >> 20);
}

// The system has no commit limit: memory is compressed and swapped as needed.
void plat_commit_info(double *proc_mb, double *used_mb, double *limit_mb) {
    *proc_mb = *used_mb = *limit_mb = -1;
    uint64_t resident, private_bytes;
    plat_process_memory(&resident, &private_bytes);
    if (private_bytes) *proc_mb = (double)(private_bytes >> 20);
}

int plat_commit_enforced(void) { return 0; }
int plat_commit_note(char *out, size_t cap) { if (cap) out[0] = 0; return 0; }
int plat_commit_counts_vram(void) { return 0; }
int plat_page_file_managed(void) { return -1; }   // no page file Windows sizes

// ---- child processes

struct PlatChild {
    pid_t pid;
    int in, out;   // in: our write end, out: our read end
};

PlatChild *plat_child_start(const char *exe, const char *arg, uint32_t *err) {
    if (err) *err = 0;
    int in[2], out[2];
    if (pipe_cloexec(in)) return NULL;
    if (pipe_cloexec(out)) {
        close(in[0]), close(in[1]);
        return NULL;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, in[0], 0);
    posix_spawn_file_actions_adddup2(&fa, out[1], 1);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    char *argv[] = {(char *)exe, (char *)arg, NULL};
    pid_t pid;
    int e = posix_spawn(&pid, exe, &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(in[0]);
    close(out[1]);
    if (e) {
        close(in[1]), close(out[0]);
        if (err) *err = (uint32_t)e;
        return NULL;
    }
    PlatChild *c = malloc(sizeof *c);
    if (!c) {
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        close(in[1]), close(out[0]);
        return NULL;
    }
    c->pid = pid, c->in = in[1], c->out = out[0];
    return c;
}

void plat_child_stop(PlatChild *c) {
    if (!c) return;
    close(c->in);
    close(c->out);
    kill(c->pid, SIGKILL);
    waitpid(c->pid, NULL, 0);
    free(c);
}

void plat_child_lower(PlatChild *c) {   // a higher nice value needs no privilege
    if (c) setpriority(PRIO_PROCESS, (id_t)c->pid, 19);
}

// A child that died makes the write fail with EPIPE; SIGPIPE is ignored for the process (main.c), as Windows has none.
int plat_child_write(PlatChild *c, const void *data, size_t n) { return write_all(c->in, data, n); }

// See win32.c plat_child_read: a child that never answers fails the read after PLAT_CHILD_READ_MS.
long plat_child_read(PlatChild *c, void *buf, size_t cap) {
    struct pollfd pf = {.fd = c->out, .events = POLLIN};
    int r;
    do r = poll(&pf, 1, PLAT_CHILD_READ_MS);
    while (r < 0 && errno == EINTR);
    if (r <= 0) return -1;   // timed out or failed
    ssize_t k;
    do k = read(c->out, buf, cap);
    while (k < 0 && errno == EINTR);
    return (long)k;
}

// ---- local IPC (platform.h): Discord's Rich Presence Unix domain socket
struct PlatIpc { int fd; };

PlatIpc *plat_ipc_connect(const char *name) {
    const char *dirs[] = {getenv("DISCORD_IPC_PATH"), getenv("XDG_RUNTIME_DIR"), getenv("TMPDIR"), "/tmp"};
    for (size_t d = 0; d < sizeof dirs / sizeof *dirs; d++) {
        if (!dirs[d] || !*dirs[d]) continue;
        char path[sizeof ((struct sockaddr_un *)0)->sun_path];
        if ((size_t)snprintf(path, sizeof path, "%s/%s", dirs[d], name) >= sizeof path) continue;
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) return NULL;
        fcntl(fd, F_SETFD, FD_CLOEXEC);
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &(int){1}, sizeof(int));   // a closed peer is an error
        struct sockaddr_un sa = {0};
        sa.sun_family = AF_UNIX;
        strcpy(sa.sun_path, path);
        if (connect(fd, (struct sockaddr *)&sa, sizeof sa) == 0) {
            PlatIpc *p = malloc(sizeof *p);
            if (!p) {
                close(fd);
                return NULL;
            }
            p->fd = fd;
            return p;
        }
        close(fd);   // nothing listens at this path: the next directory may have it
    }
    return NULL;
}

int plat_ipc_read(PlatIpc *p, void *buf, size_t n, uint32_t ms) {
    struct pollfd pf = {.fd = p->fd, .events = POLLIN};
    int r = poll(&pf, 1, (int)ms);
    if (r <= 0) return r < 0 && errno != EINTR ? -1 : 0;
    ssize_t got = recv(p->fd, buf, n, 0);
    return got > 0 ? (int)got : -1;   // 0 bytes: the peer closed the socket
}

int plat_ipc_write(PlatIpc *p, const void *buf, size_t n) {
    const uint8_t *at = buf;
    while (n) {
        ssize_t put = send(p->fd, at, n, 0);
        if (put <= 0) {
            if (put < 0 && errno == EINTR) continue;
            return 0;
        }
        at += put, n -= put;
    }
    return 1;
}

void plat_ipc_close(PlatIpc *p) {
    if (!p) return;
    close(p->fd);
    free(p);
}

// ---- UDP sockets (platform.h): the co-op layer's transport
struct PlatUdp { int fd, v6; };

static void addr_from_sockaddr(PlatAddr *out, const struct sockaddr_storage *ss) {
    uint8_t ip[16] = {0};
    if (ss->ss_family == AF_INET6) {
        const struct sockaddr_in6 *a = (const struct sockaddr_in6 *)ss;
        memcpy(ip, &a->sin6_addr, 16);
        plat_addr_set(out, ip, ntohs(a->sin6_port), a->sin6_scope_id);
    } else {
        const struct sockaddr_in *a = (const struct sockaddr_in *)ss;
        ip[10] = ip[11] = 0xff;
        memcpy(ip + 12, &a->sin_addr, 4);
        plat_addr_set(out, ip, ntohs(a->sin_port), 0);
    }
}

// The socket address of a for a socket of family v6 (1: AF_INET6, 0: AF_INET); 0 when an IPv4 socket cannot reach it.
static int addr_to_sockaddr(const PlatAddr *a, int v6, struct sockaddr_storage *ss, socklen_t *len) {
    uint8_t ip[16];
    uint32_t scope;
    plat_addr_get(a, ip, &scope);
    memset(ss, 0, sizeof *ss);
    if (v6) {
        struct sockaddr_in6 *s = (struct sockaddr_in6 *)ss;
        s->sin6_len = sizeof *s;
        s->sin6_family = AF_INET6;
        s->sin6_port = htons(plat_addr_port(a));
        s->sin6_scope_id = scope;
        memcpy(&s->sin6_addr, ip, 16);
        *len = sizeof *s;
        return 1;
    }
    static const uint8_t mapped[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    if (memcmp(ip, mapped, sizeof mapped)) return 0;
    struct sockaddr_in *s = (struct sockaddr_in *)ss;
    s->sin_len = sizeof *s;
    s->sin_family = AF_INET;
    s->sin_port = htons(plat_addr_port(a));
    memcpy(&s->sin_addr, ip + 12, 4);
    *len = sizeof *s;
    return 1;
}

int plat_net_resolve(const char *host, uint16_t port, PlatAddr *out) {
    struct addrinfo hints = {.ai_family = AF_UNSPEC, .ai_socktype = SOCK_DGRAM}, *res = NULL, *pick = NULL;
    if (!host || !*host || getaddrinfo(host, NULL, &hints, &res) != 0) return 0;
    for (struct addrinfo *r = res; r; r = r->ai_next) {
        if (r->ai_addrlen > sizeof(struct sockaddr_storage) || (r->ai_family != AF_INET && r->ai_family != AF_INET6)) continue;
        if (!pick || (r->ai_family == AF_INET && pick->ai_family != AF_INET)) pick = r;   // IPv4 first
    }
    int ok = pick != NULL;
    if (ok) {
        struct sockaddr_storage ss;
        memset(&ss, 0, sizeof ss);
        memcpy(&ss, pick->ai_addr, pick->ai_addrlen);
        addr_from_sockaddr(out, &ss);
        uint8_t ip[16];
        uint32_t scope;
        plat_addr_get(out, ip, &scope);
        plat_addr_set(out, ip, port, scope);
    }
    freeaddrinfo(res);
    return ok;
}

// A datagram socket of the family, non-blocking and closed on exec. The system has no MSG_NOSIGNAL: SO_NOSIGPIPE
// keeps a send from raising SIGPIPE.
static int udp_socket(int family) {
    int fd = socket(family, SOCK_DGRAM, 0);
    if (fd < 0) return -1;
    int on = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on);
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) || fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK)) {
        close(fd);
        return -1;
    }
    return fd;
}

PlatUdp *plat_udp_open(uint16_t port, int dual_stack) {
    int v6 = dual_stack;
    int fd = v6 ? udp_socket(AF_INET6) : -1;
    if (fd >= 0) {
        int off = 0;
        setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof off);
        struct sockaddr_in6 a = {.sin6_len = sizeof a, .sin6_family = AF_INET6, .sin6_addr = in6addr_any,
                                 .sin6_port = htons(port)};
        if (bind(fd, (struct sockaddr *)&a, sizeof a) != 0) close(fd), fd = -1;
    }
    if (fd < 0) {   // IPv4 only, asked for or all the machine has
        v6 = 0;
        fd = udp_socket(AF_INET);
        if (fd < 0) return NULL;
        struct sockaddr_in a = {.sin_len = sizeof a, .sin_family = AF_INET, .sin_port = htons(port)};
        a.sin_addr.s_addr = htonl(INADDR_ANY);
        if (bind(fd, (struct sockaddr *)&a, sizeof a) != 0) {
            close(fd);
            return NULL;
        }
    }
    PlatUdp *u = malloc(sizeof *u);
    if (!u) {
        close(fd);
        return NULL;
    }
    u->fd = fd, u->v6 = v6;
    return u;
}

uint16_t plat_udp_port(const PlatUdp *u) {
    struct sockaddr_storage ss;
    socklen_t len = sizeof ss;
    if (!u || getsockname(u->fd, (struct sockaddr *)&ss, &len) != 0) return 0;
    PlatAddr a;
    addr_from_sockaddr(&a, &ss);
    return plat_addr_port(&a);
}

int plat_udp_send(PlatUdp *u, const PlatAddr *to, const void *p, size_t n) {
    struct sockaddr_storage ss;
    socklen_t len;
    if (!u || !addr_to_sockaddr(to, u->v6, &ss, &len)) return 0;
    for (;;) {
        ssize_t k = sendto(u->fd, p, n, 0, (struct sockaddr *)&ss, len);
        if (k >= 0) return 1;
        if (errno == EINTR) continue;
        // a full buffer, a missing route or an ICMP error is a lost datagram; a datagram too long or a socket that is not
        // ours to use can never be sent
        return errno != EMSGSIZE && errno != EBADF && errno != EINVAL && errno != EAFNOSUPPORT && errno != ENOTSOCK;
    }
}

int plat_udp_recv(PlatUdp *u, void *buf, size_t cap, PlatAddr *from, uint32_t ms) {
    if (!u) return -1;
    struct pollfd pf = {.fd = u->fd, .events = POLLIN};
    int r = poll(&pf, 1, ms > (uint32_t)INT32_MAX ? INT32_MAX : (int)ms);
    if (r < 0) return errno == EINTR ? 0 : -1;
    if (r == 0) return 0;
    struct sockaddr_storage ss;
    socklen_t len = sizeof ss;
    ssize_t k = recvfrom(u->fd, buf, cap, MSG_DONTWAIT, (struct sockaddr *)&ss, &len);
    if (k < 0) return errno == EAGAIN || errno == EINTR || errno == ECONNREFUSED || errno == EHOSTUNREACH ? 0 : -1;
    if (from) addr_from_sockaddr(from, &ss);
    return (int)k;
}

void plat_udp_close(PlatUdp *u) {
    if (!u) return;
    close(u->fd);
    free(u);
}

// ---- restart
// NX2NAVI_RESTART_WAIT names the process that called plat_restart, as its id and start time: a process that has only
// reused the id is not waited for. Waiting keeps the new process off the old one's files (the log it moves at start,
// the saves) until the old one is gone. The new process gets no descriptor past stderr, and the default signal mask
// and dispositions, whatever the calling thread had.
#define RESTART_WAIT "NX2NAVI_RESTART_WAIT"

// The child keeps descriptors 0 to 2 and no other.
static void spawn_std_only(posix_spawn_file_actions_t *fa, posix_spawnattr_t *at, short flags) {
    for (int fd = 0; fd < 3; fd++) posix_spawn_file_actions_addinherit_np(fa, fd);
    sigset_t none, all;
    sigemptyset(&none);
    sigfillset(&all);
    posix_spawnattr_setsigmask(at, &none);
    posix_spawnattr_setsigdefault(at, &all);
    posix_spawnattr_setflags(at, (short)(POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_CLOEXEC_DEFAULT |
                                         flags));
}

int plat_restart(void) {
    char exe[PATH_MAX], wait[64];
    if (!*_NSGetArgc() || !plat_exe_path(exe, sizeof exe)) return 0;
    snprintf(wait, sizeof wait, "%d.%llu", (int)getpid(), (unsigned long long)process_start((int)getpid()));
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawnattr_t at;
    posix_spawnattr_init(&at);
    spawn_std_only(&fa, &at, 0);
    setenv(RESTART_WAIT, wait, 1);
    pid_t pid;
    int e = posix_spawn(&pid, exe, &fa, &at, *_NSGetArgv(), environ);
    unsetenv(RESTART_WAIT);
    posix_spawnattr_destroy(&at);
    posix_spawn_file_actions_destroy(&fa);
    return e == 0;
}

int plat_spawn(const char *const *argv, const char *cwd) {
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"   // the name without _np exists from macOS 26 only
    posix_spawn_file_actions_addchdir_np(&fa, cwd);
#pragma clang diagnostic pop
    posix_spawnattr_t at;
    posix_spawnattr_init(&at);
    spawn_std_only(&fa, &at, POSIX_SPAWN_SETSID);
    pid_t pid;
    int e = posix_spawn(&pid, argv[0], &fa, &at, (char *const *)argv, environ);
    posix_spawnattr_destroy(&at);
    posix_spawn_file_actions_destroy(&fa);
    return e == 0;
}

int plat_restart_wait(uint32_t timeout_ms) {
    const char *v = getenv(RESTART_WAIT);
    if (!v) return 0;
    int pid = 0;
    unsigned long long start = 0;
    int ok = sscanf(v, "%d.%llu", &pid, &start) == 2 && pid > 0 && start;
    unsetenv(RESTART_WAIT);   // the processes this one starts do not wait again
    if (!ok) return 0;
    // The exit event is registered first: a start time that still matches then proves it is that process, not a new
    // one with the id.
    int kq = kqueue();
    if (kq < 0) return 0;
    struct kevent ev, got;
    EV_SET(&ev, (uintptr_t)pid, EVFILT_PROC, EV_ADD | EV_ONESHOT, NOTE_EXIT, 0, NULL);
    int r = 0;
    if (!kevent(kq, &ev, 1, NULL, 0, NULL) && process_start(pid) == start) {   // ESRCH: already gone
        struct timespec ts = {(time_t)(timeout_ms / 1000), (long)(timeout_ms % 1000) * 1000000};
        do r = kevent(kq, NULL, 0, &got, 1, &ts);
        while (r < 0 && errno == EINTR);
        r = r > 0 ? 1 : -1;
    }
    close(kq);
    return r;
}
