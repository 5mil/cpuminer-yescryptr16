/*
 * yescrypt-best.c — Universal CPU dispatch
 *
 * Device coverage:
 *   AVX-512   : Intel Xeon Scalable, Core i9-11900+, AMD EPYC 7003+ (servers, workstations)
 *   AVX2      : Intel Haswell (2013)+, AMD Ryzen all generations (desktops, laptops, servers, mini-PCs)
 *   SSE2/XOP  : Intel Sandy Bridge / Ivy Bridge, AMD Bulldozer/Piledriver (old PCs, old laptops)
 *   NEON      : ARM Cortex-A (Android phones, Raspberry Pi, Apple M-series under Linux)
 *   scalar    : Everything else (old 32-bit, MIPS, RISC-V, embedded)
 */
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#  include "yescrypt-neon.c"
#elif defined(__AVX2__)
#  include "yescrypt-avx2.c"
#elif defined(__SSE2__)
#  include "yescrypt-sse.c"
#else
#  include "yescrypt-opt.c"
#endif
