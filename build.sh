#!/bin/bash
# Optimized build script — detects CPU tier and compiles accordingly

./autogen.sh || echo "autogen done"

# Detect best SIMD tier available
if grep -q avx512 /proc/cpuinfo 2>/dev/null; then
    SIMD_FLAGS="-mavx512f -mavx512bw -mavx2 -maes -msse4.2"
    echo "Build tier: AVX-512"
elif grep -q avx2 /proc/cpuinfo 2>/dev/null; then
    SIMD_FLAGS="-mavx2 -maes -msse4.2"
    echo "Build tier: AVX2"
elif grep -q sse4_1 /proc/cpuinfo 2>/dev/null; then
    SIMD_FLAGS="-msse4.2 -msse4.1"
    echo "Build tier: SSE4"
elif grep -q sse2 /proc/cpuinfo 2>/dev/null; then
    SIMD_FLAGS="-msse2"
    echo "Build tier: SSE2"
else
    SIMD_FLAGS=""
    echo "Build tier: scalar"
fi

CFLAGS="-O3 -march=native $SIMD_FLAGS \
  -funroll-loops \
  -fomit-frame-pointer \
  -falign-functions=16 \
  -falign-loops=16 \
  -fno-strict-aliasing \
  -Wall" \
  ./configure --with-curl --enable-assembly

make -j$(nproc)

if [ -f minerd ]; then
    strip -s minerd
    echo "Build complete: $(./minerd --version 2>&1 | head -1)"
fi
