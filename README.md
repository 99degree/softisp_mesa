# Mesa Offline Compute Test

Tests linking to **Mesa 26.0.6** and measures its **offline GPU computation power** via Vulkan compute shaders.

## Why Vulkan instead of GLES/EGL?

On this device (Termux on aarch64), the GLES/EGL path defaults to the **llvmpipe** software renderer (CPU-based, no X display). However, **Vulkan** gives direct access to the hardware GPU:

- **Adreno 7c+ Gen 3** (freedreno Vulkan ICD) — hardware GPU compute
- **llvmpipe** — software fallback

The test uses the Vulkan surfaceless platform, which works without any display server — true *offline* computation.

## How It Works

Two tests are performed:

### Test 1: Vector Addition (Correctness)
Dispaches 1024 work items doing `c[i] = a[i] + b[i]`. Verifies all results match expected values (3× loop unrolling). Ensures the compute pipeline, SSBOs, and dispatch are working correctly.

### Test 2: Heavy Compute Benchmark (Performance)
Dispaches a large grid (4M work items on GPU, 64K on CPU), each running 1024 iterations of heavy FP math (sin, cos, sqrt, multiplication, addition). Reports:

- Execution time per run
- Million items processed per second
- **Estimated GFLOPS** (based on ~5 FP ops per iteration)

## Results (Adreno 7c+ Gen 3)

| Metric | Value |
|--------|-------|
| Work items | 4,194,304 |
| Inner iterations | 1,024 |
| Total FP ops (est.) | ~20 billion |
| Avg time | 0.44 s |
| Throughput | 9.50 M items/s |
| **Estimated GFLOPS** | **48.66** |
| Output | Deterministic, all finite |

## Build & Run

```sh
# Build
make

# Run
./mesa_offline_compute_test

# Or build directly
gcc -o mesa_offline_compute_test mesa_offline_compute_test.c \
    -I$PREFIX/include -L$PREFIX/lib -lvulkan -lm
```

## Dependencies

- `mesa` (Vulkan ICDs: freedreno, swrast)
- `vulkan-loader` / `vulkan-headers`
- `glslang` (for SPIR-V shader compilation)
- `xxd` (to embed SPIR-V as C arrays)
