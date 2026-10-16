# pascallama.cpp

A set of optimised patches for `llama.cpp` targeting **NVIDIA Pascal GPUs** (SM 6.1 — GTX 10xx, P40, M40).

**Tested against upstream llama.cpp @ `a0e6dc738` (master, Oct 2026).** `setup.sh` clones and checks out
exactly that commit before applying the patches — upstream moves fast, and the MMQ code path got
restructured, so a moving target broke the setup between June and October 2026. If upstream later
changes `ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh` or `mmq-vec-dot.cuh` in a conflicting way, the
tile values below need to be re-ported onto the new shape (see "Updating the patch" at the bottom).

## What's different

Upstream now keeps its own Pascal-DP4A configuration file, `ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh`
(with small conservative tiles: 8/16/32/64 rows), and keeps the DP4A `#pragma unroll`s commented out in
`ggml/src/ggml-cuda/mmq-vec-dot.cuh`. For SM 6.1 this patch:

### 1. Retunes the Pascal-DP4A config to 96-row tiles

All 242 configuration rows in `mmq-config-pascal-dp4a.cuh` are set to the 96-row tile
(shaped like upstream's own rows, only the tile width changes):

```c
CASE(GGML_TYPE_Q1_0, 256, 2,  64,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
```

- Shared memory per block at 96 rows: ~28 KB (2 blocks/SM within the 64 KB limit)
- Better data parallelism → measurably higher throughput on SM 6.1 (measured below)

### 2. Re-enables `#pragma unroll` on nine DP4A vec_dot kernels

Nine `// #pragma unroll` lines are uncommented in `mmq-vec-dot.cuh`.

That is the whole change: no new files, no engine behaviour changes, Volta and newer keep upstream's
own values (the patch's rows are the Pascal-DP4A file only).

## Benchmarks

All benchmarks run on **NVIDIA Tesla P40 with MTP** (multi-token prediction). Decode speed:

| Config | GPU | Speed |
|---|---|---|
| Qwen3.6-35B-A3B MOE MTP | P40 | **~75 tok/s** |

(The older June-2026 numbers — 27 tok/s on a 27B MTP and 71 tok/s on the 35B MOE over P40+P6000 — are
kept for reference in the git history of this README.)

> Single-GPU Pascal figures. Tensor parallelism (`--split-mode tensor`) is a separate configuration.

## Setup & Build

```bash
# Clone this repo
git clone https://github.com/madsteph74/pascallama.cpp
cd pascallama.cpp

# Setup: clone upstream llama.cpp at the tested commit, apply the patches
chmod +x setup.sh
./setup.sh

# Build
chmod +x build.sh
./build.sh
```

### Build flags

- `GGML_CUDA=ON` — enable the CUDA backend
- `GGML_CUDA_FORCE_MMQ=ON` — force MMQ instead of cuBLAS GEMM
- `CMAKE_CUDA_ARCHITECTURES=61` — target Pascal (SM 6.1)

(The 71 tok/s P40+P6000 run in the git history also used `GGML_CUDA_FA_ALL_QUANTS=ON`; the current
build drops it.) 

## Patch details

The patch touches exactly two upstream files:

1. `ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh` — the 242 config rows re-typed to 96-row tiles
2. `ggml/src/ggml-cuda/mmq-vec-dot.cuh` — nine `// #pragma unroll` re-enabled

Compatibility as of the tested commit:

- **Pascal (SM 6.1)**: 96-row tiles + unrolled DP4A vec_dots
- **Volta+ (SM 7.0+)**: untouched — upstream's own values
- **Pre-Volta**: untouched — upstream's own values

## Updating the patch

If upstream changes one of the two files, regenerate the diff from a clean checkout of the new
upstream commit after re-porting the tile values, and update the pinned commit in `setup.sh`
(and this README). The June-2026 single-file patch against `mmq.cuh` is kept in
`patches/legacy/pascal-mmq.patch` for reference.

## License

MIT — same as upstream llama.cpp.

## Credits

Patches developed for production inference on NVIDIA Tesla P40 (+ Quadro P6000 for the earlier
two-GPU runs). Upstream llama.cpp by the ggml-org team and contributors.