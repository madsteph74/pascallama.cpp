# pascallama.cpp

A set of optimised patches for `llama.cpp` targeting **NVIDIA Pascal GPUs** (SM 6.1 — GTX 10xx, P40, M40).

**Tested against upstream llama.cpp @ `a0e6dc738` (master, Oct 2026 — v0.7.0-era, includes the upstream
MoE expert cache).** `setup.sh` clones and checks out exactly that commit before applying the patches —
upstream moves fast, and the MMQ code path got restructured, so a moving target broke the setup between
June and October 2026. If upstream later changes `ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh` or
`mmq-vec-dot.cuh` in a conflicting way, the tile values below need to be re-ported onto the new shape
(see "Updating the patch" at the bottom).

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

All runs 16 Oct 2026 on **Tesla P40 (+ Quadro P6000) with MTP** (multi-token prediction), rebuilt
llama.cpp at the pinned commit. Decode speed:

| Config | GPUs / split | Speed | MTP accept |
|---|---|---|---|
| Qwen3.6-35B-A3B MOE, Q8_0, MTP, ctx 262144 (KV q8_0) | P40+P6000, tensor-split | **70.8 tok/s** | 79.3 % |
| Qwen3.8-27B dense, UD-Q6_K_XL, MTP (self-draft head), ctx 262144 | P40+P6000, tensor-split | 27.6 tok/s | **85.4 %** |
| Qwen3.8-Flash-Next UD-Q2_K_XL (74.5 GB), + upstream MoE cache `--moe-cache-mib 6544` | P40+P6000, layer-split | 24.9 tok/s (17.0 baseline → **+47 %**) | — |
| Qwen3.6-35B-A3B MOE MTP (single GPU, earlier Oct figure) | P40 | ~75 tok/s | — |

(The June-2026 numbers — 27 tok/s on a 27B MTP and 71 tok/s on the 35B MOE over P40+P6000 — are kept
for reference in the git history of this README; the 16 Oct re-measure of the 35B at 262k ctx matched.)

**Why dense loses to MoE:** the dense 27B re-reads its full 24 GB of weights for *every* token, while
the 3.3B-active MoE reads a fraction of that. Even with a flotte-record 85.4 % draft acceptance, the
dense config lands ~2.6× below the MoE one — on Pascal MMQ, active bytes per token decide, and the
DP4A patches only influence *how* those bytes are read.

## Upstream MoE expert cache on Pascal (tested v0.7.0 @ pinned commit)

The upstream expert-cache merge (PR #29887 + #30112, released in v0.7.0) **works on SM 6.1** — no
compute-capability gate in the merged implementation. Measured notes for running it on Pascal:

- Decodes with experts overflowed to RAM benefit measurably: 17.0 → 24.9 tok/s (**+47 %**) at
  `--moe-cache-mib 6544` on the P40+P6000 duo (Qwen3.8-Flash-Next UD-Q2_K_XL).
- **Size the cache, don't max it**: an aggressive `-cmoe`/fit variant measured *worse* on the same duo
  (15.2 tok/s, prefill −32 %). A/B test the cache size against your model's resident-expert footprint.
- **`--split-mode tensor` and the MoE cache are mutually exclusive** (upstream throws). Choose one
  strategy: tensor-split when the whole model fits in VRAM, layer-split + cache when experts overflow
  to RAM.
- Cache hits only apply to ubatch ≤ 32 tokens; larger batches take the classic offload path.
- The cache serves the target context's MoE layers; the MTP draft never gets one by design
  (`-cmoed` offloads draft experts to CPU if needed).
- On Pascal with no NCCL, `--split-mode tensor` falls back to the internal butterfly AllReduce and
  works fine on a 2-GPU duo.

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