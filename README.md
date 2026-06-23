# pascallama.cpp

A set of optimised patches for `llama.cpp` targeting **NVIDIA Pascal GPUs** (SM 6.1 — GTX 10xx, P40, M40).

## What's different

### 1. `mmq_y` increased from 64 to 96 for Pascal

For SM 6.1 (Pascal), the matrix multiplication tile size `mmq_y` is set to **96** instead of the default **64**.

- Shared memory per block at `mmq_y=96`: ~28 KB (fits 2 blocks/SM within the 64 KB limit)
- Better data parallelism → higher throughput

### 2. `#pragma unroll` enabled on all DP4A vec_dot kernels

Nine vector dot-product kernels now have loop unrolling enabled:

| Kernel | Quantisation |
|---|---|
| `vec_dot_q4_0_q8_1_dp4a` | Q4_0 / Q8_1 |
| `vec_dot_q4_1_q8_1_dp4a` | Q4_1 / Q8_1 |
| `vec_dot_q8_0_q8_1_dp4a` | Q8_0 / Q8_1 |
| `vec_dot_q8_1_q8_1_dp4a` | Q8_1 / Q8_1 |
| `vec_dot_q8_0_16_q8_1_dp4a` | Q8_0 / Q8_1 (16-bit) |
| `vec_dot_q3_K_q8_1_dp4a` | Q3_K / Q8_1 |
| `vec_dot_q4_K_q8_1_dp4a` | Q4_K / Q8_1 |
| `vec_dot_q5_K_q8_1_dp4a` | Q5_K / Q8_1 |
| `vec_dot_q6_K_q8_1_dp4a` | Q6_K / Q8_1 |

## Benchmarks

All benchmarks run on **NVIDIA Tesla P40 + Quadro P6000** with MTP (Multi-Token Prediction).

| Config | GPU | Speed |
|---|---|---|
| Qwen3.6-27B MTP | P40 + P6000 | **27 tok/s** |
| Qwen3.6-35B-A3B MOE MTP | P40 + P6000 | **71 tok/s** |

> **Note:** These are single-GPU Pascal performance figures with the patches applied. Tensor parallelism (`--split-mode tensor`) is a separate configuration.

## Setup & Build

```bash
# Clone this repo
git clone https://github.com/madsteph74/pascallama.cpp
cd pascallama.cpp

# Setup: clone upstream llama.cpp and apply patches
chmod +x setup.sh
./setup.sh

# Build
chmod +x build.sh
./build.sh
```

### Build flags

- `GGML_CUDA=ON` — enable CUDA backend
- `GGML_CUDA_FORCE_MMQ=ON` — enable matrix-multiplication kernels
- `GGML_CUDA_FA_ALL_QUANTS=ON` — enable all quantisation variants
- `CMAKE_CUDA_ARCHITECTURES=61` — target Pascal (SM 6.1)

## Patch details

The only source modification is in `ggml/src/ggml-cuda/mmq.cuh`:

1. **Host path** (`get_mmq_y_host`): added a branch for DP4A (Pascal) returning 96 between Volta (128) and fallback (64)
2. **Device path** (`get_mmq_y_device`): added `#elif __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A` returning 96
3. **Nine vec_dot kernels**: uncommented `// #pragma unroll` → `#pragma unroll`

## Compatibility

- **Pascal (SM 6.1)**: full support — `mmq_y=96` + `#pragma unroll`
- **Volta+ (SM 7.0+)**: no change — uses upstream values (`mmq_y=128`)
- **Pre-Volta**: no change — uses fallback (`mmq_y=64`)

## License

MIT — same as upstream llama.cpp.

## Credits

Patches developed for production inference on NVIDIA Tesla P40 + Quadro P6000.
