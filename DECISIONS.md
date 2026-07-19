# Decisions

## 2026-07-19 (scaling past gpt2)

- Model access now goes through a backend interface returning plain numpy arrays, so probes, patching, and the graph never touch framework specifics. Two backends: torch (forward hooks) and MLX (quantized).
- MLX has no forward-hook API. Rather than fork mlx_lm models, the MLX backend temporarily replaces entries in the model's layer list with taps that record or substitute a block's output, restoring the originals afterward. Wrapping the layer list is the narrowest possible surface and works across mlx_lm architectures.
- Llama-3.1-8B runs as MLX 4-bit, not torch/MPS fp16: this is a 24 GB machine, fp16 weights alone are 16 GB, and there is no workable 4-bit story for MPS in torch today. Quantization affects weights only; activations are captured in fp32.
- Activation collection is chunked to disk: per-layer preallocated memmaps, appended one prompt-chunk at a time, finalized with labels and a provenance meta.json. Peak RAM is one chunk of one forward pass regardless of corpus size.
- Batched last-token capture indexes through the attention mask, never position -1, so right padding cannot leak pad-token activations into probes. A regression test pins batched-vs-solo equality for unequal-length prompts.
- When base and source prompts tokenize to different lengths, patches are tail-aligned (the last k source positions replace the last k base positions). Concept templates differ near the end and metrics read the final position, so tails are the meaningful alignment.
- gpt2 stays the default model and the test suite stays toy-model only, so tests need no downloads and run in under a second.
