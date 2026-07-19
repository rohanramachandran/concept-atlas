# Decisions

## 2026-07-19 (experiments)

- Concept sets went from 4 to 12 templates each (96 prompts per set) so validation splits stop being coin flips; probe accuracies are means over 3 seeds with the spread reported.
- Patching uses one shared base prompt per set (the template prefix that stops right before the item), so all source-target effects are measured against the same baseline and one patched forward serves every target as a logit readout. 8 sources means 17 forwards total instead of 270.
- The diagonal is the built-in sanity check: patching an item's own activations must strongly boost that item. It does, on both models, by roughly an order of magnitude over off-diagonal effects; results would not have been published otherwise.
- The per-target metric is first-token logit minus the mean over other items, with a guard that fails the run if two items share a first token.
- Probes capture at the final prompt token while templates vary item position, so early-layer peaks (colors on gpt2, layer 0 on Llama) likely reflect surface token identity; this caveat is stated in results.md rather than smoothed over.
- Findings are framed as measurements under this prompt distribution, not model-general claims. The one cross-model regularity worth noting (black and white mutually excitatory, both suppressing chromatic colors) replicated without being sought.

## 2026-07-19 (scaling past gpt2)

- Model access now goes through a backend interface returning plain numpy arrays, so probes, patching, and the graph never touch framework specifics. Two backends: torch (forward hooks) and MLX (quantized).
- MLX has no forward-hook API. Rather than fork mlx_lm models, the MLX backend temporarily replaces entries in the model's layer list with taps that record or substitute a block's output, restoring the originals afterward. Wrapping the layer list is the narrowest possible surface and works across mlx_lm architectures.
- Llama-3.1-8B runs as MLX 4-bit, not torch/MPS fp16: this is a 24 GB machine, fp16 weights alone are 16 GB, and there is no workable 4-bit story for MPS in torch today. Quantization affects weights only; activations are captured in fp32.
- Activation collection is chunked to disk: per-layer preallocated memmaps, appended one prompt-chunk at a time, finalized with labels and a provenance meta.json. Peak RAM is one chunk of one forward pass regardless of corpus size.
- Batched last-token capture indexes through the attention mask, never position -1, so right padding cannot leak pad-token activations into probes. A regression test pins batched-vs-solo equality for unequal-length prompts.
- When base and source prompts tokenize to different lengths, patches are tail-aligned (the last k source positions replace the last k base positions). Concept templates differ near the end and metrics read the final position, so tails are the meaningful alignment.
- gpt2 stays the default model and the test suite stays toy-model only, so tests need no downloads and run in under a second.
