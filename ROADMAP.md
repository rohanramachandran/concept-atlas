# Roadmap

## M1 — Probing
- [x] Train/evaluate linear probes on residual-stream activations
- [x] Report layer-wise accuracy (`sweep_layers`)
- [ ] Batched activation collection for large concept sets

## M2 — Causal graph
- [x] Activation patching between concept pairs
- [x] Weighted, typed graph extraction
- [x] JSON export (D3 format)
- [ ] Significance filtering (effect vs. random-patch baseline)

## M3 — Explorer
- [x] D3 force-directed explorer with demo graph
- [x] Filtering (edge weight threshold, concept sets)
- [ ] Probe visualization (per-layer accuracy curves per node)
- [ ] Load arbitrary graph JSON from the UI

## M4 — Model diffing
- [ ] Compare two models over the same concept sets
- [ ] Graph diff visualization (edges gained/lost/reweighted)
