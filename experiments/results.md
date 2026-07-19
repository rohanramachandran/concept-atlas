# Results

Reproduce with `run_probes` and `run_patching` for each model; raw JSON lives beside this file in `results/`.

## Probe accuracy by layer

### gpt2 (gpt2, 12 layers)

| concept set | prompts | home layer | peak accuracy | chance |
|---|---|---|---|---|
| colors | 96 | 0 | 0.807 (std 0.050) | 0.125 |
| professions | 96 | 11 | 0.947 (std 0.043) | 0.125 |
| countries | 96 | 11 | 0.912 (std 0.066) | 0.125 |

![probe accuracy gpt2](results/probes-gpt2.png)

### llama (mlx-community/Llama-3.1-8B-Instruct-4bit, 32 layers)

| concept set | prompts | home layer | peak accuracy | chance |
|---|---|---|---|---|
| colors | 96 | 29 | 0.614 (std 0.066) | 0.125 |
| professions | 96 | 29 | 0.860 (std 0.108) | 0.125 |
| countries | 96 | 24 | 0.930 (std 0.066) | 0.125 |

![probe accuracy llama](results/probes-llama.png)

## Activation patching

### gpt2: colors at layer 0

Base prompt `The color of the object was`; source activations patched tail-aligned at layer 0. Diagonal median +4.632; off-diagonal median absolute effect 0.976.

| source | target | effect |
|---|---|---|
| black | orange | -4.398 |
| white | orange | -3.918 |
| black | white | +3.552 |
| red | orange | -3.551 |
| green | orange | -3.086 |
| orange | black | -3.042 |
| yellow | black | -2.981 |
| blue | orange | -2.864 |
| purple | orange | -2.845 |
| yellow | orange | -2.775 |

![patching gpt2](results/patching-gpt2-colors.png)

### gpt2: countries at layer 11

Base prompt `The documentary was filmed in`; source activations patched tail-aligned at layer 11. Diagonal median +3.286; off-diagonal median absolute effect 0.670.

| source | target | effect |
|---|---|---|
| India | Norway | -2.432 |
| Egypt | Norway | -1.869 |
| Norway | India | -1.590 |
| Japan | Egypt | -1.559 |
| Brazil | Norway | -1.536 |
| Egypt | Brazil | -1.520 |
| Kenya | Brazil | -1.489 |
| Kenya | Norway | -1.377 |
| Norway | Egypt | -1.363 |
| Canada | Brazil | -1.324 |

![patching gpt2](results/patching-gpt2-countries.png)

### gpt2: professions at layer 11

Base prompt `He trained for years to become a`; source activations patched tail-aligned at layer 11. Diagonal median +2.764; off-diagonal median absolute effect 1.037.

| source | target | effect |
|---|---|---|
| pilot | engineer | +3.520 |
| farmer | engineer | +2.948 |
| lawyer | engineer | +2.797 |
| farmer | lawyer | -2.651 |
| chef | doctor | -2.642 |
| chef | engineer | +2.534 |
| chef | lawyer | -2.534 |
| doctor | engineer | +2.497 |
| teacher | engineer | +2.477 |
| pilot | lawyer | -2.438 |

![patching gpt2](results/patching-gpt2-professions.png)

### llama: colors at layer 29

Base prompt `The color of the object was`; source activations patched tail-aligned at layer 29. Diagonal median +5.320; off-diagonal median absolute effect 0.854.

| source | target | effect |
|---|---|---|
| orange | black | -1.542 |
| black | white | +1.519 |
| orange | blue | -1.493 |
| white | green | -1.478 |
| white | purple | -1.458 |
| purple | blue | -1.423 |
| black | purple | -1.359 |
| black | yellow | -1.347 |
| orange | green | -1.306 |
| purple | red | -1.298 |

![patching llama](results/patching-llama-colors.png)

### llama: countries at layer 24

Base prompt `The documentary was filmed in`; source activations patched tail-aligned at layer 24. Diagonal median +6.549; off-diagonal median absolute effect 1.250.

| source | target | effect |
|---|---|---|
| Norway | Kenya | -3.212 |
| Norway | India | -3.122 |
| Brazil | Kenya | -3.026 |
| France | Kenya | -2.867 |
| Norway | Egypt | -2.406 |
| Japan | Kenya | -2.273 |
| Kenya | France | -2.266 |
| Egypt | Norway | -2.175 |
| Kenya | India | -2.065 |
| Canada | Kenya | -2.035 |

![patching llama](results/patching-llama-countries.png)

### llama: professions at layer 29

Base prompt `He trained for years to become a`; source activations patched tail-aligned at layer 29. Diagonal median +5.959; off-diagonal median absolute effect 1.658.

| source | target | effect |
|---|---|---|
| engineer | chef | -4.359 |
| lawyer | chef | -4.211 |
| nurse | chef | -4.203 |
| pilot | engineer | +4.038 |
| doctor | chef | -4.036 |
| pilot | chef | -3.734 |
| nurse | pilot | -3.452 |
| pilot | doctor | -3.441 |
| teacher | chef | -3.415 |
| doctor | engineer | +3.338 |

![patching llama](results/patching-llama-professions.png)

## Notes

- Probes: logistic regression on last-token residual stream, 80/20 train/val split, mean over 3 seeds; shaded bands are the seed spread.
- Patching: one patched forward per source item; every target is a logit readout of the same forward. Effects are deltas against the unpatched base prompt.
- Capture position is the prompt's final token while templates place the item at varied positions, so early-layer peaks (colors on gpt2) likely reflect surface token identity rather than abstraction; late-layer peaks are the more meaningful signal.
- Small-scale by design: 96 prompts per concept set on a laptop. Directionally useful, not a substitute for large-sample studies.
