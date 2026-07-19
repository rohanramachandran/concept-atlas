"""Render figures and results.md from the experiment JSONs.

    python -m experiments.make_results
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import CONCEPT_SETS, RESULTS_DIR

SET_COLORS = {"colors": "#d62728", "professions": "#1f77b4", "countries": "#2ca02c"}


def probe_figure(probes: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for set_name, data in probes["sets"].items():
        layers = np.array(data["layers"])
        mean = np.array(data["acc_mean"])
        spread = np.array(data["acc_std"])
        color = SET_COLORS.get(set_name, "#555555")
        ax.plot(layers, mean, label=set_name, color=color, linewidth=1.8)
        ax.fill_between(layers, mean - spread, mean + spread, color=color, alpha=0.18)
        ax.scatter([data["home_layer"]], [data["home_accuracy"]], color=color, zorder=5, s=28)
    chance = 1.0 / 8
    ax.axhline(chance, color="#888888", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("layer")
    ax.set_ylabel("probe accuracy (mean over seeds)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Linear probe accuracy by layer: {probes['alias']}")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def patching_figure(patching: dict, out: Path) -> None:
    matrix = np.array(patching["matrix"])
    items = patching["items"]
    limit = max(abs(matrix.min()), abs(matrix.max()))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(items)), items, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(items)), items, fontsize=8)
    ax.set_xlabel("target (readout)")
    ax.set_ylabel("source (patched in)")
    ax.set_title(f"Patching effects, {patching['alias']} {patching['set']} "
                 f"(layer {patching['layer']})", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="metric delta")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    lines = ["# Results", ""]
    lines.append("Reproduce with `run_probes` and `run_patching` for each model; "
                 "raw JSON lives beside this file in `results/`.")
    lines.append("")

    probe_files = sorted(RESULTS_DIR.glob("probes-*.json"))
    lines.append("## Probe accuracy by layer")
    lines.append("")
    for path in probe_files:
        probes = json.loads(path.read_text())
        fig_path = RESULTS_DIR / f"probes-{probes['alias']}.png"
        probe_figure(probes, fig_path)
        rows = []
        for set_name in CONCEPT_SETS:
            data = probes["sets"].get(set_name)
            if data is None:
                continue
            rows.append([
                set_name,
                str(data["n_prompts"]),
                str(data["home_layer"]),
                f"{data['home_accuracy']:.3f} (std {data['acc_std'][data['home_layer']]:.3f})",
                f"{data['chance']:.3f}",
            ])
        lines.append(f"### {probes['alias']} ({probes['model']}, {probes['n_layers']} layers)")
        lines.append("")
        lines.append(table(rows, ["concept set", "prompts", "home layer",
                                  "peak accuracy", "chance"]))
        lines.append("")
        lines.append(f"![probe accuracy {probes['alias']}](results/{fig_path.name})")
        lines.append("")

    patch_files = sorted(RESULTS_DIR.glob("patching-*.json"))
    if patch_files:
        lines.append("## Activation patching")
        lines.append("")
        for path in patch_files:
            patching = json.loads(path.read_text())
            fig_path = RESULTS_DIR / f"patching-{patching['alias']}-{patching['set']}.png"
            patching_figure(patching, fig_path)
            lines.append(f"### {patching['alias']}: {patching['set']} at layer {patching['layer']}")
            lines.append("")
            lines.append(f"Base prompt `{patching['base_prompt']}`; source activations patched "
                         f"tail-aligned at layer {patching['layer']}. Diagonal median "
                         f"{patching['diagonal_median']:+.3f}; off-diagonal median absolute effect "
                         f"{patching['off_diagonal_median_abs']:.3f}.")
            lines.append("")
            rows = [[e["source"], e["target"], f"{e['effect']:+.3f}"]
                    for e in patching["top_edges"]]
            lines.append(table(rows, ["source", "target", "effect"]))
            lines.append("")
            lines.append(f"![patching {patching['alias']}](results/{fig_path.name})")
            lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Probes: logistic regression on last-token residual stream, 80/20 "
                 "train/val split, mean over 3 seeds; shaded bands are the seed spread.")
    lines.append("- Patching: one patched forward per source item; every target is a "
                 "logit readout of the same forward. Effects are deltas against the "
                 "unpatched base prompt.")
    lines.append("- Capture position is the prompt's final token while templates place "
                 "the item at varied positions, so early-layer peaks (colors on gpt2) "
                 "likely reflect surface token identity rather than abstraction; "
                 "late-layer peaks are the more meaningful signal.")
    lines.append("- Small-scale by design: 96 prompts per concept set on a laptop. "
                 "Directionally useful, not a substitute for large-sample studies.")
    lines.append("")

    out = Path(__file__).resolve().parent / "results.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
