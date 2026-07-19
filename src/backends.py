"""Model backends: one capture-and-patch interface across frameworks.

Probing, patching, and graph code consume plain numpy arrays; a backend owns
the framework mechanics of producing them. Two implementations:

- ``TorchBackend``: Hugging Face models via forward hooks (``src.hooks``),
  with attention-mask-correct last-token indexing under padding.
- ``MlxBackend``: quantized models via ``mlx_lm``. MLX modules have no hook
  API, so the backend temporarily replaces entries of the model's layer list
  with taps that record or substitute a block's output. Same two capabilities,
  4-bit weights, Apple-silicon friendly.

When a patch's source sequence is shorter or longer than the base sequence,
both backends align the *tails*: the last ``k`` positions of the source
replace the last ``k`` positions of the base. Concept prompts share templates
and differ near the end, and metrics read the final position, so tail
alignment is the meaningful choice.
"""
from __future__ import annotations

import abc
from typing import Callable, Iterator, Sequence

import numpy as np

Metric = Callable[[np.ndarray], float]


def logit_diff_metric(
    target_ids: Sequence[int],
    against_ids: Sequence[int] | None = None,
) -> Metric:
    """Mean final-position logit of ``target_ids``, minus ``against_ids`` if given."""

    def metric(logits: np.ndarray) -> float:
        last = logits[0, -1]
        score = float(last[list(target_ids)].mean())
        if against_ids is not None:
            score -= float(last[list(against_ids)].mean())
        return score

    return metric


class Backend(abc.ABC):
    """Capture residual-stream activations and run patched forwards."""

    model_name: str = ""

    @property
    @abc.abstractmethod
    def n_layers(self) -> int: ...

    @property
    @abc.abstractmethod
    def d_model(self) -> int: ...

    @abc.abstractmethod
    def capture_last_token(self, texts: Sequence[str], layers: Sequence[int]) -> dict[int, np.ndarray]:
        """Last real token's residual stream per layer: ``{layer: (len(texts), d_model)}``."""

    @abc.abstractmethod
    def capture_sequence(self, text: str, layer: int) -> np.ndarray:
        """Full-sequence residual stream at one layer: ``(1, seq, d_model)``."""

    @abc.abstractmethod
    def logits(self, text: str, patch: tuple[int, np.ndarray, Sequence[int] | None] | None = None) -> np.ndarray:
        """Output logits ``(1, seq, vocab)``, optionally with ``(layer, acts, positions)`` patched in."""

    @abc.abstractmethod
    def token_ids(self, text: str) -> list[int]:
        """Tokenize without special tokens; used to build logit metrics."""

    def iter_last_token(
        self, texts: Sequence[str], layers: Sequence[int], chunk_size: int = 8,
    ) -> Iterator[dict[int, np.ndarray]]:
        """Yield ``capture_last_token`` results in chunks of ``chunk_size`` texts."""
        for start in range(0, len(texts), chunk_size):
            yield self.capture_last_token(texts[start:start + chunk_size], layers)


def _tail_align(base_len: int, source: np.ndarray) -> tuple[int, np.ndarray]:
    """Number of trailing positions to patch, and the matching source slice."""
    k = min(base_len, source.shape[1])
    return k, source[:, -k:, :]


class TorchBackend(Backend):
    """Hugging Face causal LM through forward hooks."""

    def __init__(self, model, tokenizer, *, d_model: int | None = None, device: str | None = None,
                 model_name: str = ""):
        import torch

        from src.hooks import resolve_blocks

        self._torch = torch
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name = model_name
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self._blocks = resolve_blocks(model)
        if d_model is None:
            config = getattr(model, "config", None)
            d_model = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
            if d_model is None:
                raise ValueError("pass d_model explicitly for models without a standard config")
        self._d_model = int(d_model)

    @classmethod
    def from_pretrained(cls, name: str, device: str | None = None) -> "TorchBackend":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(name)
        return cls(model, tokenizer, device=device, model_name=name)

    @property
    def n_layers(self) -> int:
        return len(self._blocks)

    @property
    def d_model(self) -> int:
        return self._d_model

    def _encode(self, texts: Sequence[str]):
        enc = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def token_ids(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def capture_last_token(self, texts: Sequence[str], layers: Sequence[int]) -> dict[int, np.ndarray]:
        from src.hooks import ResidualCache

        torch = self._torch
        ids, mask = self._encode(texts)
        with torch.no_grad(), ResidualCache(self.model, layers=list(layers)) as cache:
            self.model(ids, attention_mask=mask)
        last = mask.sum(dim=1) - 1
        rows = torch.arange(ids.shape[0], device=ids.device)
        return {
            layer: cache.activations[layer][rows, last].float().cpu().numpy()
            for layer in layers
        }

    def capture_sequence(self, text: str, layer: int) -> np.ndarray:
        from src.hooks import ResidualCache

        torch = self._torch
        ids, mask = self._encode([text])
        with torch.no_grad(), ResidualCache(self.model, layers=[layer]) as cache:
            self.model(ids, attention_mask=mask)
        return cache.activations[layer].float().cpu().numpy()

    def logits(self, text: str, patch=None) -> np.ndarray:
        torch = self._torch
        ids, mask = self._encode([text])
        handle = None
        if patch is not None:
            layer, acts, positions = patch
            source = torch.from_numpy(np.asarray(acts))

            def hook(_module, _inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                r = source.to(device=h.device, dtype=h.dtype)
                new = h.clone()
                if positions is None:
                    k, tail = _tail_align(h.shape[1], r.cpu().numpy())
                    new[:, -k:, :] = torch.from_numpy(tail).to(device=h.device, dtype=h.dtype)
                else:
                    new[:, list(positions), :] = r[:, list(positions), :]
                if isinstance(output, tuple):
                    return (new,) + tuple(output[1:])
                return new

            handle = self._blocks[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = self.model(ids, attention_mask=mask)
            logits = out.logits if hasattr(out, "logits") else out
            return logits.float().cpu().numpy()
        finally:
            if handle is not None:
                handle.remove()


class _Tap:
    """Callable stand-in for one MLX transformer block.

    Delegates attribute access to the wrapped block, because mlx_lm model
    code inspects per-layer attributes (e.g. ``use_sliding``) while iterating.
    """

    def __init__(self, inner):
        self.inner = inner
        self.record = False
        self.recorded = None
        self.patch = None  # (acts np (1, k, d) tail-aligned mask, positions)

    def __getattr__(self, name):
        return getattr(self.__dict__["inner"], name)

    def __call__(self, *args, **kwargs):
        import mlx.core as mx

        out = self.inner(*args, **kwargs)
        if self.patch is not None:
            acts, positions = self.patch
            seq = out.shape[1]
            replacement = np.array(out, copy=True)
            if positions is None:
                k, tail = _tail_align(seq, acts)
                replacement[:, -k:, :] = tail
            else:
                replacement[:, list(positions), :] = acts[:, list(positions), :]
            out = mx.array(replacement).astype(out.dtype)
        if self.record:
            self.recorded = np.array(out.astype(mx.float32))
        return out


class MlxBackend(Backend):
    """Quantized causal LM through mlx_lm, hooks emulated by wrapping layers."""

    def __init__(self, model_name: str):
        from mlx_lm import load

        self.model, self.tokenizer = load(model_name)
        self.model_name = model_name
        inner = getattr(self.model, "model", self.model)
        if not hasattr(inner, "layers"):
            raise ValueError(f"cannot locate transformer layers on {type(self.model).__name__}")
        self._inner = inner
        self._d_model: int | None = None

    @property
    def n_layers(self) -> int:
        return len(self._inner.layers)

    @property
    def d_model(self) -> int:
        if self._d_model is None:
            acts = self.capture_last_token(["probe"], layers=[0])
            self._d_model = acts[0].shape[1]
        return self._d_model

    def _forward(self, text: str, taps: dict[int, _Tap]):
        import mlx.core as mx

        ids = mx.array([self.tokenizer.encode(text)])
        originals = {}
        try:
            for index, tap in taps.items():
                originals[index] = self._inner.layers[index]
                tap.inner = originals[index]
                self._inner.layers[index] = tap
            logits = self.model(ids)
            mx.eval(logits)
            return np.array(logits.astype(mx.float32))
        finally:
            for index, original in originals.items():
                self._inner.layers[index] = original

    def capture_last_token(self, texts: Sequence[str], layers: Sequence[int]) -> dict[int, np.ndarray]:
        out: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        for text in texts:
            taps = {layer: _Tap(None) for layer in layers}
            for tap in taps.values():
                tap.record = True
            self._forward(text, taps)
            for layer, tap in taps.items():
                out[layer].append(tap.recorded[0, -1, :])
        return {layer: np.stack(rows) for layer, rows in out.items()}

    def capture_sequence(self, text: str, layer: int) -> np.ndarray:
        tap = _Tap(None)
        tap.record = True
        self._forward(text, {layer: tap})
        return tap.recorded

    def logits(self, text: str, patch=None) -> np.ndarray:
        taps = {}
        if patch is not None:
            layer, acts, positions = patch
            tap = _Tap(None)
            tap.patch = (np.asarray(acts, dtype=np.float32), positions)
            taps[layer] = tap
        return self._forward(text, taps)

    def token_ids(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))


def causal_effect(
    backend: Backend,
    base_text: str,
    source_text: str,
    layer: int,
    metric: Metric,
    positions: Sequence[int] | None = None,
) -> float:
    """Backend-agnostic activation-patching effect.

    Runs ``base_text`` clean and with ``source_text``'s layer activations
    patched in; returns the metric delta. Positive: source pushes the metric
    up (excitatory); negative: down (inhibitory).
    """
    source_acts = backend.capture_sequence(source_text, layer)
    base = metric(backend.logits(base_text))
    patched = metric(backend.logits(base_text, patch=(layer, source_acts, positions)))
    return patched - base
