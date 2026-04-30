"""
GLM-Image attention backend selector.

Switches the diffusers GlmImageTransformer2DModel between SDPA, xFormers,
SageAttention, and FlashAttention-2 by replacing AttnProcessor objects.

This mirrors what `enable_xformers_memory_efficient_attention()` does in
diffusers, but generalised to the four common backends.
"""

# MANUAL bug-fix (Apr 2026): Native GLM-Image — attention backend support.

import importlib
import logging

import torch


def _has(mod_name):
    try:
        importlib.import_module(mod_name)
        return True
    except Exception:
        return False


def _detect_available():
    return {
        "sdpa": True,  # always available with torch >= 2.0
        "xformers": _has("xformers") and _has("xformers.ops"),
        "sage": _has("sageattention"),
        "flash": _has("flash_attn"),
    }


AVAILABLE = _detect_available()


def apply_attention_backend(transformer, backend: str) -> str:
    """Set the attention backend on the diffusers transformer.

    Returns the backend that was actually applied (may differ from request
    if requested backend is unavailable, in which case we fall back to sdpa).
    """
    backend = (backend or "sdpa").lower()
    if backend != "sdpa" and not AVAILABLE.get(backend, False):
        logging.warning(
            "[GLM-Image] Attention backend '%s' is not installed. "
            "Falling back to sdpa. Available: %s",
            backend, [k for k, v in AVAILABLE.items() if v],
        )
        backend = "sdpa"

    if backend == "xformers":
        try:
            transformer.enable_xformers_memory_efficient_attention()
            return "xformers"
        except Exception as e:
            logging.warning("[GLM-Image] xformers enable failed: %s", e)
            backend = "sdpa"

    if backend == "sage":
        try:
            _set_processor(transformer, _SageAttnProcessor())
            return "sage"
        except Exception as e:
            logging.warning("[GLM-Image] sage enable failed: %s", e)
            backend = "sdpa"

    if backend == "flash":
        try:
            _set_processor(transformer, _FlashAttnProcessor())
            return "flash"
        except Exception as e:
            logging.warning("[GLM-Image] flash enable failed: %s", e)
            backend = "sdpa"

    # SDPA (default): restore stock processors if available
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0
        _set_processor(transformer, AttnProcessor2_0())
    except Exception:
        pass
    return "sdpa"


def _set_processor(transformer, processor):
    """Apply `processor` to all attention layers in the transformer."""
    if hasattr(transformer, "set_attn_processor"):
        # Build a dict mapping every attention key to our processor instance
        attn_procs = {}
        for name in transformer.attn_processors:
            attn_procs[name] = processor
        transformer.set_attn_processor(attn_procs)
    else:
        # Walk modules and patch any with a `processor` attribute
        for m in transformer.modules():
            if hasattr(m, "processor") and hasattr(m, "set_processor"):
                m.set_processor(processor)


# ---------------------------------------------------------------------------
# Custom AttnProcessor implementations
# ---------------------------------------------------------------------------


class _SageAttnProcessor:
    """SageAttention v1/v2 processor compatible with diffusers Attention API."""

    def __init__(self):
        from sageattention import sageattn  # noqa: F401
        self._fn = None  # resolved lazily

    def _resolve(self):
        if self._fn is None:
            from sageattention import sageattn
            self._fn = sageattn
        return self._fn

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        return _generic_attn_call(
            attn, hidden_states, encoder_hidden_states, attention_mask,
            self._resolve(), is_sage=True,
        )


class _FlashAttnProcessor:
    """FlashAttention-2 processor compatible with diffusers Attention API."""

    def __init__(self):
        from flash_attn import flash_attn_func  # noqa: F401
        self._fn = None

    def _resolve(self):
        if self._fn is None:
            from flash_attn import flash_attn_func
            self._fn = flash_attn_func
        return self._fn

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        return _generic_attn_call(
            attn, hidden_states, encoder_hidden_states, attention_mask,
            self._resolve(), is_sage=False,
        )


def _generic_attn_call(attn, hidden_states, encoder_hidden_states,
                       attention_mask, attn_fn, is_sage: bool):
    """Diffusers-style Attention forward routed to a custom kernel."""
    residual = hidden_states
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        b, c, h, w = hidden_states.shape
        hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

    batch, seq_len, _ = hidden_states.shape
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    if attn.norm_cross is not None:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    q = attn.to_q(hidden_states)
    k = attn.to_k(encoder_hidden_states)
    v = attn.to_v(encoder_hidden_states)

    head_dim = q.shape[-1] // attn.heads
    q = q.view(batch, -1, attn.heads, head_dim)
    k = k.view(batch, -1, attn.heads, head_dim)
    v = v.view(batch, -1, attn.heads, head_dim)

    # Both sageattn and flash_attn_func accept (B, S, H, D) layout
    out = attn_fn(q, k, v)  # → (B, S, H, D)
    out = out.reshape(batch, -1, attn.heads * head_dim)
    out = out.to(q.dtype)

    out = attn.to_out[0](out)
    out = attn.to_out[1](out)

    if input_ndim == 4:
        out = out.transpose(-1, -2).reshape(batch, attn.heads * head_dim, h, w)

    if attn.residual_connection:
        out = out + residual
    out = out / attn.rescale_output_factor
    return out
