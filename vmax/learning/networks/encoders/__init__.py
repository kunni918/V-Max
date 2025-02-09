# Copyright 2025 Valeo.

"""Encoders module."""

from .attention_utils import AttentionLayer, FeedForward, LocalAttentionLayer, ReZero, nearest_neighbors_jax
from .embedding_utils import build_mlp_embedding
from .mgail import MGAILEncoder
from .mlp import MLPEncoder
from .mtr import MTREncoder
from .perceiver import PerceiverEncoder
from .wayformer import WayformerEncoder


Encoder = MLPEncoder | PerceiverEncoder | WayformerEncoder | MTREncoder | MGAILEncoder


def get_encoder(encoder_name: str, **kwargs) -> Encoder:
    """Retrieve an encoder class by its name.

    Args:
        encoder_name: Name of the encoder.
        **kwargs: Additional arguments to configure the encoder.

    Returns:
        The encoder class corresponding to the given name.

    Raises:
        ValueError: If an unknown encoder name is provided.

    """
    encoders = {
        "mlp": MLPEncoder,
        "perceiver": PerceiverEncoder,
        "wayformer": WayformerEncoder,
        "mtr": MTREncoder,
        "mgail": MGAILEncoder,
    }

    try:
        return encoders[encoder_name.lower()]
    except KeyError:
        raise ValueError(f"Unknown encoder: {encoder_name}") from None


__all__ = [
    "AttentionLayer",
    "Encoder",
    "FeedForward",
    "LocalAttentionLayer",
    "MGAILEncoder",
    "MLPEncoder",
    "MTREncoder",
    "PerceiverEncoder",
    "ReZero",
    "WayformerEncoder",
    "build_mlp_embedding",
    "get_encoder",
    "nearest_neighbors_jax",
]
