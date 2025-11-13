"""Utilities to convert retrieved passages into model-consumable inputs (skeleton).

This module exposes helpers to convert retrieved text into tokens or into a form
that can be re-inserted into the model's attention (e.g., by prepending text,
or by constructing fake past_key_values if supported).
"""
from typing import List, Dict, Any


def convert_passages_to_inputs(passages: List[Dict]) -> str:
    """Turn ranked passages into a single text blob suitable for prepending.

    Simple default: concatenate top passages separated by separators.
    """
    texts = [p.get("text", "") for p in passages]
    return "\n\n---\n\n".join(texts)


def reintegrate_passages(model: Any, tokenizer: Any, passages: List[Dict]) -> None:
    """Reintegrate `passages` into the model/tokenizer pipeline.

    This is a stub showing where to implement reinsertion strategies:
      - prepend to input prompt and re-encode
      - create pseudo `past_key_values` if feasible
      - or call model with retrieved context as an additional input

    For the skeleton, we do not modify the model.
    """
    raise NotImplementedError()
