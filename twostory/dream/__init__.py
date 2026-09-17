"""Vendored Dream-7B modeling code (Apache-2.0, Dream team / HKUNLP), pinned to Hub revision
6572adb5535263e4d1a337b56942ba48b6dee2a9, with the TwoStory cross-attention bridge threaded through.
Upstream copies live in reference/dream-base/; docs/dream_modeling.diff records every change."""
from .configuration_dream import DreamConfig
from .modeling_dream import DreamModel, DreamDecoderLayer, DreamRMSNorm, apply_rotary_pos_emb, repeat_kv, rotate_half
from .generation_utils import DreamGenerationConfig, sample_tokens
