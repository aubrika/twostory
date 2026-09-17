"""Fixed prompts used by every phase. Never edit after Phase 0 has recorded baselines."""

# A plain-text continuation prompt (Dream-v0-Base and Qwen2.5-7B are base models, not chat models).
PROMPT_SHORT = (
    "The history of the lighthouse at Cape Verrin begins in 1847, when the harbour board "
    "first petitioned the Admiralty for a light. The petition was refused twice. "
)

# Source paragraphs for building fixed-length prompts. Original text; repeated and truncated
# to an exact token count by prompts.build_prompt().
_FILLER = (
    "Over the following decade the board kept careful records of every wreck on the "
    "northern reef, noting the vessel, the cargo, the weather, and the loss of life. "
    "These ledgers survive in the county archive and form the basis of the present account. "
    "The keeper's cottage was built of local granite, with a slate roof that was replaced "
    "three times before the present iron sheeting was fitted. Fresh water came from a "
    "cistern fed by the roof, supplemented in dry summers by a cart from the village. "
    "The light itself was a fixed white light of the second order, converted to an "
    "occulting pattern in 1889 after complaints from masters that it was confused with "
    "the lights of the town. Fog signalling began with a bell, then a reed horn, and "
    "finally a siren driven by compressed air from an engine house at the foot of the tower. "
)


def build_prompt(tokenizer, n_tokens: int) -> str:
    """Return a prompt that tokenizes to exactly n_tokens with `tokenizer`."""
    text = PROMPT_SHORT
    while len(tokenizer(text).input_ids) < n_tokens:
        text += _FILLER
    ids = tokenizer(text).input_ids[:n_tokens]
    out = tokenizer.decode(ids)
    # Re-tokenization of decoded text can shift by a token at the boundary; trim until exact.
    while len(tokenizer(out).input_ids) > n_tokens:
        out = out[:-1]
    assert len(tokenizer(out).input_ids) == n_tokens, len(tokenizer(out).input_ids)
    return out


# Prompt lengths used for throughput measurements (Phase 3 / Phase 5).
PROMPT_LENGTHS = {"short": None, "p512": 512, "p2k": 2048}
