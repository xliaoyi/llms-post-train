"""Memory-light causal LM loss: logits only at trained positions, computed in checkpointed chunks.

Full logits for a 32k-token trajectory with a 248k vocabulary would need ~30+ GB; here only
positions with a label are projected, CHUNK at a time, and recomputed in the backward pass.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CHUNK = 2048


def _chunk_loss(h, w, y, softcap):
    logits = (h @ w.T).float()
    if softcap:  # Gemma: logits = cap * tanh(logits / cap)
        logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits, y, reduction="sum")


def sft_loss(model, input_ids, labels):
    """Mean token cross-entropy over positions where labels != -100 (batch size 1)."""
    hidden = model.model(input_ids=input_ids).last_hidden_state[0, :-1]
    target = labels[0, 1:]
    keep = target != -100
    hidden, target = hidden[keep], target[keep]
    w = model.lm_head.weight
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    total = sum(checkpoint(_chunk_loss, hidden[i:i + CHUNK], w, target[i:i + CHUNK], softcap, use_reentrant=False)
                for i in range(0, len(target), CHUNK))
    return total / max(len(target), 1)
