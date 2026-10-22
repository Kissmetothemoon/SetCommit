"""Pure-diffusion sampler with the LLaDA paper's EOS confidence guard."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
import sys

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "LLaDA"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from generate import add_gumbel_noise, get_num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, attention_mask=None, *, steps=512, gen_length=512,
             block_length=512, temperature=0.0, cfg_scale=0.0, mask_id=126336,
             eos_guard=False, logits_eos_inf=False,
             confidence_suppress_tokens=()):
    if gen_length % block_length or steps % (gen_length // block_length):
        raise ValueError("generation length and steps must divide block length")
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat((attention_mask,
            torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype,
                       device=model.device)), dim=-1)
    prompt_index = x != mask_id
    blocks = gen_length // block_length
    local_steps = steps // blocks
    for block in range(blocks):
        start = prompt.shape[1] + block * block_length
        end = start + block_length
        transfers = get_num_transfer_tokens(x[:, start:end] == mask_id, local_steps)
        for step in range(local_steps):
            mask_index = x == mask_id
            candidates = mask_index.clone()
            candidates[:, :start] = False
            candidates[:, end:] = False
            logits = model(x, attention_mask=attention_mask).logits
            if logits_eos_inf and 126081 < logits.shape[-1]:
                logits[:, :, 126081] = -torch.inf
            predicted = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            confidence_logits = logits.float()
            if eos_guard and 126081 < confidence_logits.shape[-1]:
                confidence_logits[:, :, 126081] = -torch.inf
            for token_id in confidence_suppress_tokens:
                if token_id < confidence_logits.shape[-1]:
                    confidence_logits[:, :, token_id] = -torch.inf
            confidence = F.softmax(confidence_logits, dim=-1).gather(
                -1, predicted.unsqueeze(-1)).squeeze(-1)
            confidence[:, end:] = -np.inf
            predicted = torch.where(mask_index, predicted, x)
            confidence = torch.where(mask_index, confidence, -np.inf)
            for row in range(x.shape[0]):
                k = int(transfers[row, step].item())
                if k:
                    selected = torch.topk(confidence[row], k=k).indices
                    x[row, selected] = predicted[row, selected]
    if bool((x[:, prompt.shape[1]:] == mask_id).any()):
        raise RuntimeError("official sampler ended with unresolved masks")
    return x
