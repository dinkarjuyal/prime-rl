from typing import Any

import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from prime_rl.trainer.rl.config import LossConfig


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def selective_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def compute_entropy(shifted_logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq"]:
    with torch.no_grad():
        pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
        entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)
    return entropy


@jaxtyped(typechecker=typechecker)
def shift_logits(logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a zero logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    batch, seq, vocab = logits.shape
    logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    zeros = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)  # (batch, 1, vocab)
    logits = torch.cat([zeros, logits], dim=1)  # (batch, seq, vocab)
    return logits


def compute_contrastive_loss_infonce(
    trainer_logprobs: Any,  # list of Float[Tensor, "seq_i"]
    inference_logprobs: Any,  # list of Float[Tensor, "seq_i"]
    advantages: Any,  # list of Float[Tensor, "seq_i"]
    loss_mask: Any,  # list of Float[Tensor, "seq_i"]
    temperature: float = 0.1,
) -> Float[Tensor, ""]:
    """
    Compute InfoNCE-style contrastive loss with implicit hard negative mining.
    Treats sequences with positive advantages as anchors and those with negative advantages as negatives.
    
    Args:
        trainer_logprobs: List of trainer log probabilities per sequence
        inference_logprobs: List of inference log probabilities per sequence  
        advantages: List of advantages per sequence
        loss_mask: List of loss masks per sequence
        temperature: Temperature for softmax
        
    Returns:
        Contrastive loss scalar
    """
    # Compute sequence-level log ratios (sum over tokens)
    seq_log_ratios = []
    seq_advantages = []
    
    for t_logp, i_logp, adv, mask in zip(trainer_logprobs, inference_logprobs, advantages, loss_mask):
        # Sum log prob ratios over masked tokens
        log_ratio = ((t_logp - i_logp) * mask).sum()
        seq_log_ratios.append(log_ratio)
        
        # Average advantage over masked tokens
        avg_adv = (adv * mask).sum() / torch.clamp_min(mask.sum(), 1)
        seq_advantages.append(avg_adv)
    
    seq_log_ratios = torch.stack(seq_log_ratios)  # (num_sequences,)
    seq_advantages = torch.stack(seq_advantages)  # (num_sequences,)
    
    # Separate positive (good) and negative (bad) sequences
    positive_mask = seq_advantages > 0
    
    if positive_mask.sum() == 0 or (~positive_mask).sum() == 0:
        # Need both positives and negatives for contrastive loss
        return torch.tensor(0.0, device=seq_log_ratios.device)
    
    positive_log_ratios = seq_log_ratios[positive_mask]
    negative_log_ratios = seq_log_ratios[~positive_mask]
    
    # InfoNCE: for each positive, contrast against all negatives
    # Loss = -log(exp(pos/T) / (exp(pos/T) + sum(exp(neg/T))))
    losses = []
    for pos_lr in positive_log_ratios:
        # Numerator: positive similarity
        pos_sim = pos_lr / temperature
        
        # Denominator: positive + all negatives
        neg_sims = negative_log_ratios / temperature
        denominator = torch.logsumexp(torch.cat([pos_sim.unsqueeze(0), neg_sims]), dim=0)
        
        loss = denominator - pos_sim
        losses.append(loss)
    
    return torch.stack(losses).mean()


def compute_contrastive_loss_dpo(
    trainer_logprobs: Any,  # list of Float[Tensor, "seq_i"]
    inference_logprobs: Any,  # list of Float[Tensor, "seq_i"]
    advantages: Any,  # list of Float[Tensor, "seq_i"]
    loss_mask: Any,  # list of Float[Tensor, "seq_i"]
    beta: float = 0.1,
) -> Float[Tensor, ""]:
    """
    Compute DPO-style pairwise contrastive loss with implicit hard negative mining.
    
    Args:
        trainer_logprobs: List of trainer log probabilities per sequence
        inference_logprobs: List of inference log probabilities per sequence
        advantages: List of advantages per sequence
        loss_mask: List of loss masks per sequence
        beta: Beta parameter for DPO loss
        
    Returns:
        Contrastive loss scalar
    """
    # Compute sequence-level log ratios
    seq_log_ratios = []
    seq_advantages = []
    
    for t_logp, i_logp, adv, mask in zip(trainer_logprobs, inference_logprobs, advantages, loss_mask):
        log_ratio = ((t_logp - i_logp) * mask).sum()
        seq_log_ratios.append(log_ratio)
        
        avg_adv = (adv * mask).sum() / torch.clamp_min(mask.sum(), 1)
        seq_advantages.append(avg_adv)
    
    seq_log_ratios = torch.stack(seq_log_ratios)
    seq_advantages = torch.stack(seq_advantages)
    
    # Separate positive and negative sequences
    positive_mask = seq_advantages > 0
    
    if positive_mask.sum() == 0 or (~positive_mask).sum() == 0:
        return torch.tensor(0.0, device=seq_log_ratios.device)
    
    positive_log_ratios = seq_log_ratios[positive_mask]
    negative_log_ratios = seq_log_ratios[~positive_mask]
    
    # DPO: for each positive, pair with hardest negative
    # Loss = -log(sigmoid(beta * (log_ratio_pos - log_ratio_neg)))
    losses = []
    for pos_lr in positive_log_ratios:
        # Find hardest negative (highest log ratio among negatives)
        hardest_neg_lr = negative_log_ratios.max()
        
        # DPO loss
        loss = -torch.nn.functional.logsigmoid(beta * (pos_lr - hardest_neg_lr))
        losses.append(loss)
    
    return torch.stack(losses).mean()


def compute_loss(
    trainer_logprobs: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    inference_logprobs: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    advantages: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    loss_mask: Any,  # list of Bool[Tensor, "seq_i"] with potentially different seq_i lengths
    loss_config: LossConfig,
    loss_scale: int,
    current_step: int = 0,
    max_steps: int | None = None,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    Args:
        trainer_logprobs: Log probabilities tensor for packed sequences
        inference_logprobs: Old log probabilities tensor for packed sequences
        advantages: Advantages tensor for packed sequences
        loss_mask: Loss mask tensor for packed sequences
        loss_config: Loss configuration object
        loss_scale: Scale factor to normalize the loss

    Returns:
        Tuple of (scaled_loss, aggregated_loss_tensors)
    """

    total_loss = 0
    total_mismatch_kl = []
    total_masked_mismatch_kl = []
    total_unmasked_mismatch_kl = []
    total_is_masked = []
    total_is_masked_low = []
    total_is_masked_high = []
    total_sequence_masked_low = []

    for trainer_logprobs, inference_logprobs, advantages, loss_mask in zip(
        trainer_logprobs, inference_logprobs, advantages, loss_mask
    ):
        log_importance_ratio = trainer_logprobs - inference_logprobs

        # Compute trainer-inference mismatch KL
        token_mismatch_kl = torch.exp(log_importance_ratio) - log_importance_ratio - 1

        if loss_config.ratio_type == "sequence":
            seq_log_importance_ratio = (log_importance_ratio[loss_mask]).sum()
            if loss_config.ratio_length_norm:
                seq_log_importance_ratio = seq_log_importance_ratio / torch.clamp_min(loss_mask.sum(), 1)
            log_importance_ratio = trainer_logprobs - trainer_logprobs.detach() + seq_log_importance_ratio.detach()
            log_importance_ratio = torch.clamp(log_importance_ratio, max=10.0)

        importance_ratio = torch.exp(log_importance_ratio)
        is_masked_low = importance_ratio < loss_config.mask_ratio_low
        is_masked_high = importance_ratio > loss_config.mask_ratio_high
        is_masked = is_masked_low | is_masked_high
        seq_min_ratio = importance_ratio.masked_fill(~loss_mask, torch.inf).min()
        seq_should_mask = seq_min_ratio < loss_config.sequence_mask_ratio_low
        is_masked = is_masked | seq_should_mask
        keep_mask = loss_mask & ~is_masked
        loss = (-importance_ratio * advantages)[keep_mask].sum()

        # Apply sequence-level normalization if configured
        if loss_config.ratio_type == "sequence":
            loss = loss / torch.clamp_min(loss_mask.sum(), 1)

        total_loss = total_loss + loss

        mismatch_kl = token_mismatch_kl[loss_mask].sum() / torch.clamp_min(loss_mask.sum(), 1)
        masked_mismatch_kl = token_mismatch_kl[loss_mask & is_masked].sum() / torch.clamp_min(
            (loss_mask & is_masked).sum(), 1
        )
        unmasked_mismatch_kl = token_mismatch_kl[keep_mask].sum() / torch.clamp_min(keep_mask.sum(), 1)

        # Aggregate loss tensors
        total_mismatch_kl.append(mismatch_kl)
        total_masked_mismatch_kl.append(masked_mismatch_kl)
        total_unmasked_mismatch_kl.append(unmasked_mismatch_kl)
        total_is_masked.append(is_masked[loss_mask].float())
        total_is_masked_low.append(is_masked_low[loss_mask].float())
        total_is_masked_high.append(is_masked_high[loss_mask].float())
        total_sequence_masked_low.append(seq_should_mask.float())

    # Apply loss scaling
    scaled_loss = total_loss / loss_scale

    # Add contrastive loss if enabled
    contrastive_loss = torch.tensor(0.0, device=scaled_loss.device)
    contrastive_weight = loss_config.contrastive_loss_weight
    
    # Apply linear schedule if configured
    if loss_config.contrastive_loss_weight_end is not None and max_steps is not None and max_steps > 0:
        progress = min(current_step / max_steps, 1.0)
        contrastive_weight = (
            loss_config.contrastive_loss_weight * (1 - progress) + 
            loss_config.contrastive_loss_weight_end * progress
        )
    
    if contrastive_weight > 0:
        if loss_config.contrastive_loss_type == "infonce":
            contrastive_loss = compute_contrastive_loss_infonce(
                trainer_logprobs, inference_logprobs, advantages, loss_mask,
                temperature=loss_config.contrastive_temperature
            )
        elif loss_config.contrastive_loss_type == "dpo":
            contrastive_loss = compute_contrastive_loss_dpo(
                trainer_logprobs, inference_logprobs, advantages, loss_mask,
                beta=loss_config.contrastive_beta
            )
        
        scaled_loss = scaled_loss + contrastive_weight * contrastive_loss

    return scaled_loss, {
        "mismatch_kl": torch.stack(total_mismatch_kl),
        "masked_mismatch_kl": torch.stack(total_masked_mismatch_kl),
        "unmasked_mismatch_kl": torch.stack(total_unmasked_mismatch_kl),
        "is_masked": torch.cat(total_is_masked),
        "is_masked_low": torch.cat(total_is_masked_low),
        "is_masked_high": torch.cat(total_is_masked_high),
        "sequence_masked_low": torch.stack(total_sequence_masked_low),
        "contrastive_loss": contrastive_loss.unsqueeze(0),  # Log contrastive loss
        "contrastive_weight": torch.tensor([contrastive_weight], device=scaled_loss.device),  # Log scheduled weight
    }
