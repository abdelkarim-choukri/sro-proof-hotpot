"""
SRO Training Loop (v2.1)

Orchestrates the two-stage training regime with:
  - Tau annealing (exponential decay, step-level)
  - K-sample ramping (step-function over first Stage B epoch)
  - Stage A → B transition (momentum reset + verifier LR warmup)
  - Loss magnitude monitoring (advisory gamma decay)
  - Gradient clipping safeguard on PairRouter during early Stage B

Reviewer audit targets:
  1. Surgical momentum reset: dictionary traversal in optimizer state
  2. K-ramping: epoch-fraction logic for step-function
  3. Stage A → B transition: exact location in loop execution

Dependencies:
  - sro_model.py (SROModel, SROBatch, SROOutput)
  - PyTorch, HuggingFace transformers
"""

import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from sro_model import SROModel, SROBatch, SROOutput  # uncomment in project

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Training Configuration
# ════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """All hyperparameters in one place. No magic numbers in the loop."""

    # ── Stage durations ────────────────────────────────────────────────
    stage_a_epochs: int = 2
    stage_b_epochs: int = 8
    # Total = 10 epochs

    # ── Optimizer ──────────────────────────────────────────────────────
    base_lr: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0      # Global gradient clipping

    # ── Tau schedule (exponential decay, step-level) ───────────────────
    tau_0: float = 1.0               # Starting temperature
    tau_min: float = 0.1             # Floor (never anneal to zero)
    # eta is computed dynamically so tau reaches ~0.2 at Stage B midpoint

    # ── K-sample ramping ───────────────────────────────────────────────
    K_max: int = 4                   # Final number of Gumbel samples

    # ── Loss weights (v2.1 defaults) ───────────────────────────────────
    alpha: float = 0.5               # L_select
    beta: float = 0.1                # L_qtype
    gamma: float = 0.5               # L_sent_sup

    # ── Stage A → B transition ─────────────────────────────────────────
    verifier_warmup_steps: int = 1000  # Linear LR warmup for verifier group
    router_clip_norm: float = 5.0      # Absolute grad clip on PairRouter, early Stage B
    router_clip_steps: int = 500       # How many Stage B steps to apply the clip

    # ── Monitoring ─────────────────────────────────────────────────────
    log_every_n_steps: int = 50
    gamma_decay_trigger_steps: int = 100  # Consecutive steps before advisory triggers
    gamma_decay_threshold: float = 0.5    # L_sent_sup / loss_total ratio

    # ── Paths ──────────────────────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # ── Hardware ───────────────────────────────────────────────────────
    batch_size: int = 8
    gradient_accumulation_steps: int = 1


# ════════════════════════════════════════════════════════════════════════
# Tau Schedule
# ════════════════════════════════════════════════════════════════════════

class TauScheduler:
    """
    Exponential decay tied to global training step (not epoch).
    tau_t = max(tau_min, tau_0 * eta^t)

    eta is computed so that tau reaches target_mid at the midpoint of Stage B.
    During Stage A, tau is fixed at tau_0 (no decay).
    """

    def __init__(
        self,
        tau_0: float,
        tau_min: float,
        stage_b_start_step: int,
        stage_b_total_steps: int,
        target_mid: float = 0.2,
    ):
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.stage_b_start_step = stage_b_start_step

        # Compute eta so tau = target_mid at Stage B midpoint
        mid_step = stage_b_total_steps // 2
        if mid_step > 0 and target_mid > 0:
            self.eta = (target_mid / tau_0) ** (1.0 / mid_step)
        else:
            self.eta = 0.999

    def get_tau(self, global_step: int) -> float:
        """
        Returns current tau value.

        During Stage A (global_step < stage_b_start_step): fixed at tau_0.
        During Stage B: exponential decay from tau_0 toward tau_min.
        """
        if global_step < self.stage_b_start_step:
            return self.tau_0

        steps_into_b = global_step - self.stage_b_start_step
        tau = self.tau_0 * (self.eta ** steps_into_b)
        return max(self.tau_min, tau)


# ════════════════════════════════════════════════════════════════════════
# K-Ramping Schedule
# ════════════════════════════════════════════════════════════════════════

class KRampScheduler:
    """
    ── AUDIT POINT 2: K-ramping epoch-fraction logic ────────────────

    Step function that ramps K from 1 to K_max over the first epoch
    of Stage B, in 25% increments.

    Given K_max=4:
      First  25% of first Stage B epoch: K=1
      25%-50%:                            K=2
      50%-75%:                            K=3
      75%+:                               K=4

    After the first Stage B epoch, K stays at K_max permanently.
    """

    def __init__(self, K_max: int, first_epoch_steps: int):
        self.K_max = K_max
        self.first_epoch_steps = first_epoch_steps

    def get_K(self, steps_into_stage_b: int) -> int:
        if steps_into_stage_b >= self.first_epoch_steps:
            # Past first epoch: always K_max
            return self.K_max

        # Fraction of first epoch completed
        fraction = steps_into_stage_b / max(self.first_epoch_steps, 1)

        # Step function: 25% increments
        # fraction 0.00-0.25 → K=1
        # fraction 0.25-0.50 → K=2
        # fraction 0.50-0.75 → K=3
        # fraction 0.75-1.00 → K=4
        K = min(self.K_max, 1 + int(fraction * self.K_max))
        return max(1, K)


# ════════════════════════════════════════════════════════════════════════
# Verifier LR Warmup Scheduler
# ════════════════════════════════════════════════════════════════════════

class VerifierWarmupScheduler:
    """
    Linear LR warmup for the verifier parameter group during the first
    N steps of Stage B. Multiplies the verifier group's LR by a ramp
    from 0.0 to 1.0.

    Other parameter groups are unaffected.
    """

    def __init__(self, optimizer, verifier_group_idx: int, warmup_steps: int):
        self.optimizer = optimizer
        self.verifier_group_idx = verifier_group_idx
        self.warmup_steps = warmup_steps
        self.base_lr = optimizer.param_groups[verifier_group_idx]["lr"]
        self._active = False

    def activate(self):
        """Call at the exact Stage A → B transition."""
        self._active = True
        # Start at 0 LR
        self.optimizer.param_groups[self.verifier_group_idx]["lr"] = 0.0

    def step(self, steps_into_stage_b: int):
        if not self._active:
            return

        if steps_into_stage_b >= self.warmup_steps:
            # Warmup complete: restore full LR
            self.optimizer.param_groups[self.verifier_group_idx]["lr"] = self.base_lr
            return

        # Linear ramp from 0 to base_lr
        fraction = steps_into_stage_b / self.warmup_steps
        self.optimizer.param_groups[self.verifier_group_idx]["lr"] = self.base_lr * fraction


# ════════════════════════════════════════════════════════════════════════
# Surgical Momentum Reset
# ════════════════════════════════════════════════════════════════════════

def reset_momentum_for_params(optimizer: torch.optim.Optimizer, param_ids: set):
    """
    ── AUDIT POINT 1: Surgical momentum reset ──────────────────────

    Zeroes the Adam momentum buffers (exp_avg and exp_avg_sq) for a
    specific set of parameters, identified by their id().

    This is called at the exact Stage A → B transition for PairRouter
    parameters ONLY. Encoder momentum is preserved.

    Dictionary traversal:
        optimizer.state is a dict mapping parameter tensors to their
        optimizer state dicts. Each state dict contains:
        - "step": number of optimizer steps for this param
        - "exp_avg": first moment (momentum)
        - "exp_avg_sq": second moment (adaptive LR)

    We zero exp_avg and exp_avg_sq but preserve "step" so the bias
    correction denominator remains valid.
    """
    reset_count = 0
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if id(param) in param_ids and param in optimizer.state:
                state = optimizer.state[param]
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()
                reset_count += 1

    logger.info(f"Momentum reset: zeroed exp_avg/exp_avg_sq for {reset_count} parameters")
    return reset_count


# ════════════════════════════════════════════════════════════════════════
# Loss Magnitude Monitor
# ════════════════════════════════════════════════════════════════════════

class LossMonitor:
    """
    Tracks unweighted loss magnitudes and monitors for gamma dominance.

    Advisory mode (first run): logs warnings but does not auto-decay gamma.
    """

    def __init__(self, gamma_threshold: float, trigger_steps: int, advisory: bool = True):
        self.gamma_threshold = gamma_threshold
        self.trigger_steps = trigger_steps
        self.advisory = advisory
        self.consecutive_dominant_steps = 0

    def update(
        self,
        loss_verify: Optional[float],
        loss_select: float,
        loss_qtype: float,
        loss_sent: float,
        loss_total: float,
        gamma: float,
        step: int,
    ) -> Optional[float]:
        """
        Returns new gamma if decay is triggered, else None.
        """
        if loss_total == 0:
            return None

        sent_contribution = gamma * loss_sent
        ratio = sent_contribution / loss_total

        if ratio > self.gamma_threshold:
            self.consecutive_dominant_steps += 1
        else:
            self.consecutive_dominant_steps = 0

        if self.consecutive_dominant_steps >= self.trigger_steps:
            new_gamma = gamma / 2.0
            if self.advisory:
                logger.warning(
                    f"[ADVISORY] Step {step}: L_sent_sup dominates at {ratio:.2%} of "
                    f"loss_total for {self.consecutive_dominant_steps} consecutive steps. "
                    f"Recommended gamma: {gamma} → {new_gamma}. (Advisory mode: not applied.)"
                )
                return None
            else:
                logger.warning(
                    f"Step {step}: Decaying gamma from {gamma} to {new_gamma} "
                    f"(L_sent_sup at {ratio:.2%} of loss_total for "
                    f"{self.consecutive_dominant_steps} steps)."
                )
                self.consecutive_dominant_steps = 0
                return new_gamma

        return None


# ════════════════════════════════════════════════════════════════════════
# Main Training Loop
# ════════════════════════════════════════════════════════════════════════

def train(
    model,       # SROModel instance
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
):
    """
    Full training loop implementing the v2.1 two-stage regime.

    ── AUDIT POINT 3: Stage A → B transition location ──────────────
    The transition occurs at line marked [TRANSITION] below, between
    the Stage A epoch loop and the Stage B epoch loop. It is NOT
    inside a batch loop — it executes exactly once.
    """

    device = next(model.parameters()).device
    total_epochs = config.stage_a_epochs + config.stage_b_epochs

    # ══════════════════════════════════════════════════════════════════
    # SETUP: Optimizer with parameter groups
    # ══════════════════════════════════════════════════════════════════

    # Identify parameter groups
    router_params = []
    encoder_params = []
    verifier_params = []
    special_params = []   # qtype prefix embeddings (10x LR)
    other_params = []

    router_param_ids = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "pair_router" in name:
            router_params.append(param)
            router_param_ids.add(id(param))
        elif "e_bridge" in name or "e_compare" in name:
            special_params.append(param)
        elif "verifier" in name:
            verifier_params.append(param)
        elif "paragraph_encoder.encoder" in name:
            encoder_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": config.base_lr, "weight_decay": config.weight_decay},
        {"params": special_params, "lr": config.base_lr * 10, "weight_decay": 0.0},
        {"params": router_params, "lr": config.base_lr * 3, "weight_decay": config.weight_decay},
        {"params": verifier_params, "lr": config.base_lr * 3, "weight_decay": config.weight_decay},
        {"params": other_params, "lr": config.base_lr * 3, "weight_decay": config.weight_decay},
    ])

    # Group indices for targeted operations
    ENCODER_GROUP = 0
    SPECIAL_GROUP = 1
    ROUTER_GROUP = 2
    VERIFIER_GROUP = 3
    OTHER_GROUP = 4

    # ══════════════════════════════════════════════════════════════════
    # SETUP: Schedulers
    # ══════════════════════════════════════════════════════════════════

    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    stage_a_total_steps = config.stage_a_epochs * steps_per_epoch
    stage_b_total_steps = config.stage_b_epochs * steps_per_epoch

    tau_scheduler = TauScheduler(
        tau_0=config.tau_0,
        tau_min=config.tau_min,
        stage_b_start_step=stage_a_total_steps,
        stage_b_total_steps=stage_b_total_steps,
    )

    k_scheduler = KRampScheduler(
        K_max=config.K_max,
        first_epoch_steps=steps_per_epoch,
    )

    verifier_warmup = VerifierWarmupScheduler(
        optimizer=optimizer,
        verifier_group_idx=VERIFIER_GROUP,
        warmup_steps=config.verifier_warmup_steps,
    )

    loss_monitor = LossMonitor(
        gamma_threshold=config.gamma_decay_threshold,
        trigger_steps=config.gamma_decay_trigger_steps,
        advisory=True,  # First run: observe only
    )

    # ══════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════

    global_step = 0
    best_val_metric = 0.0

    # ────────────────────────────────────────────────────────────────
    # STAGE A: Warmup (encoder/router from L_select+L_qtype+L_sent_sup,
    #          verifier pre-warmed on detached inputs)
    # ────────────────────────────────────────────────────────────────

    logger.info(f"{'='*60}")
    logger.info(f"STAGE A: Warmup ({config.stage_a_epochs} epochs)")
    logger.info(f"  tau: fixed at {config.tau_0}")
    logger.info(f"  K: 1 (verifier on detached inputs)")
    logger.info(f"  Active losses: L_select + L_qtype + L_sent_sup + L_verify(detached)")
    logger.info(f"{'='*60}")

    model.set_stage_a()
    model.train()

    for epoch in range(config.stage_a_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, device)

            tau = tau_scheduler.get_tau(global_step)  # Fixed at 1.0 during Stage A

            output = model(batch, tau=tau)

            loss = output.loss_total / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_loss += output.loss_total.item()

                if global_step % config.log_every_n_steps == 0:
                    _log_step(global_step, "A", epoch, tau, 1, output)

        logger.info(
            f"Stage A Epoch {epoch+1}/{config.stage_a_epochs} complete. "
            f"Avg loss: {epoch_loss / steps_per_epoch:.4f}"
        )

    # ────────────────────────────────────────────────────────────────
    # [TRANSITION] Stage A → Stage B
    #
    # ── AUDIT POINT 3: This is the exact transition location. ──────
    #
    # Executes exactly ONCE between the two epoch loops.
    # Three operations in strict order:
    #   1. Switch model to Stage B
    #   2. Reset PairRouter momentum (surgical)
    #   3. Activate verifier LR warmup
    # ────────────────────────────────────────────────────────────────

    logger.info(f"\n{'='*60}")
    logger.info(f"TRANSITION: Stage A → Stage B at global_step={global_step}")
    logger.info(f"{'='*60}")

    # 1. Switch model to Stage B (K=1 initially, will be ramped)
    model.set_stage_b(K=1)

    # 2. ── AUDIT POINT 1: Surgical momentum reset ─────────────────
    #    Zero exp_avg and exp_avg_sq for PairRouter params only.
    #    Encoder momentum is preserved (benefits from continuity).
    reset_count = reset_momentum_for_params(optimizer, router_param_ids)
    logger.info(f"  PairRouter momentum reset: {reset_count} parameters")

    # 3. Activate verifier LR warmup (starts at 0.0, ramps to base_lr)
    verifier_warmup.activate()
    logger.info(f"  Verifier LR warmup activated: 0.0 → base_lr over {config.verifier_warmup_steps} steps")

    steps_into_stage_b = 0

    # ────────────────────────────────────────────────────────────────
    # STAGE B: Full training (all 4 losses, end-to-end gradient)
    # ────────────────────────────────────────────────────────────────

    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE B: Full training ({config.stage_b_epochs} epochs)")
    logger.info(f"  tau: {config.tau_0} → {config.tau_min} (exponential decay)")
    logger.info(f"  K: 1 → {config.K_max} (ramped over first epoch)")
    logger.info(f"  Active losses: ALL (L_verify end-to-end)")
    logger.info(f"{'='*60}")

    for epoch in range(config.stage_b_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, device)

            # ── Tau: exponential decay ─────────────────────────────
            tau = tau_scheduler.get_tau(global_step)

            # ── K: step-function ramp ──────────────────────────────
            # AUDIT POINT 2: K-ramping based on fraction of first
            # Stage B epoch completed.
            K = k_scheduler.get_K(steps_into_stage_b)
            model.set_stage_b(K=K)

            # ── Forward + backward ─────────────────────────────────
            output = model(batch, tau=tau)

            loss = output.loss_total / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # ── Global gradient clipping ───────────────────────
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )

                # ── PairRouter-specific gradient clipping ──────────
                # Applied only for the first N steps of Stage B.
                # Conservative absolute clip catches transition shock
                # that the momentum reset alone might not absorb.
                if steps_into_stage_b < config.router_clip_steps:
                    torch.nn.utils.clip_grad_norm_(
                        router_params, config.router_clip_norm
                    )

                # ── Optimizer step ─────────────────────────────────
                optimizer.step()
                optimizer.zero_grad()

                # ── Verifier LR warmup step ────────────────────────
                verifier_warmup.step(steps_into_stage_b)

                global_step += 1
                steps_into_stage_b += 1
                epoch_loss += output.loss_total.item()

                # ── Loss monitoring ────────────────────────────────
                if global_step % config.log_every_n_steps == 0:
                    _log_step(global_step, "B", epoch + config.stage_a_epochs, tau, K, output)

                    # Check for gamma dominance (advisory)
                    new_gamma = loss_monitor.update(
                        loss_verify=output.loss_verify.item() if output.loss_verify is not None else None,
                        loss_select=output.loss_select.item(),
                        loss_qtype=output.loss_qtype.item(),
                        loss_sent=output.loss_sent.item(),
                        loss_total=output.loss_total.item(),
                        gamma=model.gamma,
                        step=global_step,
                    )
                    # In non-advisory mode, apply the decay:
                    # if new_gamma is not None:
                    #     model.gamma = new_gamma

        # ── End-of-epoch: validation + checkpoint ──────────────────
        val_metric = _validate(model, val_loader, device, tau)
        logger.info(
            f"Stage B Epoch {epoch+1}/{config.stage_b_epochs} | "
            f"Train loss: {epoch_loss / steps_per_epoch:.4f} | "
            f"Val metric: {val_metric:.4f} | "
            f"tau: {tau:.4f} | K: {K} | "
            f"Verifier LR: {optimizer.param_groups[VERIFIER_GROUP]['lr']:.2e}"
        )

        # Save best checkpoint
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            _save_checkpoint(model, optimizer, global_step, epoch, config)
            logger.info(f"  → New best: {val_metric:.4f}")

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"  Total steps: {global_step}")
    logger.info(f"  Best val metric: {best_val_metric:.4f}")
    logger.info(f"{'='*60}")


# ════════════════════════════════════════════════════════════════════════
# Helper Functions
# ════════════════════════════════════════════════════════════════════════

def _move_batch_to_device(batch, device):
    """Move all tensor fields to the target device."""
    for field_name in batch.__dataclass_fields__:
        val = getattr(batch, field_name)
        if isinstance(val, torch.Tensor):
            setattr(batch, field_name, val.to(device))
    return batch


def _log_step(global_step, stage, epoch, tau, K, output):
    """Log training metrics for a single step."""
    parts = [
        f"Step {global_step:>6d} | Stage {stage} | Epoch {epoch+1}",
        f"tau={tau:.3f} | K={K}",
        f"total={output.loss_total.item():.4f}",
        f"select={output.loss_select.item():.4f}",
        f"qtype={output.loss_qtype.item():.4f}",
        f"sent={output.loss_sent.item():.4f}",
    ]
    if output.loss_verify is not None:
        parts.append(f"verify={output.loss_verify.item():.4f}")
    logger.info(" | ".join(parts))


def _validate(model, val_loader, device, tau):
    """
    Run validation and return the primary metric.

    Uses deterministic argmax (no Gumbel noise) via model.eval().
    Returns EM-proxy metric for now; replaced with full HotpotQA eval
    in the evaluation harness.
    """
    model.eval()
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch_to_device(batch, device)
            output = model(batch, tau=tau)

            if output.verifier_logits is not None:
                preds = (output.verifier_logits > 0).float()
                total_correct += (preds == batch.judge_label).sum().item()
                total_count += batch.judge_label.size(0)

    model.train()

    if total_count == 0:
        return 0.0
    return total_correct / total_count


def _save_checkpoint(model, optimizer, global_step, epoch, config):
    """Save model + optimizer state for resuming."""
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
    }, ckpt_dir / f"checkpoint_step_{global_step}.pt")


# ════════════════════════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # This would be called from a launcher script that sets up the model,
    # dataloaders, and config. Example:
    #
    # from transformers import AutoModel
    # from sro_model import SROModel
    #
    # encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
    # model = SROModel(encoder, hidden_dim=768).to("cuda")
    #
    # train_loader = ...  # DataLoader yielding SROBatch instances
    # val_loader = ...
    #
    # config = TrainingConfig(
    #     stage_a_epochs=2,
    #     stage_b_epochs=8,
    #     base_lr=2e-5,
    #     K_max=4,
    # )
    #
    # train(model, train_loader, val_loader, config)
    print("Training loop module loaded. Call train() from a launcher script.")
