"""LoRA fine-tune. Ports train_v2.py.

Strategy B v3 (the winner): r=16, alpha=32, target q/k/v/o, 3 epochs,
lr=1e-4, bf16. Validated at 64.7% on MTSamples.

Adds production resilience over train_v2.py:
- Checkpointing every `save_steps` (default 50), keeping `save_total_limit` of 3
  most recent. Each save commits the volume so a dead container's checkpoints
  survive — and a respawn picks up where it left off.
- `WebhookProgressCallback` reports per-step loss to the orchestrator, which
  forwards to /api/webhooks/progress.
- Optional `on_checkpoint` callback per save (used by orchestrator for the
  `/api/webhooks/checkpoint` event row).
"""

from __future__ import annotations

import os
from typing import Callable

from transformers import TrainerCallback, TrainerControl, TrainerState


def format_example(ex: dict[str, str]) -> dict[str, str]:
    text = (
        f"### Instruction: {ex['instruction']}\n"
        f"### Input: {ex['input']}\n"
        f"### Output: {ex['output']}"
    )
    return {"text": text}


class WebhookProgressCallback(TrainerCallback):
    """Reports SFT progress to a callback every N steps.

    Maps trainer step → 0-100 percent within an outer training-phase window
    (default 70-95% of overall pipeline progress).
    """

    def __init__(
        self,
        on_progress: Callable[[float, str], None],
        *,
        start_pct: float = 70.0,
        end_pct: float = 95.0,
        every: int = 20,
    ):
        self.on_progress = on_progress
        self.start_pct = start_pct
        self.end_pct = end_pct
        self.every = every

    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **_,
    ):
        if logs is None:
            return
        loss = logs.get("loss")
        if state.global_step % self.every != 0:
            return
        max_steps = max(state.max_steps or 1, 1)
        frac = min(state.global_step / max_steps, 1.0)
        pct = self.start_pct + (self.end_pct - self.start_pct) * frac
        msg = (
            f"Step {state.global_step}/{state.max_steps} · loss {loss:.2f}"
            if loss is not None
            else f"Step {state.global_step}/{state.max_steps}"
        )
        try:
            self.on_progress(pct, msg)
        except Exception:
            pass


class CheckpointCommitCallback(TrainerCallback):
    """Commits the Modal volume on every checkpoint save and notifies callers.

    Without volume.commit(), HF Trainer's checkpoint dirs only live in the
    container's local writable layer — a dead container loses them. Calling
    .commit() inside on_save promotes them to the durable volume so the next
    spawn (after a Modal retry or watchdog respawn) sees them.
    """

    def __init__(
        self,
        commit: Callable[[], None],
        on_checkpoint: Callable[[int, str], None] | None = None,
    ):
        self.commit = commit
        self.on_checkpoint = on_checkpoint

    def on_save(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **_,
    ):
        try:
            self.commit()
        except Exception:
            # If commit fails the container will retry — keep training.
            pass

        if self.on_checkpoint is not None:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            try:
                self.on_checkpoint(int(state.global_step), ckpt_path)
            except Exception:
                pass


def train_lora(
    model,
    tokenizer,
    training_data: list[dict[str, str]],
    output_dir: str,
    *,
    on_progress: Callable[[float, str], None] | None = None,
    commit_volume: Callable[[], None] | None = None,
    on_checkpoint: Callable[[int, str], None] | None = None,
    resume_from_checkpoint: str | None = None,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 50,
    save_total_limit: int = 3,
) -> str:
    """Fine-tune `model` with LoRA on `training_data`. Saves adapter to `output_dir`.

    `output_dir` is the FINAL adapter destination. Checkpoints are written to
    `<output_dir>/../checkpoints/` (so they sit alongside in /models/<id>/checkpoints/).

    Returns the output directory.
    """
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)

    dataset = Dataset.from_list(training_data).map(format_example)

    callbacks: list[TrainerCallback] = []
    if on_progress is not None:
        callbacks.append(WebhookProgressCallback(on_progress))
    if commit_volume is not None:
        callbacks.append(
            CheckpointCommitCallback(commit_volume, on_checkpoint=on_checkpoint)
        )

    # Checkpoints sit beside the final adapter under /models/<id>/checkpoints/.
    # output_dir for SFTConfig is the *checkpoint root* — final save_pretrained
    # writes the adapter to a different path below.
    parent = os.path.dirname(output_dir.rstrip("/"))
    checkpoint_root = os.path.join(parent or output_dir, "checkpoints")
    os.makedirs(checkpoint_root, exist_ok=True)

    trainer = SFTTrainer(
        model=peft_model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            output_dir=checkpoint_root,
            logging_steps=20,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            bf16=True,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            dataset_text_field="text",
            gradient_checkpointing=False,
            report_to=[],
        ),
        callbacks=callbacks,
    )
    # `resume_from_checkpoint=False` means "fresh start"; passing a path or
    # `True` resumes from latest in output_dir.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint or False)

    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if commit_volume is not None:
        try:
            commit_volume()
        except Exception:
            pass
    return output_dir
