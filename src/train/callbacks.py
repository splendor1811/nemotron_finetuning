"""Custom training callbacks for SFT pipeline."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class WandbMetricsCallback(TrainerCallback):
    """Log additional metrics to wandb during training.

    Tracks learning rate, epoch progress, and any custom metrics
    that are not logged by the default Trainer integration.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return

        try:
            import wandb

            if wandb.run is None:
                return

            # Log additional computed metrics
            extra = {}
            if "loss" in logs and "grad_norm" in logs:
                extra["loss_x_grad_norm"] = logs["loss"] * logs["grad_norm"]
            if state.epoch is not None:
                extra["epoch_progress"] = state.epoch

            if extra:
                wandb.log(extra, step=state.global_step)
        except ImportError:
            pass


class BestCheckpointCallback(TrainerCallback):
    """Track best eval loss and save the best adapter checkpoint.

    Monitors eval_loss and copies the best checkpoint to a dedicated
    directory so it is easy to find after training.
    """

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.best_eval_loss: Optional[float] = None
        self.best_step: Optional[int] = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_step = state.global_step

            logger.info(
                f"New best eval_loss: {eval_loss:.4f} at step {state.global_step}"
            )

            # Copy current checkpoint to best directory
            best_dir = self.output_dir / "best_adapter"
            best_dir.mkdir(parents=True, exist_ok=True)

            # Find the latest checkpoint directory
            checkpoint_dir = self.output_dir / f"checkpoint-{state.global_step}"
            if checkpoint_dir.exists():
                # Remove old best and copy new
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(checkpoint_dir, best_dir)
                logger.info(f"Saved best adapter to {best_dir}")
            else:
                logger.warning(
                    f"Checkpoint dir {checkpoint_dir} not found; "
                    f"best adapter will be saved at end of training."
                )

            # Log to wandb
            try:
                import wandb

                if wandb.run is not None:
                    wandb.run.summary["best_eval_loss"] = eval_loss
                    wandb.run.summary["best_step"] = state.global_step
            except ImportError:
                pass

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self.best_eval_loss is not None:
            logger.info(
                f"Training complete. Best eval_loss: {self.best_eval_loss:.4f} "
                f"at step {self.best_step}"
            )
        else:
            logger.info("Training complete. No evaluation was performed.")
