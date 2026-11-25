"""Keep checkpoint files and W&B summaries associated with one training launch."""

import json
import math
from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint


class RunCheckpoint(ModelCheckpoint):
    """Publish saved checkpoint paths after Lightning's checkpoint hooks run.

    A resumed launch has its own directory and selects its best checkpoint from
    that launch's validation results. Optimizer and training-loop state still
    resume normally; checkpoint selection must not refer to another run's files.
    """

    def __init__(self, run_info, **kwargs):
        super().__init__(**kwargs)
        self.run_info = dict(run_info)

    def load_state_dict(self, state_dict):
        if state_dict.get("dirpath") == self.dirpath:
            super().load_state_dict(state_dict)

    def on_fit_start(self, trainer, pl_module):
        self._publish(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        self._publish(trainer)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        self._publish(trainer)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self._publish(trainer)

    def _publish(self, trainer):
        if not trainer.is_global_zero or trainer.fast_dev_run:
            return
        score = self.best_model_score
        score = float(score.item()) if score is not None else None
        if score is not None and not math.isfinite(score):
            score = None
        summary = {
            **self.run_info,
            "checkpoint_monitor": self.monitor,
            "checkpoint_mode": self.mode,
            "best_checkpoint_path": self.best_model_path or None,
            "last_checkpoint_path": self.last_model_path or None,
            "best_checkpoint_score": score,
        }
        path = Path(self.dirpath) / "run.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        if trainer.logger is not None:
            trainer.logger.experiment.summary.update(summary)
