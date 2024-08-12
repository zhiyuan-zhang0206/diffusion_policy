import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

class EnvRunnerTestCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.is_global_zero:
            result = pl_module.workspace_run_env_runner()
            for k, v in result.items():
                trainer.logger.log_metrics({k: v}, step=trainer.global_step)
            return