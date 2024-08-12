from lightning.pytorch.callbacks import ModelCheckpoint
from torch.nn.parallel import DistributedDataParallel
def unwrap_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model

class WorkspaceCheckpointCallback(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(WorkspaceCheckpointCallback, self).__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer, filepath):
        # checkpoint = super().on_save_checkpoint(trainer, pl_module)
        model = unwrap_model(trainer.model)
        model.workspace_save_checkpoint(filepath)
