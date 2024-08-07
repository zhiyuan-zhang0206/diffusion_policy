if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from typing import Union
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.ae_policy import AEPolicy
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicImageDatamodule
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicLowdimDatamodule
import lightning as L
import zzy_utils
from pathlib import Path
from loguru import logger
OmegaConf.register_new_resolver("eval", eval, replace=True)
torch.set_float32_matmul_precision('medium')
class TrainAEWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)


        timestamp = zzy_utils.get_current_run_timestamp()
        cfg.trainer.logger.name = timestamp
        save_path = Path(os.environ['DATA_ROOT']) / cfg.trainer.logger.project / timestamp
        cfg.trainer.logger.save_dir = save_path.as_posix()
        cfg.trainer.callbacks[0].dirpath = save_path.as_posix()
        zzy_utils.configure_default_logger(save_path)
        
        L.seed_everything(cfg.seed)
        self.model: AEPolicy = hydra.utils.instantiate(cfg.policy)
        self.trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

        self.datamodule : Union[RobomimicImageDatamodule, RobomimicLowdimDatamodule] = hydra.utils.instantiate(cfg.datamodule)

    def run(self):
        self.model.set_normalizer(self.datamodule.dataset.get_normalizer())
        self.trainer.fit(model = self.model.pl_model, datamodule = self.datamodule)
        save_path = self.save_checkpoint()
        logger.info(f'saved checkpoint to {save_path}')
        return

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainAEWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
