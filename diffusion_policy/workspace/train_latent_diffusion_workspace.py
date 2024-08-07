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
from torch.utils.data import DataLoader
import copy
import random
import wandb
import zzy_utils
from typing import Union
import tqdm
import numpy as np
from pathlib import Path
from loguru import logger
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.latent_diffusion_policy import LatentDiffusionPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicImageDatamodule
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
import lightning as L
import zzy_utils
OmegaConf.register_new_resolver("eval", eval, replace=True)
torch.set_float32_matmul_precision('medium')
class TrainLatentDiffusionWorkspace(BaseWorkspace):
    # include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        timestamp = zzy_utils.get_current_run_timestamp()
        cfg.trainer.logger.name = timestamp
        save_path = Path(os.environ['DATA_ROOT']) / cfg.trainer.logger.project / timestamp
        cfg.trainer.logger.save_dir = save_path.as_posix()
        cfg.trainer.callbacks[0].dirpath = save_path.as_posix()
        zzy_utils.configure_default_logger(save_path)
        
        L.seed_everything(cfg.seed)

        self.model: LatentDiffusionPolicy = hydra.utils.instantiate(cfg.policy)
        self.trainer : L.Trainer = hydra.utils.instantiate(cfg.trainer)

        self.datamodule : Union[L.LightningDataModule, RobomimicImageDatamodule] = hydra.utils.instantiate(cfg.datamodule)

    def run(self):
        self.model.set_normalizer(self.datamodule.dataset.get_normalizer(), datamodule = self.datamodule)
        self.trainer.fit(self.model.pl_model, datamodule=self.datamodule, ckpt_path=self.cfg.resume_ckpt_path)
        saved_path = self.save_checkpoint()
        logger.info(f"saved checkpoint to {saved_path}")
    
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainLatentDiffusionWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
