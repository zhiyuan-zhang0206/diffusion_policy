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
from diffusion_policy.workspace.train_ae_workspace import TrainAEWorkspace
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
import dill
from copy import deepcopy
import zzy_utils
OmegaConf.register_new_resolver("eval", eval, replace=True)
torch.set_float32_matmul_precision('medium')

def get_ae(ae_workspace_path:str):
    with open(ae_workspace_path, 'rb') as f:
        ae_payload = torch.load(f, pickle_module=dill)
    ae_workspace = hydra.utils.get_class(ae_payload['cfg']['_target_'])(ae_payload['cfg'])
    ae_workspace.load_payload(ae_payload)
    ae = ae_workspace.model.pl_model
    ae_workspace.close_env_runner()
    return ae
class TrainLatentDiffusionWorkspace(BaseWorkspace):
    # include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        L.seed_everything(cfg.seed)
        self.trainer : L.Trainer = hydra.utils.instantiate(cfg.trainer)
        self.model: LatentDiffusionPolicy = hydra.utils.instantiate(cfg.policy)
        # load ae workspace
        ae = get_ae(cfg.ae_workspace_path)
        self.model.load_ae(ae)
        self.model.pl_model.workspace_save_checkpoint = self.save_checkpoint
        self.model.pl_model.workspace_run_env_runner = self.run_env_runner
        self.datamodule : Union[L.LightningDataModule, RobomimicImageDatamodule] = hydra.utils.instantiate(cfg.datamodule)
        self.load_env_runner()
        if cfg.resume_ckpt_path is not None:
            with open(cfg.resume_ckpt_path, 'rb') as f:
                payload = torch.load(f, pickle_module=dill)
            self.load_payload(payload)
            self.close_env_runner()
            self.load_env_runner()

    def load_env_runner(self):
        cfg = deepcopy(self.cfg)
        cfg.task.env_runner['n_test_vis'] = 0
        cfg.task.env_runner['n_train_vis'] = 0
        cfg.task.env_runner['n_train'] = 50 if not zzy_utils.check_environ_dry_run() else 1
        cfg.task.env_runner['n_test'] = 50
        cfg.task.env_runner['n_envs'] = 10 if not zzy_utils.check_environ_dry_run() else 1
        if 'lift' in cfg.task_name:
            cfg.task.env_runner['max_steps'] = 128
        else:
            cfg.task.env_runner['max_steps'] = 256
        if 'mixed' in cfg.task_name:
            cfg.task.env_runner['n_envs'] = max(cfg.task.env_runner['n_envs'], 1)
            for c in cfg.task.env_runner.runners:
                c['output_dir'] = self.output_dir
                c['max_steps'] = cfg.task.env_runner['max_steps']
                c['n_envs'] = cfg.task.env_runner['n_envs']
                c['n_train'] = cfg.task.env_runner['n_train']
                c['n_test'] = cfg.task.env_runner['n_test']
                c['n_train_vis'] = cfg.task.env_runner['n_train_vis']
                c['n_test_vis'] = cfg.task.env_runner['n_test_vis']
        cfg.task.env_runner['output_dir'] = self.output_dir 
        self.env_runner = hydra.utils.instantiate(cfg.task.env_runner)
        self.env_runner.load_replay_buffer(self)
        self.env_runner.set_normalizer(self)
        return


    def close_env_runner(self):
        self.env_runner.close()
        return

    def run_env_runner(self):
        run_result = self.env_runner.run(self.model)
        if isinstance(run_result, list):
            result = {f'train/mean_score_{r["task_name"]}': r['train/mean_score'] for r in run_result}
            if 'test/mean_score' in run_result[0]:
                result.update({f'test/mean_score_{r["task_name"]}': r['test/mean_score'] for r in run_result})
        else:
            result = {'train/mean_score': run_result['train/mean_score']}
            if 'test/mean_score' in run_result:
                result.update({'test/mean_score': run_result['test/mean_score']})
        return result


    def run(self):
        self.model.set_normalizer(self.datamodule.dataset.get_normalizer(), datamodule = self.datamodule)
        self.trainer.fit(self.model.pl_model, datamodule=self.datamodule)
        self.close_env_runner()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainLatentDiffusionWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
