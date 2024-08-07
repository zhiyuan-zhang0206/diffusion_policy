from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from pathlib import Path
import dill
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils

import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.model.diffusion.cvae_rold import DownsampleCVAE
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from loguru import logger

class AEPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon: int, 
            hidden_size: int,
            latent_size: int,
            n_encoder_layers: int,
            n_decoder_layers: int,
            n_heads: int,
            dropout: float,
            kl_weight: float=0.001,
            sample_posterior: bool=False,
            lr: float=3e-4,
            weight_decay: float=0.0,
            warmup_steps: int=1000,
            use_cosine_lr: bool=True,
            # n_obs_steps,
            # crop_shape=(76, 76),
            ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        self.pl_model = DownsampleCVAE(
            action_dim=action_dim,
            hidden_size=hidden_size,
            latent_size=latent_size,
            horizon=horizon,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            n_heads=n_heads,
            dropout=dropout,
            kl_weight=kl_weight,
            sample_posterior=sample_posterior,
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            use_cosine_lr=use_cosine_lr,
        )
        
        self.normalizer = LinearNormalizer()
        
        # print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        # print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        self.pl_model.eval()
        forward_output = self.pl_model.forward(obs_dict, normalize_input=True, unnormalize_output=True)

        return {'action': forward_output['unnormalized_pred']} | forward_output

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.pl_model.set_normalizer(normalizer)
