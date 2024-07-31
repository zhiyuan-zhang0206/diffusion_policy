from typing import Dict
# import math
import torch
# import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from pathlib import Path
# import dill
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
# from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.conditional_unet1d_rold import DownsampleObsLDM
# from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
# from diffusion_policy.common.robomimic_config_util import get_robomimic_config
# from robomimic.algo import algo_factory
# from robomimic.algo.algo import PolicyAlgo
# import robomimic.utils.obs_utils as ObsUtils
# import robomimic.models.base_nets as rmbn
# import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from loguru import logger

class LatentDiffusionPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            ae_path:str,
        
            hidden_size:int=256,
            # latent_size:int=256,
            horizon:int=16,
            language_feature_dim:int=768,
            low_dim_feature_dim:int=None,
            n_layers:int=6,
            n_heads:int=8,
            dropout:float=0.1,

            lr:float=1e-4,
            warmup_steps:int=1000,
            num_training_steps:int=100000,
            use_lr_scheduler:bool=True,
            
            num_inference_timesteps:int=1000,
            num_train_timesteps:int= 100,
            beta_start:float = 0.0001,
            beta_end:float = 0.02,
            beta_schedule: str = "squaredcos_cap_v2",
            variance_type: str = "fixed_small", # 
            clip_sample: bool = True, # required when predict_epsilon=False
            prediction_type: str = "epsilon",
            ):
        super().__init__()

        self.normalizer = LinearNormalizer()
        self.pl_model = DownsampleObsLDM(
            ae_path=ae_path,

            hidden_size=hidden_size,
            horizon=horizon,
            language_feature_dim=language_feature_dim,
            low_dim_feature_dim=low_dim_feature_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,

            lr=lr,
            warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            use_lr_scheduler=use_lr_scheduler,
            num_inference_timesteps=num_inference_timesteps,
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            variance_type=variance_type,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
        )
        self.shape_meta = shape_meta

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        import zzy_utils
        zzy_utils.pretty_print(obs_dict)
        print()
        

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.pl_model.normalizer = self.normalizer

    def compute_loss(self, batch):
        logger.debug(f"compute loss called")
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss



