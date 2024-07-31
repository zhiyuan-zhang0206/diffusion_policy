import copy
from typing import Any
from collections import OrderedDict
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.schedulers import DDPMScheduler, DDIMScheduler

import lightning.pytorch as pl
from diffusion_policy.model.common.normalizer import LinearNormalizer
# from RoLD.models.common import SinusoidalPosEmb, get_pe, WrappedTransformerEncoder, WrappedTransformerDecoder, ResBottleneck, ImageAdapter
# from RoLD.models.autoencoder.downsample_cvae import DownsampleCVAE
from diffusion_policy.model.diffusion.cvae_rold import DownsampleCVAE, get_pe, WrappedTransformerEncoder, WrappedTransformerDecoder, ResBottleneck, ImageAdapter
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.vision.r3m import R3MImageEncoderWrapper
from diffusion_policy.model.language.DistilBERT_utils import DistilBERTWrapper

class DownsampleObsLDM(pl.LightningModule):
    def __init__(
        self,
        ae_path:str,
        
        hidden_size:int=256,
        horizon:int=16,
        language_feature_dim:int=768,
        image_feature_dim:int=512,
        low_dim_feature_dim:int=None,
        n_layers:int=6,
        n_heads:int=8,
        dropout:float=0.1,

        lr:float=1e-4,
        warmup_steps:int=1000,
        num_training_steps:int=100000,
        
        num_inference_timesteps:int=1000,
        num_train_timesteps:int= 100,
        beta_start:float = 0.0001,
        beta_end:float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        variance_type: str = "fixed_small", # 
        clip_sample: bool = True, # required when predict_epsilon=False
        prediction_type: str = "epsilon",
        # mode,
        # all_config=None
    ) -> None:
        super().__init__()
        # three modes: pretraining, finetuning, inference

        # ckpt_path = model_kwargs.ckpt_path
        # if ckpt_path is not None:
        #     assert mode == 'finetuning' or mode == 'inference'
        #     ckpt = torch.load(ckpt_path)
        #     hyper_params = copy.deepcopy(ckpt['hyper_parameters'])
        #     low_dim_feature_dim = model_kwargs.low_dim_feature_dim
        #     model_kwargs = hyper_params['model_kwargs']
        #     model_kwargs.low_dim_feature_dim = low_dim_feature_dim

        # initialize model
        self.ae_path = ae_path
        self.hidden_size = hidden_size
        # self.latent_size = latent_size
        self.horizon = horizon
        self.image_feature_dim = image_feature_dim
        self.language_feature_dim = language_feature_dim
        self.low_dim_feature_dim = low_dim_feature_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.num_inference_timesteps = num_inference_timesteps
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        
        self.save_hyperparameters()
        self.normalizer: LinearNormalizer = None
        # initialize autoencoder
        self.autoencoder = DownsampleCVAE.load_from_checkpoint(ae_path)
        self.latent_size = latent_size = self.autoencoder.latent_size
        # ae_ckpt = torch.load(ae_path)
        # ae_config = ae_ckpt['hyper_parameters']
        # if mode == 'finetuning' and ae_config['mode'] == 'pretraining':
        #     raise ValueError('you should load a finetuned AE during finetuning ldm')
        # ae_config['model_kwargs']['ckpt_path'] = ae_ckpt_path
        # autoencoder = DownsampleCVAE(**ae_config)  # init includes loading ckpt
        # freeze autoencoder
        # for p in autoencoder.parameters():
        #     p.requires_grad = False
        # del ae_ckpt

        # self.hidden_size = hidden_size = model_kwargs['hidden_size'] = autoencoder.hidden_size
        # self.latent_size = latent_size = model_kwargs['latent_size'] = autoencoder.latent_size
        # self.horizon = horizon = model_kwargs['horizon']

        self.time_emb = SinusoidalPosEmb(dim=hidden_size)
        self.register_buffer(
            'pe', get_pe(hidden_size=hidden_size, max_len=horizon*2)
        )

        self.z_up = nn.Linear(latent_size, hidden_size)
        self.denoiser = WrappedTransformerEncoder(
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.z_down = nn.Linear(hidden_size, latent_size)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            variance_type=variance_type,
            clip_sample=clip_sample,
            prediction_type=prediction_type
        )


        self.image_encoder = R3MImageEncoderWrapper()
        self.image_emb = nn.Linear(in_features=image_feature_dim, out_features=hidden_size)
        # self.language_encoder = DistilBERTWrapper()
        self.language_emb = nn.Linear(in_features=language_feature_dim, out_features=hidden_size)
        # if ckpt_path is not None:
        #     # attach obs, then load real params
        #     # copy pretrained img_emb
        #     if hasattr(autoencoder, 'img_emb'):
        #         self.img_emb = copy.deepcopy(autoencoder.img_emb)
        #     else:
        #         self.img_emb = ResBottleneck(hidden_size=hidden_size)
        #     for p in self.img_emb.parameters():
        #         p.requires_grad = True
        #     if model_kwargs.get('low_dim_feature_dim') is not None:
        #         assert mode == 'finetuning' or mode == 'inference'
        #         # copy pretrained low_dim_emb
        #         if hasattr(autoencoder, 'low_dim_emb'):
        #             self.low_dim_emb = copy.deepcopy(autoencoder.low_dim_emb)
        #         else:
        #             self.low_dim_emb = nn.Linear(model_kwargs['low_dim_feature_dim'], hidden_size)
        #         for p in self.low_dim_emb.parameters():
        #             p.requires_grad = True
        #     else:
        #         assert mode == 'pretraining'
        #         self.low_dim_emb = None

        #     self.load_state_dict(state_dict=ckpt['state_dict'], strict=False)  # only load the ldm part
        #     del ckpt
        #     print(f'WARNING: ignoring LDM config, LDM loaded from {ckpt_path}')
        # else:
        #     # apply init on other params
        #     self.apply(self._init_weights)
        #     # then attach obs 
        #     # copy pretrained img_emb
        #     if hasattr(autoencoder, 'img_emb'):
        #         self.img_emb = copy.deepcopy(autoencoder.img_emb)
        #     else:
        #         self.img_emb = ResBottleneck(hidden_size=hidden_size)
        #     for p in self.img_emb.parameters():
        #         p.requires_grad = True
        #     if model_kwargs.get('low_dim_feature_dim') is not None:
        #         assert mode == 'finetuning' or mode == 'inference'
        #         # copy pretrained low_dim_emb
        #         if hasattr(autoencoder, 'low_dim_emb'):
        #             self.low_dim_emb = copy.deepcopy(autoencoder.low_dim_emb)
        #         else:
        #             self.low_dim_emb = nn.Linear(model_kwargs['low_dim_feature_dim'], hidden_size)
        #         for p in self.low_dim_emb.parameters():
        #             p.requires_grad = True
        #     else:
        #         assert mode == 'pretraining'
        #         self.low_dim_emb = None

        # must attach autoencoder at last to avoid loading params in ldm ckpt or init weights
        # self.autoencoder = autoencoder
        # self.last_training_batch = None
        
    def configure_optimizers(self):

        tuned_parameters = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(
            tuned_parameters,
            lr=self.lr,
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps)

        self.lr_scheduler = scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    # def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            WrappedTransformerDecoder,
            WrappedTransformerEncoder,
            nn.LeakyReLU,
            ResBottleneck,
            DownsampleObsLDM,
            DownsampleCVAE  # double check
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def pred_epsilon(self, noise, timestep, language_emb, image_emb):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None]
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(noise.shape[0]).to(device=noise.device)
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        language_emb = language_emb.unsqueeze(1)
        image_emb = image_emb.unsqueeze(1)
        embeddings = torch.cat([time_emb, language_emb, image_emb, self.z_up(noise).unsqueeze(1)], dim=1)
        pred_noise = self.denoiser(embeddings)
        return self.z_down(pred_noise[:, -1, :])

    def get_language_emb(self, batch):
        features = batch['obs']['language_embedding']
        return self.language_emb(features)

    def get_image_emb(self, batch):
        features = self.image_encoder(batch['obs']['image'])
        return self.image_emb(features)

    def predict_action(self, batch):
        # scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        # scheduler.set_timesteps(self.num_inference_timesteps)
        scheduler = self.noise_scheduler
        batch_size = batch['obs']['image'].shape[0]
        language_emb = self.get_language_emb(batch=batch)
        image_emb = self.get_image_emb(batch=batch)
        # recover z
        z = torch.randn(size=(batch_size, self.latent_size), dtype=language_emb.dtype, device=language_emb.device)
        for t in scheduler.timesteps:
            model_output = self.pred_epsilon(
                noise=z, timestep=t, language_emb=language_emb, image_emb=image_emb)
            z = scheduler.step(model_output, t, z).prev_sample
        
        _, pred_action = self.autoencoder.decode(z=z.unsqueeze(1))
        return pred_action

    def forward(self, batch):
        # autoencoder
        posterior = self.autoencoder.encode(batch)
        image_emb = self.get_image_emb(batch=batch)
        if self.autoencoder.sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        language_emb = self.get_language_emb(batch)

        # diffusion
        noise = torch.randn(z.shape, device=z.device)
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(z.shape[0],), device=z.device
        ).long()
        noisy_latent = self.noise_scheduler.add_noise(z, noise, timesteps)
 
        pred = self.pred_epsilon(
            noise = noisy_latent,
            timestep = timesteps,
            language_emb = language_emb,
            image_emb = image_emb
        )
        denoise_loss = F.mse_loss(noise, pred)

        return denoise_loss

    def training_step(self, batch, batch_idx):
        forward_results = self.forward(batch=batch)
        self.log('train/denoise_loss', forward_results, sync_dist=True)
        return forward_results

    # def on_train_epoch_end(self) -> None:
    #     with torch.no_grad():
    #         batch = self.last_training_batch
    #         raw_action = batch['action']
    #         pred_action = self.predict_action(
    #             raw_language_features=batch['language'],
    #             raw_image_features=batch['image'],
    #             raw_low_dim_data=batch.get('low_dim')
    #         )
    #         model_mse_error = F.mse_loss(raw_action, pred_action)
    #         self.log('train/model_mse_error', model_mse_error, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        forward_results = self.forward(batch=batch, batch_idx=batch_idx)
        self.log('val/denoise_loss', forward_results, sync_dist=True, batch_size=batch['obs']['image'].shape[0])

        raw_action = batch['action']
        pred_action = self.predict_action(batch=batch)
        model_mse_error = F.mse_loss(raw_action, pred_action)
        self.log('val/model_mse_error', model_mse_error, sync_dist=True, batch_size=batch['obs']['image'].shape[0])
        return forward_results
