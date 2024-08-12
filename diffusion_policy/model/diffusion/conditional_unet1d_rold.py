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
import zzy_utils
import lightning as L
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
# from RoLD.models.common import SinusoidalPosEmb, get_pe, WrappedTransformerEncoder, WrappedTransformerDecoder, ResBottleneck, ImageAdapter
# from RoLD.models.autoencoder.downsample_cvae import DownsampleCVAE
from diffusion_policy.model.diffusion.cvae_rold import DownsampleCVAE, get_pe, WrappedTransformerEncoder, WrappedTransformerDecoder, ResBottleneck, ImageAdapter
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.vision.r3m import R3MImageEncoderWrapper
from diffusion_policy.model.language.DistilBERT_utils import DistilBERTWrapper
from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat
from loguru import logger
class DownsampleObsLDM(L.LightningModule):
    def __init__(
        self,
        
        latent_size:int=256,
        hidden_size:int=256,
        horizon:int=16,
        language_feature_dim:int=768,
        image_feature_dim:int=512,
        load_image_features:bool=True,
        low_dim_feature_dim:int=None,
        n_layers:int=6,
        n_heads:int=8,
        dropout:float=0.1,

        lr:float=1e-4,
        warmup_steps:int=1000,
        # num_training_steps:int=100000,
        use_lr_scheduler:bool=True,

        num_inference_timesteps:int=1000,
        num_train_timesteps:int= 1000,
        beta_start:float = 0.0001,
        beta_end:float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        variance_type: str = "fixed_small", # 
        clip_sample: bool = True, # required when predict_epsilon=False
        prediction_type: str = "epsilon",
        
        task_name:str=None,
        no_normalizer:bool=True,
    ) -> None:
        super().__init__()

        # initialize model
        self.hidden_size = hidden_size
        # self.latent_size = latent_size
        self.horizon = horizon
        self.image_feature_dim = image_feature_dim
        self.load_image_features = load_image_features # indicates whether to load R3M features directly
        self.language_feature_dim = language_feature_dim
        self.low_dim_feature_dim = low_dim_feature_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.warmup_steps = warmup_steps
        # self.num_training_steps = num_training_steps
        self.num_inference_timesteps = num_inference_timesteps
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        self.use_lr_scheduler = use_lr_scheduler

        self.task_name = task_name
        self.no_normalizer = no_normalizer
        self.save_hyperparameters()
        self.normalizer = SingleFieldLinearNormalizer()
        


        self.time_emb = SinusoidalPosEmb(dim=hidden_size)
        self.register_buffer(
            'pe', get_pe(hidden_size=hidden_size, max_len=horizon*2)
        )

        self.latent_size = latent_size
        self.z_up = nn.Linear(self.latent_size, self.hidden_size)
        self.denoiser = WrappedTransformerEncoder(
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
        )
        self.z_down = nn.Linear(self.hidden_size, self.latent_size)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            variance_type=variance_type,
            clip_sample=clip_sample,
            prediction_type=prediction_type
        )

        if not self.load_image_features:
            self.image_encoder = R3MImageEncoderWrapper()
        self.image_emb = nn.Linear(in_features=image_feature_dim, out_features=hidden_size)
        self.language_emb = nn.Linear(in_features=language_feature_dim, out_features=hidden_size)
        # self.load_state_dict(torch.load("/home/zzy/robot/data/diffusion_policy_data/data/latent_diffusion_policy_ldm/2024-08-05_19-41-56/last.ckpt")['state_dict'])

    def load_ae(self, ae: DownsampleCVAE):
        self.autoencoder = ae.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        if self.latent_size != self.autoencoder.latent_size:
            self.latent_size = self.autoencoder.latent_size
            self.z_up = nn.Linear(self.latent_size, self.hidden_size)
            self.denoiser = WrappedTransformerEncoder(
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                dropout=self.dropout,
            )
            self.z_down = nn.Linear(self.hidden_size, self.latent_size)

    def configure_optimizers(self):

        tuned_named_parameters = [ (name, param) for name, param in self.named_parameters() if param.requires_grad and 'normalizer' not in name]
        
        logger.debug('\n'.join([f'{name}: {param.shape}' for name, param in tuned_named_parameters]))

        optimizer = torch.optim.Adam(
            [p for _, p in tuned_named_parameters],
            lr=self.lr,
        )
        if self.use_lr_scheduler:
            if self.trainer.max_epochs is None or self.trainer.max_epochs == -1:
                num_training_steps = self.trainer.max_steps
            else:
                num_training_steps = self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())
            scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                        num_warmup_steps=self.warmup_steps, 
                                        num_training_steps=num_training_steps)

            self.lr_scheduler = scheduler
            return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
                }
            }
        else:
            return optimizer

    # def _init_weights(self, module):
        # ignore_types = (nn.Dropout, 
        #     SinusoidalPosEmb, 
        #     nn.TransformerEncoderLayer, 
        #     nn.TransformerDecoderLayer,
        #     nn.TransformerEncoder,
        #     nn.TransformerDecoder,
        #     nn.ModuleList,
        #     nn.Mish,
        #     nn.Sequential,
        #     WrappedTransformerDecoder,
        #     WrappedTransformerEncoder,
        #     nn.LeakyReLU,
        #     ResBottleneck,
        #     DownsampleObsLDM,
        #     DownsampleCVAE  # double check
        # )
        # if isinstance(module, (nn.Linear, nn.Embedding)):
        #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #     if isinstance(module, nn.Linear) and module.bias is not None:
        #         torch.nn.init.zeros_(module.bias)
        # elif isinstance(module, nn.MultiheadAttention):
        #     weight_names = [
        #         'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
        #     for name in weight_names:
        #         weight = getattr(module, name)
        #         if weight is not None:
        #             torch.nn.init.normal_(weight, mean=0.0, std=0.02)
        #     bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
        #     for name in bias_names:
        #         bias = getattr(module, name)
        #         if bias is not None:
        #             torch.nn.init.zeros_(bias)
        # elif isinstance(module, nn.LayerNorm):
        #     torch.nn.init.zeros_(module.bias)
        #     torch.nn.init.ones_(module.weight)
        # elif isinstance(module, ignore_types):
        #     # no param
        #     pass
        # else:
        #     raise RuntimeError("Unaccounted module {}".format(module))

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
        if self.load_image_features:
            features = batch['obs']['image_features']
        else:
            features = self.image_encoder(batch['obs']['image'])
        return self.image_emb(features)

    def predict_action(self, batch, unnormalize_output=True):
        scheduler = self.noise_scheduler
        batch_size = len(batch['obs']['language_embedding'])
        language_emb = self.get_language_emb(batch=batch)
        image_emb = self.get_image_emb(batch=batch)
        # recover z
        z = torch.randn(size=(batch_size, self.latent_size), dtype=language_emb.dtype, device=language_emb.device)
        for t in scheduler.timesteps:
            model_output = self.pred_epsilon(
                noise=z, timestep=t, language_emb=language_emb, image_emb=image_emb)
            z = scheduler.step(model_output, t, z).prev_sample
        unnormalized_z = self.normalizer.unnormalize(z)

        decoding_output = self.autoencoder.decode(z=unnormalized_z, unnormalize_output=unnormalize_output)
        return {
                    'pred_z': z,
                    'unnormalized_z': unnormalized_z,
                    'pred_action': decoding_output['pred'],
                    'unnormalized_pred_action': decoding_output['unnormalized_pred'] if unnormalize_output else None
                }

    def forward(self, batch, normalize_input=True):
        # autoencoder
        encoding_output = self.autoencoder.encode(batch, normalize_input=normalize_input)
        image_emb = self.get_image_emb(batch=batch)
        if self.autoencoder.sample_posterior:
            z = encoding_output['posterior'].sample()
        else:
            z = encoding_output['posterior'].mode()
        language_emb = self.get_language_emb(batch)

        normalized_z = self.normalizer.normalize(z)
        # diffusion
        noise = torch.randn(normalized_z.shape, device=normalized_z.device)
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(normalized_z.shape[0],), device=normalized_z.device
        ).long()
        noisy_latent = self.noise_scheduler.add_noise(normalized_z, noise, timesteps)
 
        pred = self.pred_epsilon(
            noise = noisy_latent,
            timestep = timesteps,
            language_emb = language_emb,
            image_emb = image_emb
        )
        denoise_loss = F.mse_loss(noise, pred)

        return {'denoise_loss': denoise_loss, 
                'z': z,
                'normalized_z': normalized_z, 
                'posterior': encoding_output['posterior'], 
                'normalized_action': encoding_output['normalized_action'],}

    def training_step(self, batch, batch_idx):
        forward_output = self.forward(batch=batch, normalize_input=True)
        self.log('train/denoise_loss', forward_output['denoise_loss'], sync_dist=True, prog_bar=True)
        self.log('trainer/lr', self.optimizers().param_groups[0]['lr'], sync_dist=True)
        return forward_output['denoise_loss']

    def validation_step(self, batch, batch_idx):
        forward_output = self.forward(batch=batch, normalize_input=True)
        self.log('val/denoise_loss', forward_output['denoise_loss'], sync_dist=True, batch_size=batch['action'].shape[0]) # one step normalized z loss
        
        pred_output = self.predict_action(batch=batch, unnormalize_output=True)
        self.log('val/normalized_z_mse',        F.mse_loss(forward_output['normalized_z'], pred_output['pred_z']),              sync_dist=True, batch_size=batch['action'].shape[0]) # multi step normalized z loss
        self.log('val/unnormalized_z_mse',      F.mse_loss(forward_output['z'], pred_output['unnormalized_z']),                 sync_dist=True, batch_size=batch['action'].shape[0]) # unnormalized z loss
        self.log('val/normalized_action_mse',   F.mse_loss(forward_output['normalized_action'], pred_output['pred_action']),    sync_dist=True, batch_size=batch['action'].shape[0]) # normalized action loss
        self.log('val/unnormalized_action_mse', F.mse_loss(batch['action'], pred_output['unnormalized_pred_action']),                sync_dist=True, batch_size=batch['action'].shape[0]) # unnormalized action loss
        ae_forward_output = self.autoencoder.forward(batch=batch, normalize_input=True)
        self.log('val/ae_rec_loss', ae_forward_output['rec_loss'], sync_dist=True, batch_size=batch['action'].shape[0]) # autoencoder reconstruction loss
        return forward_output

    def set_normalizer(self, datamodule):
        if zzy_utils.check_environ_dry_run():
            self.autoencoder.eval()
            for batch in datamodule.train_dataloader():
                output = self.autoencoder.forward(batch=batch, normalize_input=True)
                stat = {'max': output['z'].max(dim=0)[0], 'min': output['z'].min(dim=0)[0]}
                self.normalizer = get_range_normalizer_from_stat(stat=stat)
                return
        outputs = L.Trainer().predict(self.autoencoder, datamodule=datamodule)
        z = torch.cat([output['z'] for output in outputs], dim=0)
        stat = {'max': z.max(dim=0)[0], 'min': z.min(dim=0)[0]}
        self.normalizer = get_range_normalizer_from_stat(stat=stat)
        