
import copy
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffusers.optimization import get_cosine_schedule_with_warmup
import numpy as np
import lightning.pytorch as pl
from diffusion_policy.model.common.normalizer import LinearNormalizer
from loguru import logger

class ImageAdapter(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()

        self.down = nn.Linear(in_dim, in_dim // 4)
        self.activation = nn.LeakyReLU()
        self.up = nn.Linear(in_dim // 4, in_dim)
        
        self.out_linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        y = self.up(self.activation(self.down(x))) + x
        return self.out_linear(y)

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=1)
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class AutoencoderLoss(torch.nn.Module):
    def __init__(self, kl_weight=1e-6):
        super().__init__()
        self.kl_weight = kl_weight
    
    def recon_kl_loss(
        self, inputs, reconstructions, posteriors
    ):
        rec_loss = torch.nn.functional.mse_loss(inputs, reconstructions)
        
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss_dict = {
            "rec_loss": rec_loss,
            "kl_loss": kl_loss,
            "total_loss": rec_loss + self.kl_weight * kl_loss,
        }

        return loss_dict

def get_pe(hidden_size, max_len=100):  
    pe = torch.zeros(max_len, hidden_size)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe

class WrappedTransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: int,
        n_layers: int = None,
    ):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size*4,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
                norm_first=True
            ),
            num_layers=n_layers
        )
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, embeddings):
        return self.ln(self.encoder(embeddings))


class WrappedTransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: int,
        n_layers: int = None,
    ):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size*4,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
                norm_first=True
            ),
            num_layers=n_layers
        )
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, tgt, memory):
        return self.ln(self.decoder(tgt, memory))

class DownsampleCVAE(pl.LightningModule):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        latent_size: int,
        horizon: int,
        n_heads: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        dropout: float,
        sample_posterior: bool, 
        kl_weight: float,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        use_cosine_lr: bool=True,
    ):
        super().__init__()

        self.sample_posterior = sample_posterior # indicates whether it's an autoencoder or vae

        self.save_hyperparameters()

        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.horizon = horizon
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.use_cosine_lr = use_cosine_lr

        self.action_emb = nn.Linear(action_dim, hidden_size)

        self.cls = nn.Parameter(data=torch.zeros(size=(1, hidden_size)), requires_grad=True)
        self.z_encoder = WrappedTransformerEncoder(hidden_size=hidden_size, 
                                                   n_heads=n_heads, 
                                                   dropout=dropout, 
                                                   n_layers=n_encoder_layers)
        self.z_down = nn.Linear(hidden_size, latent_size * 2)

        self.z_up = nn.Linear(latent_size, hidden_size)
        self.conditioner = WrappedTransformerEncoder(hidden_size=hidden_size, 
                                                   n_heads=n_heads, 
                                                   dropout=dropout, 
                                                   n_layers=n_decoder_layers)
        self.decoder = WrappedTransformerDecoder(hidden_size=hidden_size, 
                                                   n_heads=n_heads, 
                                                   dropout=dropout, 
                                                   n_layers=n_decoder_layers)

        self.action_head = nn.Linear(hidden_size, action_dim)

        self.loss = AutoencoderLoss(
            kl_weight=kl_weight if self.sample_posterior else 0
        )

        self.register_buffer(
            'pe', get_pe(hidden_size=hidden_size, max_len=horizon*2))
        self.normalizer = LinearNormalizer()
        
    def configure_optimizers(self):
        tuned_named_parameters = [(name, param) for name, param in self.named_parameters() if param.requires_grad and 'normalizer' not in name]
        logger.debug('\n'.join([f'{name}: {param.shape}' for name, param in tuned_named_parameters]))
        optimizer = torch.optim.AdamW(
            [p for _, p in tuned_named_parameters],
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        if self.use_cosine_lr:
            scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                    num_warmup_steps=self.warmup_steps, 
                                    num_training_steps=self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader()))
            self.lr_scheduler = scheduler
        # print(f'max steps: {self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())}, max epochs: {self.trainer.max_epochs}')
        if self.use_cosine_lr:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'
                }
            }
        else:
            return optimizer
    
    def encode(self, batch, normalize_input=True):
        if normalize_input:
            normalized_action = self.normalize_input(batch)
            action = normalized_action
        else:
            normalized_action = None
            action = batch['action']
        # obs_emb = self.get_obs_emb(raw_image_features=batch['image'], raw_low_dim_data=batch.get('low_dim'))

        batch_size = action.shape[0]

        pos_action_emb = self.action_emb(action) + self.pe[:, :self.horizon, :].expand((batch_size, self.horizon, self.hidden_size))
        cls = self.cls.expand((batch_size, 1, self.hidden_size))

        z_encoder_input = torch.cat([cls, pos_action_emb], dim=1)
        # if obs_emb is not None:
            # z_encoder_input = torch.cat([z_encoder_input, obs_emb], dim=1)

        z_encoder_output = self.z_encoder(z_encoder_input)[:, 0, :] # take cls token as output
        z_encoder_output = self.z_down(z_encoder_output)
        posterior = DiagonalGaussianDistribution(z_encoder_output)
        return {'posterior': posterior, 'normalized_action': normalized_action}
    
    def decode(self, posterior=None, z=None, sample_posterior=True, unnormalize_output=True):
        if not self.sample_posterior:
            sample_posterior = False
        if z is None:
            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()
        # z += torch.randn_like(z) * 4
        up_z = self.z_up(z)
        batch_size = up_z.shape[0]
        condition_input = up_z.unsqueeze(1)
        condition = self.conditioner(condition_input)

        decoder_input = self.pe[:, :self.horizon, :].expand((batch_size, self.horizon, self.hidden_size))
        decoder_output = self.decoder(tgt=decoder_input, memory=condition)
        pred = self.action_head(decoder_output)

        return {
            'z': z,
            'pred': pred,
            'unnormalized_pred': self.unnormalize_output(pred) if unnormalize_output else None
        }

    def normalize_input(self, batch):
        if self.normalizer is not None:
            normalized_action = self.normalizer['action'].normalize(batch['action'])
            return normalized_action
        else:
            raise ValueError("Normalizer is not set")

    def unnormalize_output(self, pred):
        if self.normalizer is not None:
            pred = self.normalizer['action'].unnormalize(pred)
            return pred
        else:
            raise ValueError("Normalizer is not set")

    def forward(self, batch, sample_posterior=True, normalize_input=True, unnormalize_output=True):
        """
        normalize_input: whether to normalize the input
        unnormalize_output: whether to unnormalize the output
        Loss is always computed in normalized space.
        """
        encoding_output = self.encode(batch, normalize_input=normalize_input)
        if not self.sample_posterior:
            sample_posterior = False
        decoding_output = self.decode(posterior=encoding_output['posterior'], sample_posterior=sample_posterior, unnormalize_output=unnormalize_output)
        
        loss_dict = self.loss.recon_kl_loss(
            inputs=encoding_output['normalized_action'] if normalize_input else batch['action'], 
            reconstructions=decoding_output['pred'], 
            posteriors=encoding_output['posterior'])
        rec_loss_unnormalized = F.mse_loss(decoding_output['unnormalized_pred'].detach(), batch['action'].detach())

        # return loss_dict | decoding_output | encoding_output
        return {
            "rec_loss": loss_dict['rec_loss'],
            "kl_loss": loss_dict['kl_loss'],
            "total_loss": loss_dict['total_loss'],
            "rec_loss_unnormalized": rec_loss_unnormalized,
            'z': decoding_output['z'],
            'pred': decoding_output['pred'],
            'unnormalized_pred': decoding_output['unnormalized_pred'],
            'posterior': encoding_output['posterior'],
            'normalized_action': encoding_output['normalized_action']
        }
    
    def training_step(self, batch, batch_idx):
        forward_output = self.forward(batch=batch, sample_posterior=True, normalize_input=True)
        loss_dict = {f'train_{k}': v for k, v in forward_output.items() if 'loss' in k}
        self.log_dict(loss_dict, sync_dist=True, prog_bar=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], sync_dist=True)
        return loss_dict['train_total_loss']

    def validation_step(self, batch, batch_idx):
        forward_output = self.forward(batch=batch, sample_posterior=False, normalize_input=True)
        loss_dict = {f'val_{k}': v for k, v in forward_output.items() if 'loss' in k}
        self.log_dict(loss_dict, sync_dist=True)
        return loss_dict['val_total_loss']

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
        logger.info(f'Normalizer set: {self.normalizer}')

    def predict_step(self, batch):
        self.eval()
        return self.forward(batch=batch, sample_posterior=False, normalize_input=True, unnormalize_output=True)

class ResBottleneck(nn.Module):
    def __init__(self, hidden_size, norm=True) -> None:
        super().__init__()
        self.norm = norm

        self.down = nn.Linear(hidden_size, hidden_size // 4)
        self.activation = nn.LeakyReLU()
        self.up = nn.Linear(hidden_size // 4, hidden_size)
        
        if self.norm:
            self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        y = self.up(self.activation(self.down(x))) + x
        if self.norm:
            return self.ln(y)
        else:
            return y