from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
import numpy as np
import lightning.pytorch as pl

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
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1,2])
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
        self, inputs, reconstructions, posteriors, split="train"
    ):
        rec_loss = torch.nn.functional.mse_loss(inputs, reconstructions)
        
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = rec_loss + self.kl_weight * kl_loss

        log = {
            "{}/ae_total_loss".format(split): loss.clone().detach().mean(),
            "{}/kl_loss".format(split): kl_loss.detach().mean(),
            "{}/rec_loss".format(split): rec_loss.detach().mean(),
        }
        return loss, log

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
        n_encoder_layers: int = None,
        **kwargs
    ):
        super().__init__()
        if n_encoder_layers is not None:
            n_layers = n_encoder_layers
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
        n_decoder_layers: int = None,
        **kwargs
    ):
        super().__init__()
        if n_decoder_layers is not None:
            n_layers = n_decoder_layers
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
        model_kwargs,
        training_kwargs,
        mode,  # pretraining, finetuning, inference
        all_config=None
    ):
        super().__init__()
        ckpt_path = model_kwargs.ckpt_path
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # reloading config from ckpt
            hyper_params = copy.deepcopy(ckpt['hyper_parameters'])

            # replace all the model config to the original
            # but keep low_dim_feature_dim obtained in main.preprocess_config
            low_dim_feature_dim = model_kwargs.low_dim_feature_dim
            model_kwargs = hyper_params['model_kwargs']
            model_kwargs.low_dim_feature_dim = low_dim_feature_dim

        # initialze model
        self.all_config = all_config
        self.training_kwargs = training_kwargs
        self.model_kwargs = model_kwargs
        self.save_hyperparameters()

        self.action_dim = action_dim = model_kwargs['action_dim']
        self.hidden_size = hidden_size = model_kwargs['hidden_size']
        self.latent_size = latent_size = model_kwargs['latent_size']
        self.horizon = horizon = model_kwargs['horizon']

        self.action_emb = nn.Linear(action_dim, hidden_size)

        self.cls = nn.Parameter(data=torch.zeros(size=(1, hidden_size)), requires_grad=True)
        self.z_encoder = WrappedTransformerEncoder(**model_kwargs)
        self.z_down = nn.Linear(hidden_size, latent_size * 2)

        self.z_up = nn.Linear(latent_size, hidden_size)
        self.conditioner = WrappedTransformerEncoder(**model_kwargs)
        self.decoder = WrappedTransformerDecoder(**model_kwargs)

        self.action_head = nn.Linear(hidden_size, action_dim)

        self.loss = AutoencoderLoss(
            **training_kwargs.loss_kwargs
        )

        self.register_buffer(
            'pe', get_pe(hidden_size=hidden_size, max_len=horizon*2))

        self.with_obs = model_kwargs.get('with_obs', True)
        if self.with_obs:
            if model_kwargs.get('low_dim_feature_dim') is not None:
                assert mode == 'finetuning' or mode == 'inference'
                self.low_dim_emb = nn.Linear(model_kwargs['low_dim_feature_dim'], hidden_size)
            else:
                assert mode == 'pretraining'
                self.low_dim_emb = None

        if self.with_obs:  # fix this in config
            if hidden_size == 512:
                self.img_emb = ResBottleneck(hidden_size=hidden_size)
            else:
                self.img_emb = ImageAdapter(in_dim=512, out_dim=hidden_size)

        self.with_language = model_kwargs.get('with_language', False)
        if self.with_language:
            self.language_emb = nn.Linear(in_features=768, out_features=hidden_size)

        self.last_training_batch = None

        if ckpt_path is not None:
            if mode == 'finetuning':
                self.load_state_dict(ckpt['state_dict'], strict=False)  # no low_dim during pretraining
            elif mode == 'inference' or mode == 'pretraining':  # pretraining ldm load the pretrained ae
                self.load_state_dict(ckpt['state_dict'])  # load the whole ckpt
            del ckpt
            print(f'WARNING: ignoring AE config, AE loaded from {ckpt_path}')
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
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
            AutoencoderLoss,
            ImageAdapter
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
        elif isinstance(module, DownsampleCVAE):
            torch.nn.init.normal_(module.cls, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def configure_optimizers(self):
        kwargs = self.training_kwargs
        tuned_parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            tuned_parameters,
            lr=kwargs.lr,
            weight_decay=kwargs.weight_decay
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=kwargs.warmup_steps, num_training_steps=kwargs.num_training_steps)

        self.lr_scheduler = scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def get_obs_emb(self, raw_image_features, raw_low_dim_data):
        if self.with_obs:
            image_emb = self.img_emb(raw_image_features)
            if raw_low_dim_data is not None and self.low_dim_emb is not None:
                low_dim_emb = self.low_dim_emb(raw_low_dim_data)
                return torch.cat([image_emb, low_dim_emb], dim=1)
            else:
                return image_emb
        else:
            return None

    def get_language_emb(self, raw_language_features):
        if self.with_language:
            return self.language_emb(raw_language_features)
        else:
            return None
    
    def encode(self, batch):
        action = batch['action']
        obs_emb = self.get_obs_emb(raw_image_features=batch['image'], raw_low_dim_data=batch.get('low_dim'))

        batch_size = action.shape[0]

        pos_action_emb = self.action_emb(action) + self.pe[:, :self.horizon, :].expand((batch_size, self.horizon, self.hidden_size))
        cls = self.cls.expand((batch_size, 1, self.hidden_size))

        z_encoder_input = torch.cat([cls, pos_action_emb], dim=1)
        if obs_emb is not None:
            z_encoder_input = torch.cat([z_encoder_input, obs_emb], dim=1)

        z_encoder_output = self.z_encoder(z_encoder_input)[:, 0:1, :]
        z_encoder_output = self.z_down(z_encoder_output)
        posterior = DiagonalGaussianDistribution(z_encoder_output)
        return posterior, obs_emb
    
    def decode(self, obs_emb, posterior=None, z=None, sample_posterior=True, raw_language_features=None):
        if z is None:
            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()
        z = self.z_up(z)
        batch_size = z.shape[0]
        
        condition_input = z
        if obs_emb is not None:
            condition_input = torch.cat([obs_emb, condition_input], dim=1)  # obs_emb, z
        if self.with_language:
            condition_input = torch.cat([self.get_language_emb(raw_language_features), condition_input], dim=1)  # lang, obs, z
        condition = self.conditioner(condition_input)

        decoder_input = self.pe[:, :self.horizon, :].expand((batch_size, self.horizon, self.hidden_size))
        decoder_output = self.decoder(tgt=decoder_input, memory=condition)
        pred_action = self.action_head(decoder_output)
        return pred_action

    def forward(self, batch, batch_idx, sample_posterior=True, split='train'):
        posterior, obs_emb = self.encode(batch)
        pred_action = self.decode(posterior=posterior, obs_emb=obs_emb, sample_posterior=sample_posterior, raw_language_features=batch['language'])

        total_loss, log_dict = self.loss.recon_kl_loss(
            inputs=batch['action'], reconstructions=pred_action, posteriors=posterior, split=split)
        return total_loss, log_dict
    
    def training_step(self, batch, batch_idx):
        self.last_training_batch = batch
        total_loss, log_dict = self.forward(batch=batch, batch_idx=batch_idx, split='train')
        self.log_dict(log_dict, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, log_dict = self.forward(batch=batch, batch_idx=batch_idx, split='val')
        self.log_dict(log_dict, sync_dist=True)
        return total_loss

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