if __name__ == "__main__":
    import sys
    from pathlib import Path
    # print(Path(__file__).parent.parent.parent.parent.as_posix())
    sys.path.append(Path(__file__).parent.parent.parent.parent.as_posix())

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.cvae_rold import WrappedTransformerEncoder, WrappedTransformerDecoder, get_pe

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probabilities = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probabilities * torch.log(avg_probabilities + 1e-10)))

        return quantized.permute(0, 2, 1).contiguous(), loss, perplexity, encodings

class VQVAE(L.LightningModule):
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
        num_embeddings: int,
        commitment_cost: float,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        use_cosine_lr: bool=True,
    ):
        super().__init__()

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

        self.encoder = WrappedTransformerEncoder(hidden_size=hidden_size, 
                                                 n_heads=n_heads, 
                                                 dropout=dropout, 
                                                 n_layers=n_encoder_layers)
        self.encoder_proj = nn.Linear(hidden_size, latent_size)

        self.vq_layer = VectorQuantizer(num_embeddings, latent_size, commitment_cost)

        self.decoder_proj = nn.Linear(latent_size, hidden_size)
        self.decoder = WrappedTransformerDecoder(hidden_size=hidden_size, 
                                                 n_heads=n_heads, 
                                                 dropout=dropout, 
                                                 n_layers=n_decoder_layers)

        self.action_head = nn.Linear(hidden_size, action_dim)

        self.register_buffer('pe', get_pe(hidden_size=hidden_size, max_len=horizon*2))
        self.normalizer = LinearNormalizer()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        if self.use_cosine_lr:
            scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                    num_warmup_steps=self.warmup_steps, 
                                    num_training_steps=self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader()))
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
    
    def encode(self, batch, normalize_input=True):
        if normalize_input:
            normalized_action = self.normalize_input(batch)
            output = {'normalized_action': normalized_action}
            action = normalized_action
        else:
            action = batch['action']
            output = dict()

        batch_size = action.shape[0]

        pos_action_emb = self.action_emb(action) + self.pe[:, :self.horizon, :].expand((batch_size, self.horizon, self.hidden_size))

        encoder_output = self.encoder(pos_action_emb)
        z = self.encoder_proj(encoder_output)
        
        quantized, vq_loss, perplexity, encodings = self.vq_layer(z)
        
        return {'z': z, 'quantized': quantized, 'vq_loss': vq_loss, 'perplexity': perplexity, 'encodings': encodings} | output
    
    def decode(self, quantized, unnormalize_output=True):
        decoder_input = self.decoder_proj(quantized)
        decoder_output = self.decoder(tgt=decoder_input, memory=decoder_input)
        pred = self.action_head(decoder_output)
        output = {'pred': pred}
        if unnormalize_output:
            output['unnormalized_pred'] = self.unnormalize_output(pred)
        return output

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

    def forward(self, batch, normalize_input=True, unnormalize_output=True):
        encoding_output = self.encode(batch, normalize_input=normalize_input)
        decoding_output = self.decode(encoding_output['quantized'], unnormalize_output=unnormalize_output)
        
        recon_loss = F.mse_loss(decoding_output['pred'], encoding_output['normalized_action'] if normalize_input else batch['action'])
        total_loss = recon_loss + encoding_output['vq_loss']
        
        loss_dict = {
            'recon_loss': recon_loss,
            'vq_loss': encoding_output['vq_loss'],
            'total_loss': total_loss,
            'perplexity': encoding_output['perplexity']
        }
        
        return loss_dict | decoding_output | encoding_output
    
    def training_step(self, batch, batch_idx):
        forward_output = self.forward(batch=batch, normalize_input=True)
        loss_dict = {f'train_{k}': v for k, v in forward_output.items() if 'loss' in k or k == 'perplexity'}
        self.log_dict(loss_dict, sync_dist=True, prog_bar=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], sync_dist=True)
        return loss_dict['train_total_loss']

    def validation_step(self, batch, batch_idx):
        forward_output = self.forward(batch=batch, normalize_input=True)
        loss_dict = {f'val_{k}': v for k, v in forward_output.items() if 'loss' in k or k == 'perplexity'}
        self.log_dict(loss_dict, sync_dist=True)
        return loss_dict['val_total_loss']

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

def main():
    import matplotlib.pyplot as plt
    import hydra
    from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicImageDatamodule
    # Hyperparameters
    action_dim = 10
    hidden_size = 64
    latent_size = 32
    horizon = 16
    n_heads = 8
    n_encoder_layers = 3
    n_decoder_layers = 3
    dropout = 0.1
    num_embeddings = 512
    commitment_cost = 0.25
    lr = 1e-4
    weight_decay = 1e-6
    warmup_steps = 100
    batch_size = 128

    d = {'_target_': 'diffusion_policy.dataset.robomimic_replay_image_dataset.RobomimicReplayImageDataset', 
        'shape_meta': {'obs': {'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'}, 
                                'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'}, 
                                'robot0_eef_pos': {'shape': [3]}, 
                                'robot0_eef_quat': {'shape': [4]}, 
                                'robot0_gripper_qpos': {'shape': [2]}}, 
                        'action': {'shape': [10]}}, 
        'dataset_path': '/home/zzy/robot/data/robomimic_data/robomimic/datasets/lift/ph/image_abs.hdf5', 
        'horizon': 16, 
        'pad_before': 0, 
        'pad_after': 15, 
        'n_obs_steps': 60, 
        'abs_action': True, 
        'rotation_rep': 'rotation_6d', 
        'use_legacy_normalizer': False, 
        'use_cache': False, 
        'seed': 42, 
        'val_ratio': 0.02}
    dataset = hydra.utils.instantiate(d)
    datamodule = RobomimicImageDatamodule(dataset=dataset, batch_size=batch_size)

    
    model = VQVAE(
        action_dim=action_dim,
        hidden_size=hidden_size,
        latent_size=latent_size,
        horizon=horizon,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        dropout=dropout,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
    )
    model.set_normalizer(datamodule.dataset.get_normalizer())
    data = next(iter(datamodule.train_dataloader()))

    # Forward pass
    output = model(data)

    # Sanity checks
    print(f"Input shape: {data['action'].shape}")
    print(f"Output shape: {output['pred'].shape}")
    print(f"Reconstruction loss: {output['recon_loss'].item():.4f}")
    print(f"VQ loss: {output['vq_loss'].item():.4f}")
    print(f"Total loss: {output['total_loss'].item():.4f}")
    print(f"Perplexity: {output['perplexity'].item():.2f}")

    # Check if the output matches the input shape
    assert output['pred'].shape == data['action'].shape, "Output shape doesn't match input shape"

    # Check if losses are reasonable (not NaN or extremely large)
    assert not torch.isnan(output['total_loss']), "Total loss is NaN"
    assert output['total_loss'] < 1e6, "Total loss is extremely large"

    # Check if perplexity is within a reasonable range
    assert 1 <= output['perplexity'] <= num_embeddings, "Perplexity is out of expected range"

    # Visualization of input and reconstructed actions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Input Actions")
    plt.plot(data['action'][0, :, 0].detach().numpy(), label='dim 0')
    plt.plot(data['action'][0, :, 1].detach().numpy(), label='dim 1')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Actions")
    plt.plot(output['pred'][0, :, 0].detach().numpy(), label='dim 0')
    plt.plot(output['pred'][0, :, 1].detach().numpy(), label='dim 1')
    plt.legend()

    plt.tight_layout()
    plt.show()

    trainer = L.Trainer(overfit_batches=1, max_epochs=100)
    trainer.fit(model, datamodule=datamodule)

    print("All sanity checks passed!")

if __name__ == "__main__":
    main()