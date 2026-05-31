
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm2(x)
        return x + residual


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb





# Training example
# def train_step(model, x_1, antibody_emb, antigen_emb, optimizer, use_vp=False):
#     """
#     Single training step for CFM model.
#
#     Args:
#         model: ConditionalFlowMatchingModel instance
#         x_1: target sequences [batch, seq_dim]
#         antibody_emb: antibody embeddings [batch, antibody_embed_dim]
#         antigen_emb: antigen embeddings [batch, antigen_embed_dim]
#         optimizer: torch optimizer
#         use_vp: if True, use variance-preserving loss
#     """
#     optimizer.zero_grad()
#
#     if use_vp:
#         loss = model.compute_vp_cfm_loss(x_1, antibody_emb, antigen_emb)
#     else:
#         loss = model.compute_cfm_loss(x_1, antibody_emb, antigen_emb)
#
#     loss.backward()
#     optimizer.step()
#
#     return loss.item()
#
#
# # Sampling example with guidance
# def sample_antibodies(model, antibody_emb, antigen_emb, guidance_scale=2.0):
#     """
#     Sample antibodies with classifier-free guidance.
#
#     Args:
#         model: trained ConditionalFlowMatchingModel
#         antibody_emb: antibody context [batch, antibody_embed_dim]
#         antigen_emb: target antigen [batch, antigen_embed_dim]
#         guidance_scale: strength of conditioning (1.0 = no guidance, 2-4 typical)
#     """
#     model.eval()
#
#     # Use Heun sampler for better quality
#     samples = model.heun_sample(
#         antibody_emb=antibody_emb,
#         antigen_emb=antigen_emb,
#         num_steps=50,
#         guidance_scale=guidance_scale
#     )
#
#     return samples