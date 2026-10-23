import math

import einops
import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

env_val = os.getenv("DIT_USE_COMPILE", "0").lower()
USE_COMPILE = env_val in ["1", "true", "yes", "on"]
print(f"DIT: USE_COMPILE={USE_COMPILE}")

if USE_COMPILE:
    torch_compile_deco = torch.compile(
        model=None,
        mode=None,
        dynamic=False,
        options={"max_autotune": True, "triton.cudagraphs": False},
    )
else:
    torch_compile_deco = lambda x: x


def bias_dropout_add_scale(
    x: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
    prob: float,
    training: bool,
) -> torch.Tensor:
    return residual + scale * F.dropout(x, p=prob, training=training)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :]
            self.sin_cached = emb.sin()[None, :, None, None, :]

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin, use_flash=True):
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]

    if use_flash:
        return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
    else:
        q, k, v = qkv.unbind(dim=2)

        def apply_rotary(x, cos, sin):

            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)

            return x * cos + rotate_half(x) * sin

        q_rotated = apply_rotary(q, cos, sin)
        k_rotated = apply_rotary(k, cos, sin)

        return torch.stack([q_rotated, k_rotated, v], dim=2)


class LearnableLossWeighting(nn.Module):
    def __init__(self, cond_dim, is_flow=True, hidden_dim=128):
        super().__init__()

        self.s_embed = TimestepEmbedder(cond_dim)
        if not is_flow:
            self.t_embed = TimestepEmbedder(cond_dim)
        else:
            self.t_embed = None

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize the last layer to zero so that initially e^-w = 1
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s, t=None):
        emb = self.s_embed(s)
        if t is not None and self.t_embed is not None:
            emb_t = self.t_embed(t)
            emb = emb + emb_t
        return self.mlp(emb).squeeze(-1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlockCausal(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout = dropout

    def forward(self, x, rotary_cos_sin, **kwargs):
        del kwargs
        batch_size, seq_len = x.shape[0], x.shape[1]

        # attention operation
        x_skip = x
        x = self.norm1(x)

        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.amp.autocast(device_type=qkv.device.type, enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = einops.rearrange(qkv, "b s ... -> (b s) ...")
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            step=seq_len,
            dtype=torch.int32,
            device=qkv.device,
        )
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0.0, causal=True
        )

        x = einops.rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)

        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        x = bias_dropout_add_scale(
            self.attn_out(x), scale, x_skip, self.dropout, self.training
        )

        # mlp operation
        x = bias_dropout_add_scale(
            self.mlp(self.norm2(x)), scale, x, self.dropout, self.training
        )
        return x


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.softcap = 50
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def custom_sdpa(self, q, k, v, softcap=-1.0):
        B, H, S, D = q.shape
        q = q / (D**0.5)
        attn_weights = torch.einsum("bhid,bhjd->bhij", q, k)  # (B, H, S, S)
        if softcap > 0.0:
            attn_weights = softcap * torch.tanh(attn_weights / softcap)
        attn_probs = torch.softmax(attn_weights, dim=-1)  # F.softmax
        output = torch.einsum("bhij,bhjd->bhid", attn_probs, v)  # (B, H, S, D)
        return output

    def forward(self, x, rotary_cos_sin=None, c=None, use_jvp_attn=False):
        x_skip = x
        x = self.norm1(x)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )
        x = modulate(x, shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype), use_flash=not use_jvp_attn
            )

        if use_jvp_attn:  # custom attention for JVP support
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            x = self.custom_sdpa(q, k, v, softcap=self.softcap)
            x = x.transpose(1, 2)
        else:
            x = flash_attn.flash_attn_qkvpacked_func(
                qkv,
                0.0,
                causal=False,
                softcap=self.softcap,
            )

        x = einops.rearrange(
            x,
            "b s h d -> b s (h d)",
        )

        x = bias_dropout_add_scale(
            self.attn_out(x), gate_msa, x_skip, self.dropout, self.training
        )
        x = bias_dropout_add_scale(
            self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)),
            gate_mlp,
            x,
            self.dropout,
            self.training,
        )
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        if x.ndim == 2:
            return self.embedding[x]
        assert x.ndim == 3
        return torch.einsum("blv,ve->ble", x.float(), self.embedding.float()).to(
            x.dtype
        )


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim, adaLN, bias: bool = True):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=bias)
        self.linear.weight.data.zero_()
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()
        self.adaLN = adaLN
        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        x = self.norm_final(x)
        if self.adaLN:
            shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            x = modulate(x, shift, scale)
        x = self.linear(x)
        return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = config.algo.causal_attention
        self.adaLN = not self.causal
        self.config = config
        self.vocab_size = vocab_size
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)
        if "flm" in self.config.algo.name or "fmlm" in self.config.algo.name:
            if self.config.algo.double_temb:
                self.sigma_map_prime = TimestepEmbedder(cond_dim)
            else:
                self.sigma_map_prime = None
        self.rotary_emb = Rotary(dim // config.model.n_heads)

        if getattr(config.algo, "learnable_loss_weighting", False):
            self.learnable_loss_weighting = LearnableLossWeighting(cond_dim=cond_dim)

        blocks = []
        for _ in range(config.model.n_blocks):
            if self.causal:
                block = DDiTBlockCausal(
                    dim=dim, n_heads=config.model.n_heads, dropout=config.model.dropout
                )
            else:
                block = DDiTBlock(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    cond_dim=cond_dim,
                    dropout=config.model.dropout,
                )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=self.adaLN,
        )

    @torch_compile_deco
    def forward(self, x, sigma, sigma_prime=None, use_jvp_attn=False):
        x = self.vocab_embed(x)

        if self.causal:
            t_cond = None
        else:
            t_emb = self.sigma_map(sigma)
            if sigma_prime is not None:
                if self.sigma_map_prime is not None:
                    t_prime_emb = self.sigma_map_prime(sigma_prime)
                else:
                    t_prime_emb = self.sigma_map(sigma_prime)
                t_emb = t_emb + t_prime_emb

            t_cond = F.silu(t_emb)

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](
                    x,
                    rotary_cos_sin,
                    c=t_cond,
                    use_jvp_attn=use_jvp_attn,
                )
            x = self.output_layer(x, c=t_cond)

        return x
