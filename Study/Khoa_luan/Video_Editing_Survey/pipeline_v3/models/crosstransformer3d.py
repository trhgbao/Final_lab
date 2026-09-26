# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Tuple, Union

import os
import json
import torch
import numpy as np
import glob
import torch.nn.functional as F
from torch import nn
import math
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import is_torch_version, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import (
    AttentionProcessor,
    CogVideoXAttnProcessor2_0,
    FusedCogVideoXAttnProcessor2_0,
)
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
    get_3d_sincos_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CogVideoXPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=bias,
        )
        self.text_proj = nn.Linear(text_embed_dim, embed_dim)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds)

        batch, num_frames, channels, height, width = image_embeds.shape
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose(
            2, 3
        )  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(
            1, 2
        )  # [batch, num_frames x height x width, channels]

        embeds = torch.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]
        return embeds


class RefPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1920,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=bias,
        )

    def forward(self, image_embeds: torch.Tensor):
        r"""
        Args:
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        batch, num_frames, channels, height, width = image_embeds.shape
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose(
            2, 3
        )  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(
            1, 2
        )  # [batch, num_frames x height x width, channels]
        return image_embeds


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = (
            self.norm1(hidden_states, encoder_hidden_states, temb)
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
        )

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = (
            self.norm2(hidden_states, encoder_hidden_states, temb)
        )

        # feed-forward
        norm_hidden_states = torch.cat(
            [norm_encoder_hidden_states, norm_hidden_states], dim=1
        )
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
        )

        return hidden_states, encoder_hidden_states


def reshape_tensor(x, heads):
    """
    Reshapes the input tensor for multi-head attention.

    Args:
        x (torch.Tensor): The input tensor with shape (batch_size, length, width).
        heads (int): The number of attention heads.

    Returns:
        torch.Tensor: The reshaped tensor, with shape (batch_size, heads, length, width).
    """
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverCrossAttention(nn.Module):
    """

    Args:
        dim (int): Dimension of the input latent and output. Default is 3072.
        dim_head (int): Dimension of each attention head. Default is 128.
        heads (int): Number of attention heads. Default is 16.
        kv_dim (int): Dimension of the key/value input, allowing flexible cross-attention. Default is 2048.

    Attributes:
        scale (float): Scaling factor used in dot-product attention for numerical stability.
        norm1 (nn.LayerNorm): Layer normalization applied to the input image features.
        norm2 (nn.LayerNorm): Layer normalization applied to the latent features.
        to_q (nn.Linear): Linear layer for projecting the latent features into queries.
        to_kv (nn.Linear): Linear layer for projecting the input features into keys and values.
        to_out (nn.Linear): Linear layer for outputting the final result after attention.

    """

    def __init__(self, *, dim=3072, dim_head=128, heads=16, kv_dim=2048, zero_init=True):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        # Layer normalization to stabilize training
        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        # Linear transformations to produce queries, keys, and values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(
            dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False
        )
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        if zero_init:
            nn.init.zeros_(self.to_out.weight)

    def forward(self, x, latents):
        """

        Args:
            x (torch.Tensor): Input image features with shape (batch_size, n1, D), where:
                - batch_size (b): Number of samples in the batch.
                - n1: Sequence length (e.g., number of patches or tokens).
                - D: Feature dimension.

            latents (torch.Tensor): Latent feature representations with shape (batch_size, n2, D), where:
                - n2: Number of latent elements.

        Returns:
            torch.Tensor: Attention-modulated features with shape (batch_size, n2, D).

        """
        # Apply layer normalization to the input image and latent features
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        # Compute queries, keys, and values
        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        # Reshape tensors to split into attention heads
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # Use PyTorch scaled_dot_product_attention (FlashAttention / Memory-Efficient Attention)
        # Avoids allocating massive O(N_q * N_kv) attention matrix in VRAM, saving ~50GB VRAM
        out = F.scaled_dot_product_attention(q, k, v)

        # Reshape and permute to prepare for final linear transformation
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)


class CrossTransformer3DModel(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        add_noise_in_inpaint_model: bool = False,
        is_train_cross: bool = False,
        cross_attn_in_channels: int = 16,
        cross_attn_interval: int = 2,
        cross_attn_dim_head: int = 128,
        cross_attn_num_heads: int = 16,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        post_patch_height = sample_height // patch_size
        post_patch_width = sample_width // patch_size
        post_time_compression_frames = (
            sample_frames - 1
        ) // temporal_compression_ratio + 1
        self.num_patches = (
            post_patch_height * post_patch_width * post_time_compression_frames
        )
        self.post_patch_height = post_patch_height
        self.post_patch_width = post_patch_width
        self.post_time_compression_frames = post_time_compression_frames
        self.patch_size = patch_size

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size, in_channels, inner_dim, text_embed_dim, bias=True
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. 3D positional embeddings
        try:
            spatial_pos_embedding = get_3d_sincos_pos_embed(
                inner_dim,
                (post_patch_width, post_patch_height),
                post_time_compression_frames,
                spatial_interpolation_scale,
                temporal_interpolation_scale,
                output_type="pt",
            )
        except TypeError:
            spatial_pos_embedding = get_3d_sincos_pos_embed(
                inner_dim,
                (post_patch_width, post_patch_height),
                post_time_compression_frames,
                spatial_interpolation_scale,
                temporal_interpolation_scale,
            )
        if isinstance(spatial_pos_embedding, np.ndarray):
            spatial_pos_embedding = torch.from_numpy(spatial_pos_embedding)
        spatial_pos_embedding = spatial_pos_embedding.flatten(0, 1)
        pos_embedding = torch.zeros(
            1, max_text_seq_length + self.num_patches, inner_dim, requires_grad=False
        )
        pos_embedding.data[:, max_text_seq_length:].copy_(spatial_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)

        # 3. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(
            inner_dim, time_embed_dim, timestep_activation_fn
        )

        # 4. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 5. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

        # cross configs
        self.inner_dim = inner_dim
        self.cross_attn_interval = cross_attn_interval
        self.num_cross_attn = num_layers // cross_attn_interval
        self.cross_attn_dim_head = cross_attn_dim_head
        self.cross_attn_num_heads = cross_attn_num_heads
        self.cross_attn_kv_dim = None
        self.cross_attn_in_channels = cross_attn_in_channels
        self.patch_size = patch_size

        self.is_train_cross = is_train_cross
        if is_train_cross:
            self._init_cross_inputs()

    def _init_cross_inputs(self):
        try:
            device = next(self.parameters()).device
            weight_dtype = next(self.parameters()).dtype
        except StopIteration:
            device = torch.device("cpu")
            weight_dtype = torch.float32

        if not hasattr(self, "ref_patch_embed") or self.ref_patch_embed is None:
            self.ref_patch_embed = RefPatchEmbed(
                self.patch_size, self.cross_attn_in_channels, self.inner_dim, bias=True
            ).to(device, dtype=weight_dtype)

        if not hasattr(self, "perceiver_cross_attention") or self.perceiver_cross_attention is None:
            self.perceiver_cross_attention = nn.ModuleList(
                [
                    PerceiverCrossAttention(
                        dim=self.inner_dim,
                        dim_head=self.cross_attn_dim_head,
                        heads=self.cross_attn_num_heads,
                        kv_dim=self.cross_attn_kv_dim,
                    ).to(device, dtype=weight_dtype)
                    for _ in range(self.num_cross_attn)
                ]
            )
        self.is_train_cross = True

    def inject_dora(
        self,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        target_keywords: Tuple[str, ...] = (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        ),
    ):
        """Injects DoRA adapters into all transformer blocks for Stage 1 Geometry Adaptation."""
        from .dora import apply_dora_to_module
        return apply_dora_to_module(
            self.transformer_blocks,
            target_keywords=target_keywords,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    def freeze_base_for_stage1(self):
        """Freezes all base weights and enables grad ONLY on DoRA parameters."""
        for param in self.parameters():
            param.requires_grad = False
        dora_count = 0
        for name, param in self.named_parameters():
            if "lora_A" in name or "lora_B" in name or "magnitude" in name:
                param.requires_grad = True
                dora_count += param.numel()
        print(f"### [Stage 1] Frozen Base Transformer. Active DoRA parameters: {dora_count / 1e6:.2f} M")

    def freeze_for_stage2(self):
        """Freezes base weights and DoRA parameters, enables grad ONLY on Perceiver Cross Attention & RefPatchEmbed."""
        if not hasattr(self, "perceiver_cross_attention") or not hasattr(self, "ref_patch_embed"):
            self._init_cross_inputs()
        for param in self.parameters():
            param.requires_grad = False
        ca_count = 0
        if hasattr(self, "ref_patch_embed") and self.ref_patch_embed is not None:
            for param in self.ref_patch_embed.parameters():
                param.requires_grad = True
                ca_count += param.numel()
        if hasattr(self, "perceiver_cross_attention") and self.perceiver_cross_attention is not None:
            for param in self.perceiver_cross_attention.parameters():
                param.requires_grad = True
                ca_count += param.numel()
        print(f"### [Stage 2] Frozen Base & DoRA. Active Cross-Attention parameters: {ca_count / 1e6:.2f} M")

    def merge_dora(self):
        """Merges all DoRA adapters into the base weights for zero-overhead inference."""
        from .dora import DoRALinear
        count = 0
        for module in self.modules():
            if isinstance(module, DoRALinear):
                module.merge_weights()
                count += 1
        print(f"### Successfully merged {count} DoRA layers into base transformer.")

    def save_dora(self, save_path: str):
        """Saves only the DoRA weights to disk."""
        from .dora import get_dora_state_dict
        dora_dict = get_dora_state_dict(self)
        torch.save(dora_dict, save_path)
        print(f"### Saved DoRA weights ({len(dora_dict)} tensors) to {save_path}")

    def load_dora(self, load_path: str, strict: bool = False):
        """Loads DoRA weights from disk."""
        from .dora import load_dora_state_dict
        state_dict = torch.load(load_path, map_location="cpu")
        load_dora_state_dict(self, state_dict, strict=strict)
        print(f"### Loaded DoRA weights from {load_path}")

    def save_stage2(self, save_path: str):
        """Saves Stage 2 appearance parameters (ref_patch_embed and perceiver_cross_attention) to disk."""
        stage2_dict = {}
        if hasattr(self, "ref_patch_embed") and self.ref_patch_embed is not None:
            stage2_dict["ref_patch_embed"] = self.ref_patch_embed.state_dict()
        if hasattr(self, "perceiver_cross_attention") and self.perceiver_cross_attention is not None:
            stage2_dict["perceiver_cross_attention"] = self.perceiver_cross_attention.state_dict()
        torch.save(stage2_dict, save_path)
        print(f"### Saved Stage 2 Appearance weights to {save_path}")

    def load_stage2(self, load_path: str, strict: bool = True):
        """Loads Stage 2 appearance parameters from disk."""
        state_dict = torch.load(load_path, map_location="cpu")
        if "ref_patch_embed" in state_dict and hasattr(self, "ref_patch_embed") and self.ref_patch_embed is not None:
            self.ref_patch_embed.load_state_dict(state_dict["ref_patch_embed"], strict=strict)
        if "perceiver_cross_attention" in state_dict and hasattr(self, "perceiver_cross_attention") and self.perceiver_cross_attention is not None:
            self.perceiver_cross_attention.load_state_dict(state_dict["perceiver_cross_attention"], strict=strict)
        print(f"### Loaded Stage 2 Appearance weights from {load_path}")

    # Aliases for flexible API compatibility
    enable_dora = inject_dora
    save_dora_checkpoint = save_dora
    load_dora_checkpoint = load_dora
    save_stage2_checkpoint = save_stage2
    load_stage2_checkpoint = load_stage2

    def setup_two_stage_training(self, stage: int = 1):
        """Sets parameter requires_grad flags for Stage 1 or Stage 2."""
        if stage == 1:
            self.freeze_base_for_stage1()
        elif stage == 2:
            self.freeze_for_stage2()
        else:
            raise ValueError(f"Unknown training stage: {stage}. Must be 1 or 2.")

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,  # noise
        encoder_hidden_states: torch.Tensor,  # text
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        inpaint_latents: Optional[torch.Tensor] = None,  # condition
        cross_latents: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
    ):
        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        # [2, 13, 16, 48, 84] cat [2, 13, 17, 48, 84] = [2, 13, 33, 48, 84]
        hidden_states = torch.concat([hidden_states, inpaint_latents], 2)
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        if self.is_train_cross and cross_latents is not None:
            cross_hidden_states = self.ref_patch_embed(cross_latents)
        else:
            cross_hidden_states = None

        # 3. Position embedding
        text_seq_length = encoder_hidden_states.shape[1]
        if not self.config.use_rotary_positional_embeddings:
            seq_length = height * width * num_frames // (self.config.patch_size**2)
            # pos_embeds = self.pos_embedding[:, : text_seq_length + seq_length]
            pos_embeds = self.pos_embedding
            emb_size = hidden_states.size()[-1]
            pos_embeds_without_text = pos_embeds[:, text_seq_length:].view(
                1,
                self.post_time_compression_frames,
                self.post_patch_height,
                self.post_patch_width,
                emb_size,
            )
            pos_embeds_without_text = pos_embeds_without_text.permute([0, 4, 1, 2, 3])
            pos_embeds_without_text = F.interpolate(
                pos_embeds_without_text,
                size=[
                    self.post_time_compression_frames,
                    height // self.config.patch_size,
                    width // self.config.patch_size,
                ],
                mode='trilinear',
                align_corners=False,
            )
            pos_embeds_without_text = pos_embeds_without_text.permute(
                [0, 2, 3, 4, 1]
            ).view(1, -1, emb_size)
            pos_embeds = torch.cat(
                [pos_embeds[:, :text_seq_length], pos_embeds_without_text], dim=1
            )
            pos_embeds = pos_embeds[:, : text_seq_length + seq_length]
            hidden_states = hidden_states + pos_embeds
            hidden_states = self.embedding_dropout(hidden_states)
        # seperate
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 4. Transformer blocks

        ca_idx = 0
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states, encoder_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
            if self.is_train_cross and cross_hidden_states is not None:
                if i % self.cross_attn_interval == 0:
                    ca_module = self.perceiver_cross_attention[ca_idx]
                    if self.training and self.gradient_checkpointing:
                        def create_cross_forward(m):
                            def cf(x_in, h_in):
                                return m(x_in, h_in)
                            return cf
                        cross_out = torch.utils.checkpoint.checkpoint(
                            create_cross_forward(ca_module),
                            cross_hidden_states,
                            hidden_states,
                            use_reentrant=False,
                        )
                    else:
                        cross_out = ca_module(cross_hidden_states, hidden_states)
                    hidden_states = hidden_states + cross_out
                    ca_idx += 1

        # if not self.config.use_rotary_positional_embeddings:
        #     # CogVideoX-2B
        #     hidden_states = self.norm_final(hidden_states)
        # else:
        # use CogVideoX-5B
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, text_seq_length:]

        # 5. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        output = hidden_states.reshape(
            batch_size, num_frames, height // p, width // p, channels, p, p
        )
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        subfolder = kwargs.pop("subfolder", None)
        transformer_additional_kwargs = kwargs.pop("transformer_additional_kwargs", {})
        try:
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, subfolder=subfolder, **kwargs)
        except Exception as e:
            target_path = pretrained_model_name_or_path
            if subfolder:
                target_path = os.path.join(target_path, subfolder)
            print(f"--> [CrossTransformer3DModel] Default from_pretrained failed ({e}), falling back to from_pretrained_2d...")
            return cls.from_pretrained_2d(target_path, subfolder=None, transformer_additional_kwargs=transformer_additional_kwargs)

    @classmethod
    def from_pretrained_2d(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={}
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(
            f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ..."
        )

        config_file = os.path.join(pretrained_model_path, 'config.json')
        config = None
        if os.path.isfile(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
        else:
            # 1. Search parent dir and siblings
            cand_configs = [
                os.path.join(os.path.dirname(pretrained_model_path), 'config.json'),
            ]
            cand_configs.extend(glob.glob(os.path.join(os.path.dirname(pretrained_model_path), "**/config.json"), recursive=True))
            if os.path.exists("/kaggle/input"):
                cand_configs.extend([c for c in glob.glob("/kaggle/input/**/config.json", recursive=True) if "transformer" in c.lower() or "trajectorycrafter" in c.lower() or "cogvideox" in c.lower()])
            for cc in cand_configs:
                if os.path.isfile(cc):
                    try:
                        with open(cc, "r") as f:
                            _cfg = json.load(f)
                        if isinstance(_cfg, dict) and ("num_attention_heads" in _cfg or "num_layers" in _cfg or "transformer" in cc.lower()):
                            config = _cfg
                            print(f"--> [AutoDiscovery] Located config.json at: {cc}")
                            break
                    except Exception:
                        pass
        if config is None:
            print("--> [Notice] config.json not found on disk, using CogVideoX-Fun 5B default config.")
            config = {
                "num_attention_heads": 30,
                "attention_head_dim": 64,
                "in_channels": 16,
                "out_channels": 16,
                "flip_sin_to_cos": True,
                "time_embed_dim": 512,
                "text_embed_dim": 4096,
                "num_layers": 30,
                "dropout": 0.0,
                "attention_bias": True,
                "sample_width": 90,
                "sample_height": 60,
                "sample_frames": 49,
                "patch_size": 2,
                "temporal_compression_ratio": 4,
                "max_text_seq_length": 226,
                "activation_fn": "gelu-approximate",
                "timestep_activation_fn": "silu",
                "norm_elementwise_affine": True,
                "norm_eps": 1e-05,
                "spatial_interpolation_scale": 1.875,
                "temporal_interpolation_scale": 1.0,
                "use_rotary_positional_embeddings": False
            }

        from diffusers.utils import WEIGHTS_NAME

        model = cls.from_config(config, **transformer_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")
        state_dict = {}
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file
            model_files_safetensors = glob.glob(
                os.path.join(pretrained_model_path, "*.safetensors")
            )
            if not model_files_safetensors:
                model_files_safetensors = glob.glob(
                    os.path.join(pretrained_model_path, "**", "*.safetensors"), recursive=True
                )
            for model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]

        if not state_dict:
            from safetensors.torch import load_file
            cand_w_files = (
                glob.glob(os.path.join(os.path.dirname(pretrained_model_path), "*.safetensors")) +
                glob.glob(os.path.join(os.path.dirname(pretrained_model_path), "**", "*.safetensors"), recursive=True) +
                glob.glob(os.path.join(pretrained_model_path, "*.bin")) +
                glob.glob(os.path.join(pretrained_model_path, "*.pt"))
            )
            cand_w_files = [f for f in cand_w_files if "vae" not in f.lower() and "text_encoder" not in f.lower()]
            if cand_w_files:
                print(f"--> [AutoDiscovery] Found transformer weights: {[os.path.basename(f) for f in cand_w_files]}")
                for cw in cand_w_files:
                    if cw.endswith(".safetensors"):
                        _sd = load_file(cw)
                    else:
                        _sd = torch.load(cw, map_location="cpu")
                        if isinstance(_sd, dict) and "state_dict" in _sd:
                            _sd = _sd["state_dict"]
                    for k in _sd:
                        state_dict[k] = _sd[k]

        if (
            model.state_dict()['patch_embed.proj.weight'].size()
            != state_dict['patch_embed.proj.weight'].size()
        ):
            new_shape = model.state_dict()['patch_embed.proj.weight'].size()
            if len(new_shape) == 5:
                state_dict['patch_embed.proj.weight'] = (
                    state_dict['patch_embed.proj.weight']
                    .unsqueeze(2)
                    .expand(new_shape)
                    .clone()
                )
                state_dict['patch_embed.proj.weight'][:, :, :-1] = 0
            else:
                if (
                    model.state_dict()['patch_embed.proj.weight'].size()[1]
                    > state_dict['patch_embed.proj.weight'].size()[1]
                ):
                    model.state_dict()['patch_embed.proj.weight'][
                        :, : state_dict['patch_embed.proj.weight'].size()[1], :, :
                    ] = state_dict['patch_embed.proj.weight']
                    model.state_dict()['patch_embed.proj.weight'][
                        :, state_dict['patch_embed.proj.weight'].size()[1] :, :, :
                    ] = 0
                    state_dict['patch_embed.proj.weight'] = model.state_dict()[
                        'patch_embed.proj.weight'
                    ]
                else:
                    model.state_dict()['patch_embed.proj.weight'][
                        :, :, :, :
                    ] = state_dict['patch_embed.proj.weight'][
                        :,
                        : model.state_dict()['patch_embed.proj.weight'].size()[1],
                        :,
                        :,
                    ]
                    state_dict['patch_embed.proj.weight'] = model.state_dict()[
                        'patch_embed.proj.weight'
                    ]

        tmp_state_dict = {}
        for key in state_dict:
            if (
                key in model.state_dict().keys()
                and model.state_dict()[key].size() == state_dict[key].size()
            ):
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        if len(u) > 0:
            print(f"--> [Model Weights] Loaded state_dict ({len(m)} missing keys, {len(u)} unexpected keys)")
        else:
            print(f"--> [Model Weights] Loaded state_dict ({len(m)} uninitialized/trainable keys, 0 unexpected keys)")

        params = [p.numel() if "mamba" in n else 0 for n, p in model.named_parameters()]
        print(f"### Mamba Parameters: {sum(params) / 1e6} M")

        params = [
            p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()
        ]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")

        return model

    @classmethod
    def from_pretrained_cus(
        cls,
        pretrained_model_path,
        subfolder=None,
        config_path=None,
        transformer_additional_kwargs={},
    ):
        if subfolder:
            config_path = config_path or pretrained_model_path
            config_file = os.path.join(config_path, subfolder, 'config.json')
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        else:
            config_file = os.path.join(
                config_path or pretrained_model_path, 'config.json'
            )

        print(
            f"Loading 3D transformer's pretrained weights from {pretrained_model_path} ..."
        )

        # Check if config file exists
        config = None
        if os.path.isfile(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
        else:
            cand_configs = [
                os.path.join(os.path.dirname(pretrained_model_path), 'config.json'),
            ]
            cand_configs.extend(glob.glob(os.path.join(os.path.dirname(pretrained_model_path), "**/config.json"), recursive=True))
            if os.path.exists("/kaggle/input"):
                cand_configs.extend([c for c in glob.glob("/kaggle/input/**/config.json", recursive=True) if "transformer" in c.lower() or "trajectorycrafter" in c.lower() or "cogvideox" in c.lower()])
            for cc in cand_configs:
                if os.path.isfile(cc):
                    try:
                        with open(cc, "r") as f:
                            _cfg = json.load(f)
                        if isinstance(_cfg, dict) and ("num_attention_heads" in _cfg or "num_layers" in _cfg):
                            config = _cfg
                            print(f"--> [AutoDiscovery] Located config.json at: {cc}")
                            break
                    except Exception:
                        pass
        if config is None:
            print("--> [Notice] config.json not found on disk, using CogVideoX-Fun 5B default config.")
            config = {
                "num_attention_heads": 30,
                "attention_head_dim": 64,
                "in_channels": 16,
                "out_channels": 16,
                "flip_sin_to_cos": True,
                "time_embed_dim": 512,
                "text_embed_dim": 4096,
                "num_layers": 30,
                "dropout": 0.0,
                "attention_bias": True,
                "sample_width": 90,
                "sample_height": 60,
                "sample_frames": 49,
                "patch_size": 2,
                "temporal_compression_ratio": 4,
                "max_text_seq_length": 226,
                "activation_fn": "gelu-approximate",
                "timestep_activation_fn": "silu",
                "norm_elementwise_affine": True,
                "norm_eps": 1e-05,
                "spatial_interpolation_scale": 1.875,
                "temporal_interpolation_scale": 1.0,
                "use_rotary_positional_embeddings": False
            }

        from diffusers.utils import WEIGHTS_NAME

        model = cls.from_config(config, **transformer_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file

            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file

            model_files_safetensors = glob.glob(
                os.path.join(pretrained_model_path, "*.safetensors")
            )
            state_dict = {}
            for model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]

        if (
            model.state_dict()['patch_embed.proj.weight'].size()
            != state_dict['patch_embed.proj.weight'].size()
        ):
            new_shape = model.state_dict()['patch_embed.proj.weight'].size()
            if len(new_shape) == 5:
                state_dict['patch_embed.proj.weight'] = (
                    state_dict['patch_embed.proj.weight']
                    .unsqueeze(2)
                    .expand(new_shape)
                    .clone()
                )
                state_dict['patch_embed.proj.weight'][:, :, :-1] = 0
            else:
                if (
                    model.state_dict()['patch_embed.proj.weight'].size()[1]
                    > state_dict['patch_embed.proj.weight'].size()[1]
                ):
                    model.state_dict()['patch_embed.proj.weight'][
                        :, : state_dict['patch_embed.proj.weight'].size()[1], :, :
                    ] = state_dict['patch_embed.proj.weight']
                    model.state_dict()['patch_embed.proj.weight'][
                        :, state_dict['patch_embed.proj.weight'].size()[1] :, :, :
                    ] = 0
                    state_dict['patch_embed.proj.weight'] = model.state_dict()[
                        'patch_embed.proj.weight'
                    ]
                else:
                    model.state_dict()['patch_embed.proj.weight'][
                        :, :, :, :
                    ] = state_dict['patch_embed.proj.weight'][
                        :,
                        : model.state_dict()['patch_embed.proj.weight'].size()[1],
                        :,
                        :,
                    ]
                    state_dict['patch_embed.proj.weight'] = model.state_dict()[
                        'patch_embed.proj.weight'
                    ]

        tmp_state_dict = {}
        for key in state_dict:
            if (
                key in model.state_dict().keys()
                and model.state_dict()[key].size() == state_dict[key].size()
            ):
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        if len(u) > 0:
            print(f"--> [Model Weights] Loaded state_dict ({len(m)} missing keys, {len(u)} unexpected keys)")
        else:
            print(f"--> [Model Weights] Loaded state_dict ({len(m)} uninitialized/trainable keys, 0 unexpected keys)")

        params = [p.numel() if "mamba" in n else 0 for n, p in model.named_parameters()]
        print(f"### Mamba Parameters: {sum(params) / 1e6} M")

        params = [
            p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()
        ]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")

        return model


if __name__ == '__main__':
    device = "cuda:0"
    weight_dtype = torch.bfloat16
    model_path = "/group/40075/wangboyu/CogVideoX-Fun/CogVideoX-Fun-V1.1-5b-InP"

    transformer_additional_kwargs = {
        'is_train_cross': True,
        'cross_attn_in_channels': 16,
        'cross_attn_interval': 2,
        'cross_attn_dim_head': 128,
        'cross_attn_num_heads': 16,
    }

    transformer = CrossTransformer3DModel.from_pretrained_2d(
        model_path,
        subfolder="transformer",
        transformer_additional_kwargs=transformer_additional_kwargs,
    )

    transformer.to(device, dtype=weight_dtype)
    for param in transformer.parameters():
        param.requires_grad = False
    transformer.eval()

    b = 1
    dim = 16
    noisy_latents = torch.ones(b, 13, dim, 60, 90).to(device, dtype=weight_dtype)
    inpaint_latents = torch.ones(b, 13, dim + 1, 60, 90).to(device, dtype=weight_dtype)
    # cross_latents          = torch.ones(b, 13, dim, 60, 90).to(device, dtype=weight_dtype)
    cross_latents = torch.ones(b, 1, dim, 60, 90).to(device, dtype=weight_dtype)
    prompt_embeds = torch.ones(b, 226, 4096).to(device, dtype=weight_dtype)
    image_rotary_emb = (
        torch.ones(17550, 64).to(device, dtype=weight_dtype),
        torch.ones(17550, 64).to(device, dtype=weight_dtype),
    )
    timesteps = torch.tensor([311]).to(device, dtype=weight_dtype)
    assert len(timesteps) == b

    model_output = transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps,
        inpaint_latents=inpaint_latents,
        cross_latents=cross_latents,
        image_rotary_emb=image_rotary_emb,
        return_dict=False,
    )[0]

    print(model_output)
