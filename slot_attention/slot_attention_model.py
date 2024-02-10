from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from slot_attention.utils import (
    assert_shape,
    build_grid,
    conv_transpose_out_shape,
    linear,
    gru_cell,
)


class SlotAttention(nn.Module):
    def __init__(
        self,
        in_features,
        num_iterations,
        num_slots,
        slot_size,
        mlp_hidden_size,
        epsilon=1e-8,
        do_input_mlp=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.do_input_mlp = do_input_mlp

        if self.do_input_mlp:
            self.input_layer_norm = nn.LayerNorm(self.in_features)
            self.input_mlp = nn.Sequential(
                linear(self.in_features, self.in_features, weight_init="kaiming"),
                nn.ReLU(),
                linear(self.in_features, self.in_features),
            )

        self.norm_inputs = nn.LayerNorm(self.in_features)
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = linear(self.in_features, self.slot_size, bias=False)
        self.project_v = linear(self.in_features, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = gru_cell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            linear(self.slot_size, self.mlp_hidden_size, weight_init="kaiming"),
            nn.ReLU(),
            linear(self.mlp_hidden_size, self.slot_size),
        )

        # Parameters for Gaussian init (shared by all slots).
        self.slot_mu = nn.Parameter(torch.zeros(1, 1, self.slot_size))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, self.slot_size))
        nn.init.xavier_uniform_(self.slot_mu, gain=nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(
            self.slot_log_sigma,
            gain=nn.init.calculate_gain("linear"),
        )

    def step(self, slots, k, v, batch_size, num_inputs):
        slots_prev = slots
        slots = self.norm_slots(slots)

        # Attention.
        q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
        assert_shape(q.size(), (batch_size, self.num_slots, self.slot_size))
        q *= self.slot_size**-0.5  # Normalization
        attn_logits = torch.matmul(k, q.transpose(2, 1))
        attn = F.softmax(attn_logits, dim=-1)
        # `attn` has shape: [batch_size, num_inputs, num_slots].
        assert_shape(attn.size(), (batch_size, num_inputs, self.num_slots))
        attn_vis = attn.clone()

        # Weighted mean.
        attn = attn + self.epsilon
        attn = attn / torch.sum(attn, dim=-2, keepdim=True)
        updates = torch.matmul(attn.transpose(-1, -2), v)
        # `updates` has shape: [batch_size, num_slots, slot_size].
        assert_shape(updates.size(), (batch_size, self.num_slots, self.slot_size))

        # Slot update.
        # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
        slots = self.gru(
            updates.view(batch_size * self.num_slots, self.slot_size),
            slots_prev.view(batch_size * self.num_slots, self.slot_size),
        )
        slots = slots.view(batch_size, self.num_slots, self.slot_size)
        assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
        slots = slots + self.mlp(self.norm_mlp(slots))
        assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))

        return slots, attn_vis

    def forward(self, inputs: torch.Tensor, return_attns=False):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        # `inputs` also has shape [batch_size, enc_height * enc_width, cnn_hidden_size].
        batch_size, num_inputs, inputs_size = inputs.shape

        if self.do_input_mlp:
            inputs = self.input_mlp(self.input_layer_norm(inputs))

        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k.size(), (batch_size, num_inputs, self.slot_size))
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v.size(), (batch_size, num_inputs, self.slot_size))

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots_init = inputs.new_empty(
            batch_size, self.num_slots, self.slot_size
        ).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots_init

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots, attn_vis = self.step(slots, k, v, batch_size, num_inputs)
        # Detach slots from the current graph and compute one more step.
        # This is implicit slot attention from https://cocosci.princeton.edu/papers/chang2022objfixed.pdf
        slots, attn_vis = self.step(slots.detach(), k, v, batch_size, num_inputs)
        if return_attns:
            return slots, attn_vis
        else:
            return slots


class SlotAttentionModel(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int],
        num_slots: int,
        num_iterations,
        in_channels: int = 3,
        kernel_size: int = 5,
        slot_size: int = 64,
        mlp_hidden_size: int = 128,
        hidden_dims: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_resolution: Tuple[int, int] = (8, 8),  # (8, 8)
        use_separation_loss=False,
        use_area_loss=False,
    ):
        super().__init__()
        self.supports_masks = True

        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.hidden_dims = hidden_dims
        self.decoder_resolution = decoder_resolution
        self.out_features = self.hidden_dims[-1]
        self.use_separation_loss = use_separation_loss
        self.use_area_loss = use_area_loss

        assert self.hidden_dims[-1] == self.slot_size  # DH addition

        modules = []
        channels = self.in_channels
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels=h_dim,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2,
                    ),
                    nn.LeakyReLU(),
                )
            )
            channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(
            self.in_channels, self.out_features, resolution
        )
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.LeakyReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

        # Build Decoder
        modules = []

        in_size = decoder_resolution[0]
        out_size = in_size

        for i in range(len(self.hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i - 1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)

        assert_shape(
            resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. Try changing `decoder_resolution`.",
        )

        # same convolutions
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.out_features,
                    self.out_features,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    output_padding=0,
                ),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(
                    self.out_features,
                    4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    output_padding=0,
                ),
            )
        )

        assert_shape(resolution, (out_size, out_size), message="")

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(
            self.in_channels, self.out_features, self.decoder_resolution
        )

        self.slot_attention = SlotAttention(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.mlp_hidden_size,
        )

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        encoder_out = self.encoder(x)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, filter_size, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, filter_size, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]

        slots = self.slot_attention(encoder_out)
        assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
        # `slots` has shape: [batch_size, num_slots, slot_size].
        batch_size, num_slots, slot_size = slots.shape

        slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
        decoder_in = slots.repeat(
            1, 1, self.decoder_resolution[0], self.decoder_resolution[1]
        )

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
        assert_shape(
            out.size(), (batch_size * num_slots, num_channels + 1, height, width)
        )

        # Perform researchers' `unstack_and_split` using `torch.view`.
        out = out.view(batch_size, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]
        masks = out[:, :, -1:, :, :]
        # Normalize alpha masks over slots.
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks, slots

    def loss_function(
        self, input, separation_tau=None, area_tau=None, global_step=None
    ):
        recon_combined, recons, masks, slots = self.forward(input)
        # `masks` has shape [batch_size, num_entries, channels, height, width]
        mse_loss = F.mse_loss(recon_combined, input)

        loss = mse_loss
        to_return = {"loss": loss}
        if self.use_area_loss:
            max_num_objects = self.num_slots - 1
            width, height = masks.shape[-1], masks.shape[-2]
            one_object_area = 9 * 9
            background_size = width * height - max_num_objects * one_object_area
            slot_area = torch.sum(masks.squeeze(), dim=(-1, -2))

            batch_size, num_slots = slot_area.shape[0], slot_area.shape[1]
            area_loss = 0
            for batch_idx in range(batch_size):
                for slot_idx in range(num_slots):
                    area = slot_area[batch_idx, slot_idx]
                    area_loss += min(
                        (area - 2 * one_object_area) ** 2,
                        max(background_size - area, 0) * (background_size - area),
                    ) / (2 * one_object_area)**2
            
            area_loss /= batch_size * num_slots
            loss += area_loss * area_tau
            to_return["loss"] = loss
            to_return["area_loss"] = area_loss
            to_return["area_tau"] = torch.tensor(area_tau)
        if self.use_separation_loss:
            if self.use_separation_loss == "max":
                separation_loss = 1 - torch.mean(torch.max(masks, dim=1).values.float())
            elif self.use_separation_loss == "entropy":
                entropy = torch.special.entr(masks + 1e-8)
                separation_loss = torch.mean(entropy.sum(dim=1))

            loss += separation_loss * separation_tau
            to_return["loss"] = loss
            to_return["mse_loss"] = mse_loss
            to_return["separation_loss"] = separation_loss
            to_return["separation_tau"] = torch.tensor(separation_tau)
        return to_return, masks


class SoftPositionEmbed(nn.Module):
    def __init__(
        self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]
    ):
        super().__init__()
        self.dense = nn.Linear(in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: torch.Tensor):
        # Permute to move num_channels to 1st dimension. PyTorch layers need
        # num_channels as 1st dimension, tensorflow needs num_channels last.
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj
