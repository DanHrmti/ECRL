import torch
import torch.nn as nn
import torch.nn.functional as F

from slot_attention.utils import linear
from slot_attention.transformer import PositionalEncoding, TransformerDecoder

from slot_attention.slot_attention_model import SlotAttention


def conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
    weight_init="xavier",
):

    m = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
    )

    if weight_init == "kaiming":
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(m.weight)

    if bias:
        nn.init.zeros_(m.bias)

    return m


def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):

    eps = torch.finfo(logits.dtype).tiny

    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau

    y_soft = F.softmax(gumbels, dim)

    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.m = conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            weight_init="kaiming",
        )
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = self.m(x)
        return F.relu(F.group_norm(x, 1, self.weight, self.bias))


class dVAE(nn.Module):
    def __init__(self, vocab_size, img_channels=3):
        super().__init__()

        self.encoder = nn.Sequential(
            Conv2dBlock(img_channels, 64, 4, 4),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            conv2d(64, vocab_size, 1),
        )

        self.decoder = nn.Sequential(
            Conv2dBlock(vocab_size, 64, 1),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            conv2d(64, img_channels, 1),
        )


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs


class SLATE(nn.Module):
    def __init__(
        self,
        num_slots,
        vocab_size,
        d_model,
        resolution,
        num_iterations,
        slot_size,
        mlp_hidden_size,
        num_heads,
        dropout,
        num_dec_blocks,
    ):
        super().__init__()
        self.supports_masks = False

        self.num_slots = num_slots
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.dvae = dVAE(vocab_size)

        self.positional_encoder = PositionalEncoding(
            1 + (resolution[0] // 4) ** 2, d_model, dropout
        )

        self.slot_attn = SlotAttention(
            d_model,
            num_iterations,
            num_slots,
            slot_size,
            mlp_hidden_size,
            do_input_mlp=True,
        )

        self.dictionary = OneHotDictionary(vocab_size + 1, d_model)
        self.slot_proj = linear(slot_size, d_model, bias=False)

        self.tf_dec = TransformerDecoder(
            num_dec_blocks, (resolution[0] // 4) ** 2, d_model, num_heads, dropout
        ).train()

        self.out = linear(d_model, vocab_size, bias=False)

    def forward(self, image, tau, hard):
        """
        image: batch_size x img_channels x H x W
        """

        B, C, H, W = image.size()

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(image), dim=1)
        _, _, H_enc, W_enc = z_logits.size()
        z = gumbel_softmax(z_logits, tau, hard, dim=1)

        # dvae recon
        recon = self.dvae.decoder(z)
        mse = ((image - recon) ** 2).sum() / B

        # hard z
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()

        # target tokens for transformer
        z_transformer_target = z_hard.permute(0, 2, 3, 1).flatten(
            start_dim=1, end_dim=2
        )

        # add BOS token
        z_transformer_input = torch.cat(
            [torch.zeros_like(z_transformer_target[..., :1]), z_transformer_target],
            dim=-1,
        )
        z_transformer_input = torch.cat(
            [torch.zeros_like(z_transformer_input[..., :1, :]), z_transformer_input],
            dim=-2,
        )
        z_transformer_input[:, 0, 0] = 1.0

        # tokens to embeddings
        emb_input = self.dictionary(z_transformer_input)
        emb_input = self.positional_encoder(emb_input)

        # apply slot attention
        slots, attns = self.slot_attn(emb_input[:, 1:], return_attns=True)
        attns = attns.transpose(-1, -2)
        attns = (
            attns.reshape(B, self.num_slots, 1, H_enc, W_enc)
            .repeat_interleave(H // H_enc, dim=-2)
            .repeat_interleave(W // W_enc, dim=-1)
        )
        attns = image.unsqueeze(1) * attns + 1.0 - attns
        # `attns` has shape [batch_size, num_slots, channels, height, width]

        # apply transformer
        slots = self.slot_proj(slots)
        decoder_output = self.tf_dec(emb_input[:, :-1], slots)
        pred = self.out(decoder_output)
        cross_entropy = (
            -(z_transformer_target * torch.log_softmax(pred, dim=-1))
            .flatten(start_dim=1)
            .sum(-1)
            .mean()
        )

        return (recon.clamp(0.0, 1.0), cross_entropy, mse, attns)

    def loss_function(self, input, tau, hard=False):
        _, cross_entropy, mse, _ = self.forward(input, tau, hard)
        return {
            "loss": cross_entropy + mse,
            "cross_entropy": cross_entropy,
            "mse": mse,
            "tau": torch.tensor(tau),
        }

    def reconstruct_autoregressive(self, image, eval=False):
        """
        image: batch_size x img_channels x H x W
        """

        gen_len = (image.size(-1) // 4) ** 2

        B, C, H, W = image.size()

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(image), dim=1)
        _, _, H_enc, W_enc = z_logits.size()

        # hard z
        z_hard = torch.argmax(z_logits, axis=1)
        z_hard = (
            F.one_hot(z_hard, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()
        )
        one_hot_tokens = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        # add BOS token
        one_hot_tokens = torch.cat(
            [torch.zeros_like(one_hot_tokens[..., :1]), one_hot_tokens], dim=-1
        )
        one_hot_tokens = torch.cat(
            [torch.zeros_like(one_hot_tokens[..., :1, :]), one_hot_tokens], dim=-2
        )
        one_hot_tokens[:, 0, 0] = 1.0

        # tokens to embeddings
        emb_input = self.dictionary(one_hot_tokens)
        emb_input = self.positional_encoder(emb_input)

        # slot attention
        slots, attns = self.slot_attn(emb_input[:, 1:], return_attns=True)
        attns = attns.transpose(-1, -2)
        attns = (
            attns.reshape(B, self.num_slots, 1, H_enc, W_enc)
            .repeat_interleave(H // H_enc, dim=-2)
            .repeat_interleave(W // W_enc, dim=-1)
        )
        attns = image.unsqueeze(1) * attns + (1.0 - attns)
        slots = self.slot_proj(slots)

        # generate image tokens auto-regressively
        z_gen = z_hard.new_zeros(0)
        z_transformer_input = z_hard.new_zeros(B, 1, self.vocab_size + 1)
        z_transformer_input[..., 0] = 1.0
        for t in range(gen_len):
            decoder_output = self.tf_dec(
                self.positional_encoder(self.dictionary(z_transformer_input)), slots
            )
            z_next = F.one_hot(
                self.out(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size
            )
            z_gen = torch.cat((z_gen, z_next), dim=1)
            z_transformer_input = torch.cat(
                [
                    z_transformer_input,
                    torch.cat([torch.zeros_like(z_next[:, :, :1]), z_next], dim=-1),
                ],
                dim=1,
            )

        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
        recon_transformer = self.dvae.decoder(z_gen)

        if eval:
            return recon_transformer.clamp(0.0, 1.0), attns

        return recon_transformer.clamp(0.0, 1.0)
