import copy

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _zero_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _parse_dw_ks(configs) -> list[int]:
    default_kernels = [11, 15, 21, 29, 39, 51]
    n_layers = int(getattr(configs, "e_layers", len(default_kernels)))
    raw = getattr(configs, "dw_ks", None)
    if raw is None:
        if n_layers <= len(default_kernels):
            return default_kernels[:n_layers]
        last = default_kernels[-1]
        extra = [last + 12 * idx for idx in range(1, n_layers - len(default_kernels) + 1)]
        return default_kernels + extra
    if isinstance(raw, str):
        kernels = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        kernels = [int(value) for value in raw]
    if len(kernels) != n_layers:
        raise ValueError(
            f"ConvTimeNet requires len(dw_ks) == e_layers, got {len(kernels)} and {n_layers}"
        )
    return kernels


def _parse_patch_stride(configs, patch_len: int) -> int:
    raw_stride = getattr(configs, "patch_sd", 0.5)
    if float(raw_stride) <= 1:
        return max(1, int(patch_len * float(raw_stride)))
    return int(raw_stride)


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: Tensor, mode: str) -> Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"Unsupported RevIN mode: {mode}")

    def _get_statistics(self, x: Tensor) -> None:
        reduce_dims = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1:, :].detach()
        else:
            self.mean = torch.mean(x, dim=reduce_dims, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=reduce_dims, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: Tensor) -> Tensor:
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: Tensor) -> Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            return x + self.last
        return x + self.mean


class BoxCoder(nn.Module):
    def __init__(self, patch_count: int, patch_stride: int, patch_size: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.patch_count = patch_count
        self.patch_stride = patch_stride
        self.size_bias = (patch_size - 1) / 2
        anchors = [
            index * patch_stride + 0.5 * (patch_size - 1)
            for index in range(patch_count)
        ]
        self.register_buffer("anchor", torch.tensor(anchors, dtype=torch.float32))

    def forward(self, boxes: Tensor) -> tuple[Tensor, Tensor]:
        bounds = self.decode(boxes)
        points = self.meshgrid(bounds)
        return points, bounds

    def decode(self, rel_codes: Tensor) -> Tensor:
        dx = rel_codes[:, :, :, 0]
        ds = torch.relu(rel_codes[:, :, :, 1] + self.size_bias)
        pred_boxes = torch.zeros_like(rel_codes)
        ref_x = self.anchor.view(1, self.anchor.shape[0], 1)
        pred_boxes[:, :, :, 0] = dx + ref_x - ds
        pred_boxes[:, :, :, 1] = dx + ref_x + ds
        pred_boxes = pred_boxes / (self.seq_len - 1)
        return pred_boxes.clamp_(min=0.0, max=1.0)

    def meshgrid(self, boxes: Tensor) -> Tensor:
        batch_size, patch_count, channels = boxes.shape[:3]
        device = boxes.device
        channel_boxes = torch.zeros((batch_size, patch_count, 2), device=device)
        channel_boxes[:, :, 1] = 1.0
        xs = boxes.view(batch_size * patch_count, channels, 2)
        xs = F.interpolate(xs, size=self.patch_size, mode="linear", align_corners=True)
        ys = F.interpolate(channel_boxes, size=channels, mode="linear", align_corners=True)
        xs = xs.view(batch_size, patch_count, channels, self.patch_size, 1)
        ys = ys.unsqueeze(3).expand(batch_size, patch_count, channels, self.patch_size).unsqueeze(-1)
        return torch.stack([xs, ys], dim=-1)


class OffsetPredictor(nn.Module):
    def __init__(self, in_feats: int, patch_size: int, stride: int, zero_init: bool = True):
        super().__init__()
        self.channel = in_feats
        self.patch_size = patch_size
        self.stride = stride
        self.offset_predictor = nn.Sequential(
            nn.Conv1d(1, 64, patch_size, stride=stride, padding=0),
            nn.GELU(),
            nn.Conv1d(64, 2, 1, 1, padding=0),
        )
        if zero_init:
            self.offset_predictor.apply(_zero_init)

    def forward(self, x: Tensor) -> Tensor:
        patch_x = x.unsqueeze(1).permute(0, 1, 3, 2)
        patch_x = F.unfold(
            patch_x,
            kernel_size=(self.patch_size, self.channel),
            stride=self.stride,
        ).permute(0, 2, 1)
        batch_size, patch_count = patch_x.shape[:2]
        patch_x = patch_x.contiguous().view(batch_size, patch_count, self.patch_size, self.channel)
        patch_x = patch_x.permute(0, 1, 3, 2)
        patch_x = patch_x.contiguous().view(batch_size * patch_count * self.channel, 1, self.patch_size)
        pred_offset = self.offset_predictor(patch_x)
        return pred_offset.view(batch_size, patch_count, self.channel, 2).contiguous()


class DepatchSampling(nn.Module):
    def __init__(self, in_feats: int, seq_len: int, patch_size: int, stride: int):
        super().__init__()
        self.in_feats = in_feats
        self.patch_size = patch_size
        self.patch_count = (seq_len - patch_size) // stride + 1
        self.offset_predictor = OffsetPredictor(in_feats, patch_size, stride)
        self.box_coder = BoxCoder(self.patch_count, stride, patch_size, seq_len)

    def forward(self, x: Tensor) -> Tensor:
        image = x.unsqueeze(1)
        batch_size = image.shape[0]
        sampling_locations, _ = self.box_coder(self.offset_predictor(x))
        sampling_locations = sampling_locations.view(
            batch_size, self.patch_count * self.in_feats, self.patch_size, 2
        )
        sampling_locations = (sampling_locations - 0.5) * 2
        output = F.grid_sample(image, sampling_locations, align_corners=True)
        output = output.view(batch_size, self.patch_count, self.in_feats, self.patch_size)
        return output.permute(0, 2, 1, 3).contiguous()


class SublayerConnection(nn.Module):
    def __init__(self, enable_res_parameter: bool, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.scale = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x: Tensor, out_x: Tensor) -> Tensor:
        if not self.enable:
            return x + self.dropout(out_x)
        return x + self.dropout(self.scale * out_x)


class ConvEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int = 256,
        kernel_size: int = 9,
        dropout: float = 0.1,
        activation: str = "gelu",
        enable_res_param: bool = True,
        norm: str = "batch",
        re_param: bool = True,
        small_ks: int = 3,
    ):
        super().__init__()
        self.norm_type = norm
        self.re_param = re_param
        if not re_param:
            self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size, 1, padding="same", groups=d_model)
        else:
            self.large_ks = kernel_size
            self.small_ks = small_ks
            self.dw_conv_large = nn.Conv1d(
                d_model, d_model, kernel_size, stride=1, padding="same", groups=d_model
            )
            self.dw_conv_small = nn.Conv1d(
                d_model, d_model, small_ks, stride=1, padding="same", groups=d_model
            )
        self.dw_act = _activation(activation)
        self.sublayer1 = SublayerConnection(enable_res_param, dropout)
        self.dw_norm = nn.BatchNorm1d(d_model) if norm == "batch" else nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1, 1),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, 1, 1),
        )
        self.sublayer2 = SublayerConnection(enable_res_param, dropout)
        self.ff_norm = nn.BatchNorm1d(d_model) if norm == "batch" else nn.LayerNorm(d_model)

    def _merge_reparam_conv(self) -> tuple[Tensor, Tensor]:
        left_pad = (self.large_ks - self.small_ks) // 2
        right_pad = (self.large_ks - self.small_ks) - left_pad
        weight = self.dw_conv_large.weight + F.pad(self.dw_conv_small.weight, (left_pad, right_pad), value=0)
        bias = self.dw_conv_large.bias + self.dw_conv_small.bias
        return weight, bias

    def _apply_norm(self, x: Tensor, norm: nn.Module) -> Tensor:
        if self.norm_type == "batch":
            return norm(x)
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, src: Tensor) -> Tensor:
        if not self.re_param:
            out_x = self.dw_conv(src)
        elif self.training:
            out_x = self.dw_conv_large(src) + self.dw_conv_small(src)
        else:
            weight, bias = self._merge_reparam_conv()
            out_x = F.conv1d(src, weight, bias, stride=1, padding="same", groups=src.shape[1])
        src = self.sublayer1(src, self.dw_act(out_x))
        src = self._apply_norm(src, self.dw_norm)
        src2 = self.ff(src)
        src2 = self.sublayer2(src, src2)
        return self._apply_norm(src2, self.ff_norm)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        kernel_sizes: list[int],
        d_model: int,
        d_ff: int = 256,
        norm: str = "batch",
        dropout: float = 0.0,
        activation: str = "gelu",
        enable_res_param: bool = True,
        re_param: bool = True,
        re_param_kernel: int = 3,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConvEncoderLayer(
                    d_model,
                    d_ff=d_ff,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    activation=activation,
                    enable_res_param=enable_res_param,
                    norm=norm,
                    re_param=re_param,
                    small_ks=re_param_kernel,
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, src: Tensor) -> Tensor:
        out = src
        for layer in self.layers:
            out = layer(out)
        return out


class ConviEncoder(nn.Module):
    def __init__(
        self,
        patch_num: int,
        patch_len: int,
        kernel_sizes: list[int],
        d_model: int = 128,
        d_ff: int = 256,
        norm: str = "batch",
        dropout: float = 0.0,
        act: str = "gelu",
        enable_res_param: bool = True,
        re_param: bool = True,
        re_param_kernel: int = 3,
    ):
        super().__init__()
        self.proj = nn.Linear(patch_len, d_model)
        self.encoder = ConvEncoder(
            kernel_sizes,
            d_model,
            d_ff=d_ff,
            norm=norm,
            dropout=dropout,
            activation=act,
            enable_res_param=enable_res_param,
            re_param=re_param,
            re_param_kernel=re_param_kernel,
        )

    def forward(self, x: Tensor) -> Tensor:
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)
        x = self.proj(x)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = self.encoder(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.reshape(-1, n_vars, x.shape[-2], x.shape[-1])
        return x.permute(0, 1, 3, 2)


class FlattenHead(nn.Module):
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.linear(self.flatten(x)))


class ConvTimeNetBackbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        kernel_sizes: list[int],
        d_model: int,
        d_ff: int,
        norm: str,
        dropout: float,
        act: str,
        head_dropout: float,
        padding_patch: str | None,
        revin: bool,
        affine: bool,
        subtract_last: bool,
        enable_res_param: bool,
        re_param: bool,
        re_param_kernel: int,
    ):
        super().__init__()
        self.revin = revin
        if revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        seq_len = (patch_num - 1) * stride + patch_len
        self.depatch = DepatchSampling(c_in, seq_len, patch_len, stride)
        self.backbone = ConviEncoder(
            patch_num=patch_num,
            patch_len=patch_len,
            kernel_sizes=kernel_sizes,
            d_model=d_model,
            d_ff=d_ff,
            norm=norm,
            dropout=dropout,
            act=act,
            enable_res_param=enable_res_param,
            re_param=re_param,
            re_param_kernel=re_param_kernel,
        )
        self.head = FlattenHead(c_in, d_model * patch_num, target_window, head_dropout=head_dropout)

    def forward(self, z: Tensor) -> Tensor:
        if self.revin:
            z = self.revin_layer(z.transpose(1, 2), "norm").transpose(1, 2)
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = self.depatch(z)
        z = z.permute(0, 1, 3, 2)
        z = self.backbone(z)
        z = self.head(z)
        if self.revin:
            z = self.revin_layer(z.transpose(1, 2), "denorm").transpose(1, 2)
        return z


class Model(nn.Module):
    def __init__(self, configs, norm: str = "batch", act: str = "gelu"):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        patch_len = int(getattr(configs, "patch_ks", 32))
        stride = _parse_patch_stride(configs, patch_len)
        dw_ks = _parse_dw_ks(configs)

        self.model = ConvTimeNetBackbone(
            c_in=configs.enc_in,
            context_window=configs.seq_len,
            target_window=configs.pred_len,
            patch_len=patch_len,
            stride=stride,
            kernel_sizes=dw_ks,
            d_model=int(getattr(configs, "d_model", 64)),
            d_ff=int(getattr(configs, "d_ff", 256)),
            norm=norm,
            dropout=float(getattr(configs, "dropout", 0.05)),
            act=act,
            head_dropout=float(getattr(configs, "head_dropout", 0.0)),
            padding_patch=getattr(configs, "padding_patch", "end"),
            revin=bool(getattr(configs, "revin", 1)),
            affine=bool(getattr(configs, "affine", 0)),
            subtract_last=bool(getattr(configs, "subtract_last", 0)),
            enable_res_param=bool(getattr(configs, "enable_res_param", 1)),
            re_param=bool(getattr(configs, "re_param", 1)),
            re_param_kernel=int(getattr(configs, "re_param_kernel", 3)),
        )

    def forecast(self, x_enc: Tensor) -> Tensor:
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = self.model(x_enc)
        return x_enc.permute(0, 2, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len :, :]
        return None
