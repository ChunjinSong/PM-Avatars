import torch
import torch.nn as nn
import numpy as np


def sine_init(m, w0):
    with torch.no_grad():
        if hasattr(m.layer, 'weight'):
            num_input = m.layer.weight.size(-1)
            m.layer.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m.layer, 'weight'):
            num_input = m.layer.weight.size(-1)
            m.layer.weight.uniform_(-1.0 / num_input, 1.0 / num_input)


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        return torch.sin(freq * x + phase_shift)


class SirenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, w):
        x = self.layer(x)
        x = x * w
        return torch.sin(x)


class LinearLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.layer(x)
        return x


class PoseExtractor(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, D=3, sin_w0=2.0):
        super().__init__()
        sin_dims = [dim_in]
        for d in range(D):
            sin_dims.append(dim_hidden)
        sin_dims.append(dim_out)
        self.num_sin_layers = len(sin_dims)
        self.sin_w0 = sin_w0
        for layer in range(0, self.num_sin_layers - 2):
            setattr(self, "sin_lin" + str(layer), SirenLayer(sin_dims[layer], sin_dims[layer + 1]))

        layer = self.num_sin_layers - 2
        setattr(self, "sin_lin" + str(layer), LinearLayer(sin_dims[layer], sin_dims[layer + 1]))

        self.init_siren()
        self.out_dim = sin_dims[-1]

    def init_siren(self):
        for layer in range(0, self.num_sin_layers - 1):
            lin = getattr(self, "sin_lin" + str(layer))
            sine_init(lin, w0=self.sin_w0)

    def forward(self, x):
        for layer in range(0, self.num_sin_layers - 2):
            sin_lin = getattr(self, "sin_lin" + str(layer))
            x = sin_lin(x, self.sin_w0)

        layer = self.num_sin_layers - 2
        lin = getattr(self, "sin_lin" + str(layer))
        x = lin(x)

        frequencies = x[..., :x.shape[-1] // 2]
        phase_shifts = x[..., x.shape[-1] // 2:]
        return frequencies, phase_shifts


class Modulator(nn.Module):

    def __init__(self, dim_in, dim_out, D, W, sin_w0=25.0):
        super().__init__()
        sin_dims = [dim_in]
        for d in range(D - 1):
            sin_dims.append(W)
        sin_dims.append(dim_out)
        self.num_sin_layers = len(sin_dims)
        self.sin_w0 = sin_w0

        for layer in range(0, self.num_sin_layers - 1):
            setattr(self, "sin_lin" + str(layer), FiLMLayer(sin_dims[layer], sin_dims[layer + 1]))
        self.init_siren()

    def init_siren(self):
        for layer in range(0, self.num_sin_layers - 1):
            lin = getattr(self, "sin_lin" + str(layer))

            sine_init(lin, w0=self.sin_w0)

    def forward(self, input, feat_deform):
        frequencies = feat_deform[0]
        phase_shifts = feat_deform[1]

        if self.sin_w0 > 15.0:
            frequencies = frequencies * 15 + 30
        else:
            frequencies = frequencies * self.sin_w0 + self.sin_w0 * 2.0
        x = input
        x_out = []
        for layer in range(0, self.num_sin_layers - 1):
            sin_lin = getattr(self, "sin_lin" + str(layer))
            x = sin_lin(x, frequencies[layer], phase_shifts[layer])
            x_out.append(x)

        x_out = torch.concat(x_out, dim=-1)
        return x_out


class PointEnc(nn.Module):
    def __init__(self,
                 dim_in=3,
                 dim_out=32,
                 sin_w0=2.0
                 ):
        super().__init__()

        sin_dims = [dim_in]
        sin_dims.append(dim_out)
        self.dims = dim_out
        self.num_sin_layers = len(sin_dims)
        self.sin_w0 = sin_w0

        for layer in range(0, self.num_sin_layers - 1):
            out_dim = sin_dims[layer + 1]
            setattr(self, "sin_lin" + str(layer), SirenLayer(sin_dims[layer], out_dim))

        self.init_siren()

        self.out_dim = sin_dims[-1]

    def init_siren(self):
        for layer in range(0, self.num_sin_layers - 1):
            lin = getattr(self, "sin_lin" + str(layer))
            first_layer_sine_init(lin)

    def forward(self, x):
        for layer in range(0, self.num_sin_layers - 1):
            sin_lin = getattr(self, "sin_lin" + str(layer))
            x = sin_lin(x, self.sin_w0)
        return x


class WinFun(nn.Module):
    def __init__(self,
                 dim_in_pose=128,
                 dim_hidden=128,
                 dim_in_point=66,
                 dim_out=24,
                 sin_w0=2.0
                 ):
        super().__init__()
        self.sin_w0 = sin_w0
        self.layer_pose = LinearLayer(dim_in_pose, dim_hidden)
        self.layer_point = LinearLayer(dim_in_point, dim_hidden)

        sin_dims = [dim_hidden, dim_hidden, dim_hidden]
        self.num_sin_layers = len(sin_dims)

        for layer in range(0, len(sin_dims) - 1):
            setattr(self, "sin_lin" + str(layer), SirenLayer(sin_dims[layer], sin_dims[layer + 1]))

        self.last_layer = LinearLayer(dim_hidden, dim_out)
        self.dims = dim_hidden
        self.init_siren()

    def init_siren(self):
        sine_init(self.layer_pose, w0=self.sin_w0)
        sine_init(self.layer_point, w0=self.sin_w0)
        sine_init(self.last_layer, w0=self.sin_w0)
        for layer in range(0, self.num_sin_layers - 1):
            lin = getattr(self, "sin_lin" + str(layer))
            sine_init(lin, w0=self.sin_w0)

    def forward(self, feat_x, feat_pose, p1):
        feat_pose = feat_pose.flatten(end_dim=1)
        feat_pose = self.layer_pose(feat_pose)
        feat_pose = feat_pose.reshape(-1, 1, self.dims * 24)
        feat_pose = feat_pose.expand(-1, feat_x.shape[0] // feat_pose.shape[0], -1)
        feat_pose = feat_pose.reshape(-1, 24, self.dims)

        feat_x = feat_x.flatten(end_dim=1)
        feat_x = self.layer_point(feat_x)
        feat_x = feat_x.reshape(-1, 24, self.dims)

        f_xp = feat_pose + feat_x
        f_xp = f_xp * p1[..., None]

        p2 = torch.max(f_xp, dim=1)[0]  # [N, 128]

        for layer in range(0, self.num_sin_layers - 1):
            sin_lin = getattr(self, "sin_lin" + str(layer))
            p2 = sin_lin(p2, self.sin_w0)

        p2 = self.last_layer(p2)  # [N, 24]
        p2 = torch.sigmoid(p2)

        p = p1 * p2

        return p, feat_pose * p[..., None]
