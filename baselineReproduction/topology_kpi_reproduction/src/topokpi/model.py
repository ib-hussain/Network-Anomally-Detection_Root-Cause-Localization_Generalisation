from __future__ import annotations
import math

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for the proposed neural model; install requirements.txt") from exc


def normalize_adjacency(a: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    a = torch.clamp(a, min=0) + eye
    degree = a.sum(-1).clamp_min(1e-12)
    inv = degree.rsqrt()
    return inv[:, None] * a * inv[None, :]


class SinusoidalPosition(nn.Module):
    def __init__(self, hidden: int, max_length: int = 4096):
        super().__init__()
        position = torch.arange(max_length).float().unsqueeze(1)
        scale = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.0) / hidden))
        pe = torch.zeros(max_length, hidden)
        pe[:, 0::2] = torch.sin(position * scale)
        pe[:, 1::2] = torch.cos(position * scale[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.shape[1]]


class GraphLayer(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # x [B,T,N,H], graph aggregation over N.
        neighbor = torch.einsum("ij,btjh->btih", a, x)
        return self.norm(x + self.dropout(torch.nn.functional.gelu(self.linear(neighbor))))


class TopologyAwareSSL(nn.Module):
    """Topology-aware self-supervised model described in manuscript Eqs. 14–24."""
    def __init__(self, kpis: int = 8, hidden: int = 64, temporal_layers: int = 2,
                 graph_layers: int = 2, heads: int = 4, feedforward: int = 256,
                 dropout: float = .1, topology_input_strength: float = .2):
        super().__init__()
        self.kpis = kpis
        self.beta = topology_input_strength
        self.input_projection = nn.Linear(kpis, hidden)
        self.position = SinusoidalPosition(hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, feedforward, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(layer, temporal_layers)
        self.graph = nn.ModuleList([GraphLayer(hidden, dropout) for _ in range(graph_layers)])
        self.reconstruction = nn.Linear(hidden, kpis)
        self.prediction = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, kpis))
        # Eq. (kpi_weight): w_k = softmax(rho_k). rho must be a trainable parameter,
        # not a fixed buffer, otherwise the KPI weighting collapses to a permanent
        # uniform average and the manuscript's learned weighting is never realized.
        self.kpi_logits = nn.Parameter(torch.zeros(kpis))

    def forward(self, x: torch.Tensor, a_enc: torch.Tensor):
        a = normalize_adjacency(a_enc)
        topology_input = x + self.beta * torch.einsum("ij,btjd->btid", a, x)
        h = self.input_projection(topology_input)
        b, t, n, hidden = h.shape
        temporal = h.permute(0, 2, 1, 3).reshape(b * n, t, hidden)
        temporal = self.temporal(self.position(temporal))
        h = temporal.reshape(b, n, t, hidden).permute(0, 2, 1, 3)
        for layer in self.graph:
            h = layer(h, a)
        return self.reconstruction(h), self.prediction(h[:, -1]), h

    def scores(self, x: torch.Tensor, y_next: torch.Tensor, a_enc: torch.Tensor, alpha: float = .5):
        reconstruction, prediction, latent = self(x, a_enc)
        kpi = alpha * (x[:, -1] - reconstruction[:, -1]).abs() + (1 - alpha) * (y_next - prediction).abs()
        weights = torch.softmax(self.kpi_logits, dim=0)
        node = (kpi * weights).sum(-1)
        return kpi, node, reconstruction, prediction, latent


def ssl_loss(model: TopologyAwareSSL, original: torch.Tensor, masked: torch.Tensor,
             mask: torch.Tensor, y_next: torch.Tensor, a_enc: torch.Tensor,
             lambda_rec: float = 1., lambda_pred: float = 1., lambda_topology: float = .1):
    reconstruction, prediction, latent = model(masked, a_enc)
    rec = ((reconstruction - original).abs() * mask).sum() / mask.sum().clamp_min(1)
    pred = (prediction - y_next).abs().mean()
    a = normalize_adjacency(a_enc)
    last = latent[:, -1]
    squared = (last[:, :, None, :] - last[:, None, :, :]).pow(2).sum(-1)
    topo = (squared * a[None]).sum() / (a.sum() * original.shape[0] + 1e-12)
    total = lambda_rec * rec + lambda_pred * pred + lambda_topology * topo
    return total, {"loss": total.detach(), "reconstruction": rec.detach(), "prediction": pred.detach(), "topology": topo.detach()}
