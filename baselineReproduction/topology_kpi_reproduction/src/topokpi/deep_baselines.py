"""Executable compatibility implementations of the deep baselines.

These classes preserve each paper's central scoring idea and the hyperparameters
reported in this manuscript. They are not copies of, or substitutes for citing,
the official repositories. Strict numerical comparison should use official-code
adapters with pinned commits.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


def _flatten_nodes(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2) if x.ndim == 4 else x


class Baseline(nn.Module):
    def loss(self, x: torch.Tensor, y_next: torch.Tensor | None = None, epoch: int = 1) -> torch.Tensor:
        raise NotImplementedError

    def score(self, x: torch.Tensor, y_next: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class OmniAnomalyCompat(Baseline):
    def __init__(self, features: int, hidden: int = 64, latent: int = 16, mc_samples: int = 10):
        super().__init__(); self.mc_samples = mc_samples
        self.gru = nn.GRU(features, hidden, batch_first=True)
        self.mu, self.logvar = nn.Linear(hidden, latent), nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, features))

    def _terms(self, x):
        x = _flatten_nodes(x); h, _ = self.gru(x); mu, logvar = self.mu(h), self.logvar(h).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar)
        rec = self.decoder(z); kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
        return rec, kl

    def loss(self, x, y_next=None, epoch=1):
        x = _flatten_nodes(x); rec, kl = self._terms(x); return (rec - x).square().mean() + .01 * kl

    def score(self, x, y_next=None):
        x = _flatten_nodes(x); values = []
        for _ in range(self.mc_samples): values.append((self._terms(x)[0] - x).square().mean((1, 2)))
        return torch.stack(values).mean(0)


class MSCREDCompat(Baseline):
    def __init__(self, features: int, windows=(5, 10, 20), channels=(32, 64, 128)):
        super().__init__(); self.windows = tuple(windows)
        # Channel widths are compressed to keep F x F signatures tractable while retaining a 3-block encoder.
        c1, c2, c3 = channels
        self.encoder = nn.Sequential(nn.Conv2d(len(windows), c1, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(c3, c2, 3, padding=1), nn.ReLU(), nn.Conv2d(c2, c1, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(c1, len(windows), 3, padding=1))

    def signatures(self, x):
        x = _flatten_nodes(x); outputs = []
        for width in self.windows:
            z = x[:, -min(width, x.shape[1]):]
            outputs.append(torch.einsum("btf,btg->bfg", z, z) / z.shape[1])
        return torch.stack(outputs, 1)

    def reconstruct(self, x):
        sig = self.signatures(x); return sig, self.decoder(self.encoder(sig))

    def loss(self, x, y_next=None, epoch=1):
        sig, rec = self.reconstruct(x); return (rec - sig).square().mean()

    def score(self, x, y_next=None):
        sig, rec = self.reconstruct(x); return (rec - sig).square().mean((1, 2, 3))


class USADCompat(Baseline):
    def __init__(self, features: int, window: int = 20, hidden=(64, 32), latent: int = 16):
        super().__init__(); size = features * window; h1, h2 = hidden
        self.encoder = nn.Sequential(nn.Linear(size, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, latent))
        def decoder(): return nn.Sequential(nn.Linear(latent, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, size))
        self.d1, self.d2 = decoder(), decoder()

    def outputs(self, x):
        flat = _flatten_nodes(x).flatten(1); z = self.encoder(flat); r1, r2 = self.d1(z), self.d2(z)
        r12 = self.d2(self.encoder(r1)); return flat, r1, r2, r12

    def loss(self, x, y_next=None, epoch=1):
        flat, r1, r2, r12 = self.outputs(x); n = max(epoch, 1)
        l1 = (1/n) * F.mse_loss(r1, flat) + (1 - 1/n) * F.mse_loss(r12, flat)
        l2 = (1/n) * F.mse_loss(r2, flat) - (1 - 1/n) * F.mse_loss(r12, flat)
        return l1 + l2

    def score(self, x, y_next=None):
        flat, r1, r2, _ = self.outputs(x); return .5 * (r1-flat).square().mean(1) + .5 * (r2-flat).square().mean(1)


class THOCCompat(Baseline):
    def __init__(self, features: int, hidden: int = 64, clusters: int = 6):
        super().__init__(); self.gru = nn.GRU(features, hidden, batch_first=True); self.centers = nn.Parameter(torch.randn(clusters, hidden) / math.sqrt(hidden))

    def distances(self, x):
        h, _ = self.gru(_flatten_nodes(x)); resolutions = torch.stack((h[:, -1], h[:, h.shape[1]//2], h.mean(1)), 1)
        return torch.cdist(resolutions, self.centers).min(-1).values

    def loss(self, x, y_next=None, epoch=1): return self.distances(x).mean()
    def score(self, x, y_next=None): return self.distances(x).mean(1)


class MTADGATCompat(Baseline):
    def __init__(self, features: int, hidden: int = 64, heads: int = 4):
        super().__init__(); self.feature_gate = nn.Sequential(nn.Linear(features, features), nn.Sigmoid())
        self.temporal = nn.MultiheadAttention(features, heads if features % heads == 0 else 1, batch_first=True)
        self.gru = nn.GRU(features, hidden, batch_first=True); self.forecast = nn.Linear(hidden, features); self.reconstruct = nn.Linear(hidden, features)

    def outputs(self, x):
        x = _flatten_nodes(x); gated = x * self.feature_gate(x.mean(1))[:, None]; attended, _ = self.temporal(gated, gated, gated)
        h, _ = self.gru(attended); return self.forecast(h[:, -1]), self.reconstruct(h), x

    def loss(self, x, y_next=None, epoch=1):
        forecast, rec, target = self.outputs(x); y = _flatten_nodes(y_next[:, None])[:, 0] if y_next is not None else target[:, -1]
        return .5 * F.mse_loss(forecast, y) + .5 * F.mse_loss(rec, target)

    def score(self, x, y_next=None):
        forecast, rec, target = self.outputs(x); y = _flatten_nodes(y_next[:, None])[:, 0] if y_next is not None else target[:, -1]
        return .5 * (forecast-y).square().mean(1) + .5 * (rec-target).square().mean((1, 2))


class GDNCompat(Baseline):
    def __init__(self, features: int, embedding: int = 64, top_k: int | None = None):
        super().__init__(); self.features = features; self.top_k = min(top_k or 10, features-1)
        self.embedding = nn.Parameter(torch.randn(features, embedding)); self.value = nn.Linear(1, embedding); self.out = nn.Linear(embedding, 1)

    def predict(self, x):
        last = _flatten_nodes(x)[:, -1]; similarity = F.normalize(self.embedding) @ F.normalize(self.embedding).T
        similarity.fill_diagonal_(-torch.inf); index = similarity.topk(self.top_k, dim=-1).indices
        mask = torch.zeros_like(similarity).scatter(1, index, 1); attention = torch.softmax(similarity.masked_fill(mask == 0, -torch.inf), -1)
        h = self.value(last.unsqueeze(-1)) + torch.einsum("fg,bge->bfe", attention, self.value(last.unsqueeze(-1)))
        return self.out(F.gelu(h)).squeeze(-1)

    def loss(self, x, y_next=None, epoch=1):
        target = _flatten_nodes(y_next[:, None])[:, 0] if y_next is not None else _flatten_nodes(x)[:, -1]; return F.mse_loss(self.predict(x), target)
    def score(self, x, y_next=None):
        target = _flatten_nodes(y_next[:, None])[:, 0] if y_next is not None else _flatten_nodes(x)[:, -1]; return (self.predict(x)-target).abs().max(1).values


class InterFusionCompat(OmniAnomalyCompat):
    """Hierarchical metric/temporal latent compatibility model."""
    def __init__(self, features: int, hidden: int = 64, latent: int = 16, mc_samples: int = 10):
        super().__init__(features, hidden, latent * 2, mc_samples)


class AnomalyTransformerCompat(Baseline):
    def __init__(self, features: int, d_model=64, heads=4, layers=3, d_ff=256, dropout=.1):
        super().__init__(); self.input = nn.Linear(features, d_model)
        self.attn = nn.ModuleList([nn.MultiheadAttention(d_model, heads, dropout, batch_first=True) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)) for _ in range(layers)])
        self.out = nn.Linear(d_model, features)

    def outputs(self, x):
        x = _flatten_nodes(x); h = self.input(x); discrepancies = []
        for attn, ff in zip(self.attn, self.ff):
            z, weights = attn(h, h, h, need_weights=True, average_attn_weights=False); h = h + z; h = h + ff(h)
            t = x.shape[1]; positions = torch.arange(t, device=x.device); prior = torch.softmax(-torch.abs(positions[:, None]-positions[None, :]).float(), -1)
            series = weights.mean(1); discrepancies.append(F.kl_div((series+1e-8).log(), prior[None].expand_as(series), reduction="none").sum(-1).mean(-1))
        return x, self.out(h), torch.stack(discrepancies).mean(0)

    def loss(self, x, y_next=None, epoch=1):
        target, rec, discrepancy = self.outputs(x); return (rec-target).square().mean() - .01 * discrepancy.mean()
    def score(self, x, y_next=None):
        target, rec, discrepancy = self.outputs(x); return torch.softmax(-discrepancy, 0) * (rec-target).square().mean((1, 2))


class TranADCompat(Baseline):
    def __init__(self, features: int, d_model=64, heads=4, layers=2, d_ff=256, dropout=.1):
        super().__init__(); self.input = nn.Linear(features * 2, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers); self.out1 = nn.Linear(d_model, features); self.out2 = nn.Linear(d_model, features)

    def outputs(self, x):
        x = _flatten_nodes(x); zeros = torch.zeros_like(x); h1 = self.encoder(self.input(torch.cat((x, zeros), -1))); r1 = self.out1(h1)
        condition = (r1-x).square(); h2 = self.encoder(self.input(torch.cat((x, condition), -1))); return x, r1, self.out2(h2)

    def loss(self, x, y_next=None, epoch=1):
        target, r1, r2 = self.outputs(x); n=max(epoch,1); return (1/n)*F.mse_loss(r1,target)+(1-1/n)*F.mse_loss(r2,target)
    def score(self, x, y_next=None):
        target, r1, r2 = self.outputs(x); return .5*(r1-target).square().mean((1,2))+.5*(r2-target).square().mean((1,2))


class DCDetectorCompat(Baseline):
    def __init__(self, features: int, window: int = 20, d_model=64, heads=4, patches=(1, 5), dropout=.1):
        super().__init__(); self.patches=patches; self.proj=nn.Linear(features,d_model)
        self.time_attn=nn.MultiheadAttention(d_model,heads,dropout,batch_first=True); self.patch_attn=nn.MultiheadAttention(d_model,heads,dropout,batch_first=True)

    def discrepancy(self,x):
        h=self.proj(_flatten_nodes(x)); _, time=self.time_attn(h,h,h,need_weights=True,average_attn_weights=False); time=time.mean(1)
        width=max(self.patches); usable=(h.shape[1]//width)*width; patches=h[:,:usable].reshape(h.shape[0],-1,width,h.shape[-1]).mean(2)
        _, patch=self.patch_attn(patches,patches,patches,need_weights=True,average_attn_weights=False); patch=patch.mean(1)
        time_entropy=-(time.clamp_min(1e-8)*time.clamp_min(1e-8).log()).sum(-1).mean(-1)
        patch_entropy=-(patch.clamp_min(1e-8)*patch.clamp_min(1e-8).log()).sum(-1).mean(-1)
        return (time_entropy-patch_entropy).abs()
    def loss(self,x,y_next=None,epoch=1): return -self.discrepancy(x).mean()
    def score(self,x,y_next=None): return self.discrepancy(x)


def build_deep_baseline(name: str, features: int, window: int = 20) -> Baseline:
    factories = {
        "OmniAnomaly": lambda: OmniAnomalyCompat(features), "MSCRED": lambda: MSCREDCompat(features),
        "USAD": lambda: USADCompat(features, window), "THOC": lambda: THOCCompat(features),
        "MTAD-GAT": lambda: MTADGATCompat(features), "GDN": lambda: GDNCompat(features),
        "InterFusion": lambda: InterFusionCompat(features), "Anomaly Transformer": lambda: AnomalyTransformerCompat(features),
        "TranAD": lambda: TranADCompat(features), "DCdetector": lambda: DCDetectorCompat(features, window),
    }
    if name not in factories: raise ValueError(f"Unknown deep baseline {name!r}")
    return factories[name]()
