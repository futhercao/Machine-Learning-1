"""PointMLP (Ma et al., ICLR 2022) for point cloud classification.

Architecture summary:
  - Input embedding: 3 -> embed_dim (1x1 conv), xyz only
  - 4 stages, each: LocalGrouper (FPS+KNN + geometric affine) ->
    pre-residual MLP blocks -> max-pool -> post-residual MLP blocks
  - Channels double each stage; points halve each stage
  - Global max-pool + classifier FC head

Input:  (B, N, 3)  xyz only
Output: (B, num_classes)

Reference: Ma et al., "Rethinking Network Design and Local Geometry in Point Cloud:
A Simple Residual MLP Framework", ICLR 2022.  Official OA 94.1%, mAcc 91.5% on ModelNet40.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sampling / grouping utils
# ---------------------------------------------------------------------------
def square_distance(src, dst):
    return torch.cdist(src, dst) ** 2


def index_points(points, idx):
    B = points.shape[0]
    view = list(idx.shape)
    view[0] = B
    batch_indices = torch.arange(B, device=points.device).view(B, *([1] * (len(view) - 1))).expand(*view)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """FPS. xyz: (B, N, 3) -> (B, npoint)"""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        c = xyz[batch_idx, farthest, :].unsqueeze(1)
        dist = ((xyz - c) ** 2).sum(-1)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(-1)
    return centroids


def knn(xyz, new_xyz, k):
    """For each query in new_xyz, return idx of k nearest neighbors in xyz.
    xyz: (B, N, 3), new_xyz: (B, S, 3) -> (B, S, k)"""
    dist = square_distance(new_xyz, xyz)
    return dist.topk(k, dim=-1, largest=False)[1]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ConvBNReLU1D(nn.Module):
    def __init__(self, c_in, c_out, k=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, bias=bias),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    """Residual block: Conv-BN-ReLU -> Conv-BN, add input, ReLU."""
    def __init__(self, channels, res_expansion=1.0, bias=True):
        super().__init__()
        hidden = int(channels * res_expansion)
        self.net1 = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=bias),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(hidden, channels, 1, bias=bias),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class LocalGrouper(nn.Module):
    """FPS + KNN grouping with Geometric Affine normalization."""
    def __init__(self, channel, groups, k, use_xyz=True, normalize='anchor'):
        super().__init__()
        self.groups = groups
        self.k = k
        self.use_xyz = use_xyz
        self.normalize = normalize
        add_ch = 3 if use_xyz else 0
        self.affine_alpha = nn.Parameter(torch.ones(1, 1, 1, channel + add_ch))
        self.affine_beta = nn.Parameter(torch.zeros(1, 1, 1, channel + add_ch))

    def forward(self, xyz, points):
        # xyz: (B, N, 3), points: (B, N, C)
        B, N, C = points.shape
        S = self.groups
        fps_idx = farthest_point_sample(xyz, S)
        new_xyz = index_points(xyz, fps_idx)
        new_pts = index_points(points, fps_idx)

        idx = knn(xyz, new_xyz, self.k)
        grouped_xyz = index_points(xyz, idx)         # (B, S, k, 3)
        grouped_pts = index_points(points, idx)       # (B, S, k, C)
        if self.use_xyz:
            grouped_pts = torch.cat([grouped_pts, grouped_xyz], dim=-1)

        anchor = torch.cat([new_pts, new_xyz], dim=-1) if self.use_xyz else new_pts
        anchor = anchor.unsqueeze(2)

        if self.normalize == 'center':
            mean = grouped_pts.mean(dim=2, keepdim=True)
            std = (grouped_pts - mean).pow(2).mean(dim=[1, 2, 3], keepdim=True).sqrt()
            grouped_pts = (grouped_pts - mean) / (std + 1e-5)
        elif self.normalize == 'anchor':
            std = (grouped_pts - anchor).pow(2).mean(dim=[1, 2, 3], keepdim=True).sqrt()
            grouped_pts = (grouped_pts - anchor) / (std + 1e-5)

        grouped_pts = self.affine_alpha * grouped_pts + self.affine_beta
        grouped_pts = torch.cat([grouped_pts, anchor.expand(-1, -1, self.k, -1)], dim=-1)
        return new_xyz, grouped_pts


class PreExtraction(nn.Module):
    """ConvBNReLU input -> n residual blocks on (B, S, k, C_in)."""
    def __init__(self, channels, out_channels, blocks=1, res_expansion=1.0, k=24, use_xyz=True):
        super().__init__()
        in_ch = (channels + 3) * 2 if use_xyz else channels * 2
        self.transfer = ConvBNReLU1D(in_ch, out_channels, k=1, bias=True)
        ops = []
        for _ in range(blocks):
            ops.append(ConvBNReLURes1D(out_channels, res_expansion=res_expansion))
        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        B, S, K, D = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B * S, D, K)
        x = self.transfer(x)
        x = self.ops(x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = x.reshape(B, S, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, res_expansion=1.0):
        super().__init__()
        self.ops = nn.Sequential(*[
            ConvBNReLURes1D(channels, res_expansion=res_expansion) for _ in range(blocks)
        ])

    def forward(self, x):
        return self.ops(x)


# ---------------------------------------------------------------------------
# Full PointMLP model
# ---------------------------------------------------------------------------
class PointMLP(nn.Module):
    def __init__(self, num_classes=40, points=1024, embed_dim=64,
                 groups=1, res_expansion=1.0, activation='relu', bias=True,
                 use_xyz=True, normalize='anchor',
                 dim_expansion=(2, 2, 2, 2),
                 pre_blocks=(2, 2, 2, 2),
                 pos_blocks=(2, 2, 2, 2),
                 k_neighbors=(24, 24, 24, 24),
                 reducers=(2, 2, 2, 2)):
        super().__init__()
        self.stages = len(pre_blocks)
        self.embedding = ConvBNReLU1D(3, embed_dim, k=1, bias=bias)

        last_ch = embed_dim
        anchor_pts = points
        self.local_groupers = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        for i in range(self.stages):
            out_ch = last_ch * dim_expansion[i]
            anchor_pts = anchor_pts // reducers[i]
            self.local_groupers.append(
                LocalGrouper(last_ch, anchor_pts, k_neighbors[i],
                             use_xyz=use_xyz, normalize=normalize))
            self.pre_blocks_list.append(
                PreExtraction(last_ch, out_ch, blocks=pre_blocks[i],
                              res_expansion=res_expansion,
                              k=k_neighbors[i], use_xyz=use_xyz))
            self.pos_blocks_list.append(
                PosExtraction(out_ch, blocks=pos_blocks[i], res_expansion=res_expansion))
            last_ch = out_ch

        self.classifier = nn.Sequential(
            nn.Linear(last_ch, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, N, 3) — xyz only
        xyz = x[:, :, :3].contiguous()
        f = self.embedding(xyz.permute(0, 2, 1))         # (B, embed_dim, N)
        for i in range(self.stages):
            pts = f.permute(0, 2, 1)                     # (B, N, C)
            xyz, grouped = self.local_groupers[i](xyz, pts)
            f = self.pre_blocks_list[i](grouped)
            f = self.pos_blocks_list[i](f)
        f = F.adaptive_max_pool1d(f, 1).squeeze(-1)
        return self.classifier(f)


def build_model(name='pointmlp', num_classes=40, **kwargs):
    """Entry point — currently only PointMLP is supported."""
    if name not in ('pointmlp',):
        raise ValueError(f"Unknown model: {name}. Supported: pointmlp")
    return PointMLP(num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    m = build_model().cuda()
    x = torch.randn(2, 1024, 3).cuda()
    y = m(x)
    print(y.shape)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"PointMLP params: {n_params/1e6:.2f}M")
