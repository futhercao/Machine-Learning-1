"""
Point cloud classification models.

Implements:
  - PointNet++ SSG (Single-Scale Grouping)
  - PointNet++ MSG (Multi-Scale Grouping)
  - DGCNN

All accept input (B, N, C) where C = 3 (xyz) or 6 (xyz + normal).
Forward returns logits (B, num_classes).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility functions (PointNet++ style)
# ============================================================
def square_distance(src, dst):
    """src: (B, N, C), dst: (B, M, C) -> (B, N, M) pairwise sq dist."""
    return torch.cdist(src, dst, p=2) ** 2


def index_points(points, idx):
    """Gather points according to idx.
    points: (B, N, C), idx: (B, ...) -> (B, ..., C)
    """
    device = points.device
    B = points.shape[0]
    view = list(idx.shape)
    view[0] = B
    repeat = [1] * len(view); repeat[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, *([1]*(len(view)-1)))
    batch_indices = batch_indices.expand(*view)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """FPS sampling. xyz: (B, N, 3) -> idx (B, npoint)"""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B,1,3)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Ball query. xyz: (B,N,3), new_xyz: (B,S,3) -> group_idx (B,S,nsample)"""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    sqrdists = square_distance(new_xyz, xyz)  # (B,S,N)
    group_idx = torch.arange(N, dtype=torch.long, device=xyz.device).view(1, 1, N).expand(B, S, N).clone()
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # replace invalid (= N) with first valid (group_first)
    group_first = group_idx[:, :, 0:1].expand_as(group_idx)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """Returns new_xyz (B,S,3) and new_points (B,S,nsample,C+3)."""
    B, N, _ = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)  # (B, npoint)
    new_xyz = index_points(xyz, fps_idx)          # (B, npoint, 3)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, idx)          # (B, npoint, nsample, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """For the last layer: group all points together."""
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
    grouped_xyz = xyz.view(B, 1, N, 3)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# ============================================================
# PointNet Set Abstraction
# ============================================================
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        layers = []
        last = in_channel
        for c in mlp:
            layers += [nn.Conv2d(last, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            last = c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        # xyz: (B, N, 3), points: (B, N, C) or None
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_points: (B, S, nsample, C+3) -> (B, C+3, nsample, S) for Conv2d
        new_points = new_points.permute(0, 3, 2, 1)
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, 2)[0]  # (B, mlp[-1], S)
        new_points = new_points.permute(0, 2, 1)  # (B, S, mlp[-1])
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """MSG: multi-scale grouping at different radii."""
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlps = nn.ModuleList()
        for mlp in mlp_list:
            layers = []
            last = in_channel + 3
            for c in mlp:
                layers += [nn.Conv2d(last, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                last = c
            self.mlps.append(nn.Sequential(*layers))

    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        out_list = []
        for r, ns, mlp in zip(self.radius_list, self.nsample_list, self.mlps):
            idx = query_ball_point(r, ns, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_pts = index_points(points, idx)
                grouped = torch.cat([grouped_xyz, grouped_pts], dim=-1)
            else:
                grouped = grouped_xyz
            grouped = grouped.permute(0, 3, 2, 1)  # (B, C, ns, npoint)
            grouped = mlp(grouped)
            grouped = torch.max(grouped, 2)[0]     # (B, C', npoint)
            out_list.append(grouped)
        new_points = torch.cat(out_list, dim=1).permute(0, 2, 1)  # (B, npoint, sum C')
        return new_xyz, new_points


# ============================================================
# PointNet++ SSG / MSG
# ============================================================
class PointNet2SSG(nn.Module):
    def __init__(self, num_classes=40, use_normals=True):
        super().__init__()
        self.use_normals = use_normals
        in_c = 6 if use_normals else 3
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_c, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512); self.bn1 = nn.BatchNorm1d(512); self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256);  self.bn2 = nn.BatchNorm1d(256); self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, N, 3 or 6)
        xyz = x[:, :, :3]
        feats = x[:, :, 3:] if self.use_normals else None
        l1_xyz, l1_pts = self.sa1(xyz, feats)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)
        x = l3_pts.squeeze(1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


class PointNet2MSG(nn.Module):
    def __init__(self, num_classes=40, use_normals=True):
        super().__init__()
        self.use_normals = use_normals
        in_c = 3 if use_normals else 0
        self.sa1 = PointNetSetAbstractionMsg(
            512, [0.1, 0.2, 0.4], [16, 32, 128], in_c,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        out1 = 64 + 128 + 128  # = 320
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.2, 0.4, 0.8], [32, 64, 128], out1,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        out2 = 128 + 256 + 256  # = 640
        self.sa3 = PointNetSetAbstraction(None, None, None, out2 + 3, [256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512); self.bn1 = nn.BatchNorm1d(512); self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256);  self.bn2 = nn.BatchNorm1d(256); self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        xyz = x[:, :, :3]
        feats = x[:, :, 3:] if self.use_normals else None
        l1_xyz, l1_pts = self.sa1(xyz, feats)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)
        x = l3_pts.squeeze(1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


# ============================================================
# DGCNN
# ============================================================
def knn(x, k):
    """x: (B, C, N) -> (B, N, k) indices"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = (x ** 2).sum(dim=1, keepdim=True)
    pairwise = -xx - inner - xx.transpose(2, 1)
    return pairwise.topk(k=k, dim=-1)[1]


def get_graph_feature(x, k=20):
    """x: (B, C, N) -> (B, 2C, N, k) edge features."""
    B, C, N = x.shape
    idx = knn(x, k=k)
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)
    x_t = x.transpose(2, 1).contiguous()  # (B, N, C)
    feat = x_t.view(B * N, -1)[idx, :].view(B, N, k, C)
    x_t = x_t.view(B, N, 1, C).expand_as(feat)
    edge = torch.cat([feat - x_t, x_t], dim=-1).permute(0, 3, 1, 2).contiguous()
    return edge


class DGCNN(nn.Module):
    def __init__(self, num_classes=40, k=20, emb_dims=1024, dropout=0.5, in_channels=6):
        super().__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64); self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128); self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        self.conv1 = nn.Conv2d(in_channels * 2, 64, 1, bias=False)
        self.conv2 = nn.Conv2d(64 * 2, 64, 1, bias=False)
        self.conv3 = nn.Conv2d(64 * 2, 128, 1, bias=False)
        self.conv4 = nn.Conv2d(128 * 2, 256, 1, bias=False)
        self.conv5 = nn.Conv1d(64 + 64 + 128 + 256, emb_dims, 1, bias=False)
        self.linear1 = nn.Linear(emb_dims * 2, 512)
        self.bn6 = nn.BatchNorm1d(512); self.dp1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256); self.dp2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, N, C) -> (B, C, N)
        x = x.permute(0, 2, 1)
        B, _, N = x.shape
        x = get_graph_feature(x, k=self.k)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x1 = x.max(dim=-1)[0]
        x = get_graph_feature(x1, k=self.k)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x2 = x.max(dim=-1)[0]
        x = get_graph_feature(x2, k=self.k)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x3 = x.max(dim=-1)[0]
        x = get_graph_feature(x3, k=self.k)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x4 = x.max(dim=-1)[0]
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = torch.cat([x_max, x_avg], 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), 0.2); x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), 0.2); x = self.dp2(x)
        return self.linear3(x)


def build_model(name, num_classes=40, use_normals=True, **kw):
    name = name.lower()
    if name == 'pointnet2_ssg':
        return PointNet2SSG(num_classes, use_normals)
    if name == 'pointnet2_msg':
        return PointNet2MSG(num_classes, use_normals)
    if name == 'dgcnn':
        return DGCNN(num_classes, in_channels=6 if use_normals else 3, **kw)
    raise ValueError(name)


if __name__ == '__main__':
    x = torch.randn(2, 1024, 6).cuda()
    for n in ['pointnet2_ssg', 'pointnet2_msg', 'dgcnn']:
        m = build_model(n).cuda()
        y = m(x)
        print(n, y.shape, sum(p.numel() for p in m.parameters()))
