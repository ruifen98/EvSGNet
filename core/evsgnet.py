import torch
import torch.nn as nn
from loss import batch_episym
import torch.nn.functional as F

class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            trans(1, 2))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out
class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1)
        )
    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out
class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out
class OABlock(nn.Module):
    def __init__(self, net_channels, depth=6, clusters=250):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        l2_nums = clusters
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = []
        for _ in range(self.layer_num // 2):
            self.l2.append(OAFilter(channels, l2_nums))
        self.up1 = diff_unpool(channels, l2_nums)
        self.l2 = nn.Sequential(*self.l2)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)
        self.shot_cut = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.input = LGDAFN(channels)
        self.output = LGDAFN(channels)

    def forward(self, data):
        x1_1 = data
        x1_1 = self.input(x1_1)
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = torch.cat([x1_1, x_up], dim=1)
        out = self.shot_cut(out)
        out = self.output(out)
        return out
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx[:, :, :]
def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_out + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature
class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)

class LGDAFN(nn.Module):
    def __init__(self, in_channels=128, reduction=4, use_residual=True):
        super(LGDAFN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        inter_channels = int(self.in_channels // reduction)
        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU()
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(self.out_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
        )
        self.global_att_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
        )
        self.global_local_diff = nn.Sequential(
            nn.Conv2d(self.out_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_out = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.use_residual = use_residual

    def forward(self, x):
        batch_size, C, N, _ = x.shape
        input_conv = self.conv_in(x)
        local_scale = self.local_att(input_conv)
        global_avg_scale = self.global_att_avg(input_conv)
        global_mean = torch.mean(input_conv, dim=2, keepdim=True)
        diff_feat = input_conv - global_mean
        diff_scale = self.global_local_diff(diff_feat)
        scale_out = local_scale + global_avg_scale + diff_scale
        scale_out = self.sigmoid(scale_out)
        weighted_feat = input_conv * scale_out
        output = self.conv_out(weighted_feat)
        output = output + input_conv
        if self.use_residual:
            output = output + x
        return output

def batch_symeig(X):
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='U')
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv

def weighted_8points(x_in, logits, temperature=0.5):

    mask_logit = logits[:, 0, :, 0]
    weight_logit = logits[:, 1, :, 0]

    conf = torch.sigmoid(mask_logit)
    weight_norm = torch.softmax(weight_logit / temperature, dim=-1)  # ÎÂ¶ÈËõ·Å£¬ÎÈ¶¨ÌÝ¶È
    final_weight = conf * weight_norm  # ÖÃÐÅ¶È¹ýÂË + ¹éÒ»»¯È¨ÖØ


    final_weight = final_weight / (torch.sum(final_weight, dim=-1, keepdim=True) + 1e-6)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()

    wX = final_weight.unsqueeze(-1) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))
    e_hat = e_hat / (torch.norm(e_hat, dim=1, keepdim=True) + 1e-6)

    return e_hat
class DGCNN_MAX_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DGCNN_MAX_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel
        self.input = LGDAFN(self.in_channel)
        self.output = LGDAFN(self.in_channel)
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        B, _, N, _ = features.shape
        features = self.input(features)
        out = get_graph_feature(features, k=self.knn_num)
        out = self.conv(out)
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.unsqueeze(3)
        out = self.output(out)
        return out


REGION_CONNECT_MAP = [
    [5, 7, 8], [6, 7, 8], [3, 6, 7],
    [2, 5, 8], [0, 2, 8], [0, 3, 6],
    [1, 2, 5], [0, 1, 2], [0, 1, 3],
]

class EDLEvidenceHead(nn.Module):
    def __init__(self, in_channel, num_classes=2, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.evidence_net = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, 1),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(),
            nn.Conv2d(in_channel // 2, num_classes, 1),
            nn.ReLU()
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        evidence = self.evidence_net(x)
        alpha = evidence + 1.0
        total_S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = self.num_classes / (total_S + self.eps)
        return uncertainty, alpha

class UDEG(nn.Module):
    def __init__(
            self,
            in_channel,
            knn_num=9,
            use_dynamic_graph=True,
            use_selective_graph=True,
            use_residual=True,
            beta=0.8
    ):
        super().__init__()
        self.in_channel = in_channel
        self.knn_num = knn_num
        self.use_residual = use_residual
        self.use_dynamic_graph = use_dynamic_graph
        self.use_selective_graph = use_selective_graph
        self.beta = beta
        self.u_threshold = nn.Parameter(torch.tensor(self.beta))
        self.long_range_num = 3
        self.uncertainty_fuse = nn.Sequential(
            nn.Conv2d(in_channel * 2 + 2, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self.edl_head = EDLEvidenceHead(in_channel)
        self.gcn_weight = nn.Parameter(torch.randn(in_channel, in_channel))
        nn.init.xavier_uniform_(self.gcn_weight)

        self.conv = nn.Sequential(
            ResNet_Block(in_channel, in_channel),
            LGDAFN(in_channel),
            nn.Conv2d(in_channel, in_channel, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

    def get_knn_idx(self, x):
        B, C, N = x.shape
        idx = knn(x, k=self.knn_num)
        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
        return (idx + idx_base).view(-1)

    def get_knn(self, x, idx=None):
        B, C, N = x.shape
        if idx is None:
            idx = self.get_knn_idx(x)
        x = x.transpose(2, 1).contiguous()
        neighbor = x.view(B * N, C)[idx]
        return neighbor.view(B, N, -1, C)

    def compute_uncertainty(self, x, coords=None, idx=None):
        B, C, N = x.shape
        if idx is None:
            idx = self.get_knn_idx(x)

        neighbor = self.get_knn(x, idx)  # [B, N, K, C]
        center_feat = x.transpose(2, 1).unsqueeze(2)  # [B, N, 1, C]
        feat_diff = neighbor - center_feat  # [B, N, K, C]
        dirichlet_channel = torch.sum(feat_diff ** 2, dim=2)  # [B, N, C]
        dirichlet_scalar = torch.mean(dirichlet_channel, dim=-1, keepdim=True)  # [B, N, 1]

        dirichlet_channel = torch.log1p(dirichlet_channel)
        dirichlet_scalar = torch.log1p(dirichlet_scalar)

        dirichlet_channel = dirichlet_channel.permute(0, 2, 1).unsqueeze(-1)  # [B, C, N, 1]
        dirichlet_scalar = dirichlet_scalar.permute(0, 2, 1).unsqueeze(-1)  # [B, 1, N, 1]
        if coords is not None:
            xc, yc = coords[..., 0], coords[..., 1]
            x_min = xc.min(dim=-1, keepdim=True)[0]
            x_max = xc.max(dim=-1, keepdim=True)[0]
            y_min = yc.min(dim=-1, keepdim=True)[0]
            y_max = yc.max(dim=-1, keepdim=True)[0]
            norm_x = (xc - x_min) / (x_max - x_min + 1e-6)
            norm_y = (yc - y_min) / (y_max - y_min + 1e-6)
            coords_norm = torch.stack([norm_x, norm_y], dim=-1)  # [B, N, 2]

            coords_input = coords_norm.transpose(2, 1).contiguous()
            coord_neighbor = self.get_knn(coords_input, idx)  # [B, N, K, 2]
            coord_center = coords_norm.unsqueeze(2)  # [B, N, 1, 2]

            coord_diff = coord_neighbor - coord_center
            dirichlet_coord = torch.sum(coord_diff ** 2, dim=2)  # [B, N, 2]
            dirichlet_coord_scalar = torch.mean(dirichlet_coord, dim=-1, keepdim=True)  # [B, N, 1]

            dirichlet_coord_scalar = torch.log1p(dirichlet_coord_scalar)
            dirichlet_coord_scalar = dirichlet_coord_scalar.permute(0, 2, 1).unsqueeze(-1)  # [B, 1, N, 1]
        else:
            dirichlet_coord_scalar = torch.zeros(B, 1, N, 1, device=x.device)

        # ========== 3. 空语联合融合 ==========
        x_origin = x.unsqueeze(-1)
        x_cat = torch.cat([x_origin, dirichlet_channel, dirichlet_scalar, dirichlet_coord_scalar],
                          dim=1)  # [B, 2C+2, N, 1]
        x_input = self.uncertainty_fuse(x_cat)

        return self.edl_head(x_input)

    @torch.no_grad()
    def coord_to_grid_id(self, coords):
        B, N, _ = coords.shape
        xc, yc = coords[..., 0], coords[..., 1]
        x_min = xc.min(dim=-1, keepdim=True)[0]
        x_max = xc.max(dim=-1, keepdim=True)[0]
        x_norm = (xc - x_min) / (x_max - x_min + 1e-6)
        y_min = yc.min(dim=-1, keepdim=True)[0]
        y_max = yc.max(dim=-1, keepdim=True)[0]
        y_norm = (yc - y_min) / (y_max - y_min + 1e-6)

        x_idx = torch.floor(x_norm * 3).long()
        y_idx = torch.floor(y_norm * 3).long()
        x_idx = torch.clamp(x_idx, 0, 2)
        y_idx = torch.clamp(y_idx, 0, 2)
        return y_idx * 3 + x_idx

    @torch.no_grad()
    def get_spatial_anchors(self, uncertainty, coords):
        B, N, _ = coords.shape
        device = coords.device
        score = 1 - uncertainty.squeeze(1).squeeze(-1)
        grid_id = self.coord_to_grid_id(coords)

        region_anchors = torch.zeros((B, 9), dtype=torch.long, device=device)
        for b in range(B):
            for r in range(9):
                mask = (grid_id[b] == r)
                if torch.sum(mask) == 0:
                    anchor_idx = torch.argmax(score[b])
                else:
                    pts_idx = torch.where(mask)[0]
                    anchor_idx = pts_idx[torch.argmax(score[b, pts_idx])]
                region_anchors[b, r] = anchor_idx
        return region_anchors

    def build_dynamic_graph(self, local_idx, uncertainty, coords):
        B, N, _ = coords.shape
        local_idx = local_idx.view(B, N, self.knn_num)

        region_anchors = self.get_spatial_anchors(uncertainty, coords)
        grid_id = self.coord_to_grid_id(coords)

        connect_map = torch.tensor(REGION_CONNECT_MAP, dtype=torch.long, device=coords.device)
        connect_regions = connect_map[grid_id]
        anchors = torch.gather(
            region_anchors.unsqueeze(1).expand(-1, N, -1),
            dim=2,
            index=connect_regions
        )
        # 修复点：不再拼接local，只返回纯锚点索引，实现局部/长程完全解耦
        anchor_only_idx = anchors.reshape(B * N, -1)
        pure_local_idx = local_idx.reshape(B * N, -1)
        return pure_local_idx, anchor_only_idx

    def graph_conv(self, x, idx=None, coords=None, uncertainty=None):
        B, C, N = x.shape

        anchor_only_idx = None
        pure_local_idx = idx
        if coords is not None and self.use_dynamic_graph:
            if uncertainty is None:
                uncertainty, _ = self.compute_uncertainty(x, coords=coords)
            pure_local_idx, anchor_only_idx = self.build_dynamic_graph(idx, uncertainty, coords)

        def compute_gcn_feat(graph_idx):
            neighbor = self.get_knn(x, graph_idx)
            center = x.transpose(2, 1).unsqueeze(2)
            lap_feat = (neighbor - center).mean(dim=2).permute(0, 2, 1)
            out = torch.matmul(lap_feat.transpose(1, 2), self.gcn_weight)
            return out.transpose(1, 2)

        pure_local_feat = compute_gcn_feat(pure_local_idx)

        if self.use_selective_graph and coords is not None:
            anchor_only_feat = compute_gcn_feat(anchor_only_idx)
            u = uncertainty.squeeze(1).squeeze(-1)  # [B, N]
            active_u = F.relu(u - self.u_threshold)
            w_global = 1 - torch.exp(-3.0 * active_u)
            w_global = w_global.unsqueeze(1)  # [B, 1, N]
            fused_feat = pure_local_feat + w_global * anchor_only_feat
            out = fused_feat
        else:
            out = pure_local_feat

        return out

    def forward(self, x, coords=None):
        B, C, N, _ = x.shape
        node_feat = x.squeeze(3)

        idx = self.get_knn_idx(node_feat)
        uncertainty, alpha = self.compute_uncertainty(node_feat, coords=coords, idx=idx)

        global_feat = self.graph_conv(node_feat, idx, coords, uncertainty=uncertainty).unsqueeze(3)
        out = self.conv(global_feat)

        if self.use_residual:
            out = out + x

        return out, uncertainty, alpha

    def forward(self, x, coords=None):
        B, C, N, _ = x.shape
        node_feat = x.squeeze(3)
        idx = self.get_knn_idx(node_feat)
        uncertainty, alpha = self.compute_uncertainty(node_feat, coords=coords, idx=idx)
        global_feat = self.graph_conv(node_feat, idx, coords, uncertainty=uncertainty).unsqueeze(3)
        out = self.conv(global_feat)
        if self.use_residual:
            out = out + x

        return out, uncertainty, alpha,idx
class EDA(nn.Module):
    def __init__(self, in_channels, num_classes=2, k=9):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.eps = 1e-6
        self.gamma = nn.Parameter(
            torch.tensor(0.5)
        )
        self.calibration = nn.Sequential(
            LGDAFN(in_channels+1),
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def gather_knn(self, x, idx):
        B, C, N = x.shape
        x = x.transpose(2, 1).contiguous()
        neighbor = x.view(B * N, C)[idx]
        neighbor = neighbor.view(B, N, self.k, C)
        return neighbor
    def evidence_diffusion(self, alpha, reliability, knn_idx):
        neighbor_alpha = self.gather_knn(alpha, knn_idx)
        neighbor_reliability = self.gather_knn(reliability, knn_idx)
        weight = neighbor_reliability
        alpha_hat = (weight * neighbor_alpha).sum(2) / (weight.sum(2) + self.eps)
        alpha_hat = alpha_hat.transpose(1, 2)
        return alpha_hat
    def forward(self, feat, alpha, uncertainty, knn_idx):
        feature = feat
        alpha = alpha.squeeze(-1)
        uncertainty = uncertainty.squeeze(-1)
        reliability = 1.0 - uncertainty
        alpha_hat = self.evidence_diffusion(alpha, reliability, knn_idx)
        alpha_new = alpha + self.gamma *uncertainty * (alpha_hat - alpha)
        evidence_sum = alpha_new.sum(dim=1, keepdim=True)
        uncertainty_new = self.num_classes / (evidence_sum + 1e-6)
        alpha_new = alpha_new.unsqueeze(-1)
        uncertainty_new = uncertainty_new.unsqueeze(-1)
        calib_input = torch.cat([feature, uncertainty_new], dim=1)
        gate = self.calibration(calib_input)
        feat_new = feature + gate * feature * (1.0 - uncertainty_new)
        return feat_new, alpha_new, uncertainty_new

class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.half_channel = out_channel // 2
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.local0_stage1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.local0_stage2 = nn.Sequential(
            DGCNN_MAX_Block(self.k_num * 2, self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.local0_stage3 = nn.Sequential(
            OABlock(self.out_channel, clusters=256),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.local0_skip1_2 = nn.Sequential(
            nn.Conv2d(self.out_channel, self.out_channel, 1),
            nn.BatchNorm2d(self.out_channel),
        )
        self.local0_skip1_3 = nn.Sequential(
            nn.Conv2d(self.out_channel, self.out_channel, 1),
            nn.BatchNorm2d(self.out_channel),
        )
        self.local0_skip2_3 = nn.Sequential(
            nn.Conv2d(self.out_channel, self.out_channel, 1),
            nn.BatchNorm2d(self.out_channel),
        )

        self.local1_stage1 = nn.Sequential(
            OABlock(self.out_channel, clusters=128),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.local1_stage2 = nn.Sequential(
            DGCNN_MAX_Block(self.k_num, self.out_channel),
        )
        self.local1_skip_in_2 = nn.Sequential(
            nn.Conv2d(self.out_channel, self.out_channel, 1),
            nn.BatchNorm2d(self.out_channel),
        )

        self.udeg_0 = UDEG(self.out_channel)
        self.eda_0 = EDA(self.out_channel)
        self.udeg_1 = UDEG(self.out_channel)
        self.eda_1 = EDA(self.out_channel)

        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.alpha_0 = nn.Parameter(torch.tensor(0.5))
        self.alpha_1 = nn.Parameter(torch.tensor(0.5))

        if self.predict == True:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))
            self.cross_key_fuse = nn.Sequential(
                nn.Conv2d(2 * self.out_channel, self.out_channel, kernel_size=1),
                nn.BatchNorm2d(self.out_channel),
                nn.ReLU(inplace=True)
            )

    def down_sampling(self, x, y, weights, indices, features=None, predict=False, key_outs=None):
        B, _, N, _ = x.size()
        keep_num = int(N * self.sr)
        indices = indices[:, :keep_num]
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)
        key_outs_ds = []
        if key_outs is not None:
            for out in key_outs:
                out_indices = indices.repeat(1, self.out_channel, 1, 1)
                out_ds = torch.gather(out, dim=2, index=out_indices)
                key_outs_ds.append(out_ds)

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2,
                                     index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out, indices.squeeze(1).squeeze(-1), key_outs_ds
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
            return x_out, y_out, w_out, feature_out, indices.squeeze(1).squeeze(-1), key_outs_ds

    def forward(self, x, y, cross_key_outs=None):
        B, _, N, _ = x.size()
        coords = x[:, 0, :, :2]
        out = x.transpose(1, 3).contiguous()
        out = self.conv(out)
        self.out1 = out
        if cross_key_outs is not None:
            cross_key_cat = torch.cat([out, cross_key_outs], dim=1)
            out = out + self.cross_key_fuse(cross_key_cat)
        feat0_1 = self.local0_stage1(out)
        feat0_2_input = feat0_1 + self.local0_skip1_2(feat0_1)
        feat0_2 = self.local0_stage2(feat0_2_input)
        feat0_3_input = feat0_2 + self.local0_skip1_3(feat0_1) + self.local0_skip2_3(feat0_2)
        out_l = self.local0_stage3(feat0_3_input)
        w0 = self.linear_0(out_l).view(B, -1)
        out_g, uncertainty_0, alpha_0,idx_0 = self.udeg_0(out_l, coords)
        out_g,alpha_0,uncertainty_0 = self.eda_0(out_g,alpha_0,uncertainty_0,idx_0)
        a0 = torch.sigmoid(self.alpha_0)
        out_fused = a0 * out_l + (1 - a0) * out_g
        out = out_fused + out
        self.out2 = out
        feat1_1 = self.local1_stage1(out)
        feat1_2_input = feat1_1 + self.local1_skip_in_2(out)
        out_l = self.local1_stage2(feat1_2_input)
        w1 = self.linear_1(out_l).view(B, -1)
        out_g, uncertainty_1, alpha_1,idx_1 = self.udeg_1(out_l, coords)
        out_g, alpha_1, uncertainty_1 = self.eda_1(out_g, alpha_1, uncertainty_1, idx_1)
        a1 = torch.sigmoid(self.alpha_1)
        out_fused = a1 * out_l + (1 - a1) * out_g
        out = out_fused + out
        self.out3 = out
        key_outs = [self.out1, self.out2, self.out3]
        alphas = [alpha_0, alpha_1]
        w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
        if not self.predict:
            x_ds, y_ds, w0_ds, ds_indices, key_outs_ds = self.down_sampling(
                x, y, w0, indices, None, self.predict, key_outs=key_outs
            )
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds[:, :int(N * self.sr)]], ds_indices, key_outs_ds, alphas
        else:
            x_ds, y_ds, w0_ds, out, indices_ds1, key_outs_ds = self.down_sampling(
                x, y, w0, indices, out, self.predict, key_outs=key_outs
            )
            out = self.embed_2(out)
            w2 = self.linear_2(out)
            e_hat = weighted_8points(x_ds, w2)
            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds,
                                                          w1_ds[:, :int(N * self.sr)]], e_hat, key_outs_ds, alphas
class CrossStageAttentionFusion(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
    def forward(self, features):
        feat_sum = sum(features)
        c_att = self.channel_att(feat_sum)
        s_att = self.spatial_att(feat_sum)
        fused = feat_sum * c_att * s_att
        out = self.fuse_conv(fused) + feat_sum
        return out
class UnCertainNet(nn.Module):
    def __init__(self, config):
        super(UnCertainNet, self).__init__()
        self.ds_0 = DS_Block(initial=True, predict=False, out_channel=128, k_num=9, sampling_rate=0.6)
        self.ds_1 = DS_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr)
        self.cross_stage_fusion = CrossStageAttentionFusion(channels=128)
    def forward(self, x, y):
        B, _, N, _ = x.shape
        x1, y1, ws0, w_ds0, ds0_indices, key_outs_ds0, alphas0 = self.ds_0(x, y)
        ds0_all_feat = self.cross_stage_fusion(key_outs_ds0)
        w_ds0_0 = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)
        w_ds0_1 = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0_0.detach(), w_ds0_1.detach()], dim=-1)
        x2, y2, ws1, w_ds1, e_hat, key_outs_ds1, alphas1 = self.ds_1(x_, y1, cross_key_outs=ds0_all_feat)
        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)
        alpha_list = alphas0 + alphas1
        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat, alpha_list