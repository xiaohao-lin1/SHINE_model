"""SHINE — Small-world Hierarchical Interconnected Graph Neural Network.

Architecture (manuscript Sec. III):
  EEG  -- FPA --> per-channel power features  -- F2G --> 8 region-grouped graphs
       -- LocalGraphs --> 8 region embeddings  -- PDG --> stacked GCN with
       layer-wise decay  -- Classifier --> binary prediction.

Module map (class name → manuscript module):
  ``MultiscaleConv`` (the three ``fpa_branch_*`` instances) → FPA / multiscale conv (Eq. 1)
  ``LearnableVarianceWindow``                               → FPA / sliding window + learnable variance (Eq. 2-3)
  ``LearnableVariance``                                     → FPA / γ·Var(·) + β
  ``fusion_block``                                          → FPA / fusion (Eq. 4)
  channel reorder (``forward`` line 1)                      → F2G (Eq. 5)
  ``_local_graph_filter`` + ``RegionAggregator``            → Local Graphs (Eq. 6-7)
  ``GCNLayer`` + ``pdg_layers`` + ``_layerwise_decay``      → PDG (Eq. 8-14)
  ``classifier``                                            → MLP head (Eq. 15)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GCNLayer(nn.Module):
    """One GCN layer: ``ReLU(adj @ (x W - b))``. PDG (Eq. 12-13)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        out = torch.matmul(x, self.weight) - self.bias
        return F.relu(torch.matmul(adj, out))


class LearnableVariance(nn.Module):
    """V_s = γ · Var(x) + β  (manuscript Eq. 3). γ, β are scalar learnable parameters."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        var = x.var(dim=self.dim, keepdim=True)
        var = self.gamma * var + self.beta
        var = torch.clamp(var, 1e-6, 1e6)
        return torch.log(var)


class LearnableVarianceWindow(nn.Module):
    """FPA sliding window + learnable variance + average pooling.

    Steps: unfold along time into windows of ``variance_window_samples`` samples
    with stride ``variance_stride_ratio · variance_window_samples`` (Eq. 2);
    apply LearnableVariance over each window (Eq. 3); average-pool across windows.
    """

    def __init__(self, dim, length, step, variance_window_samples, variance_stride_ratio):
        super().__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))
        self.variance = LearnableVariance(dim=-1)
        self.variance_window_samples = variance_window_samples
        self.variance_stride_ratio = variance_stride_ratio

    def forward(self, x):
        x = x.unfold(
            self.dim,
            self.variance_window_samples,
            int(self.variance_stride_ratio * self.variance_window_samples),
        )
        var = self.variance(x).squeeze(-1)
        return self.pooling(var)


class RegionAggregator:
    """Mean-pool channels within each functional region → one node per region (Eq. 7)."""

    def __init__(self, channels_per_region):
        self.channels_per_region = channels_per_region
        self._cum_idx = self._cumulative_indices(channels_per_region)
        self.num_regions = len(channels_per_region)

    @staticmethod
    def _cumulative_indices(counts):
        out, running = [], 0
        for n in [0] + counts:
            running += n
            out.append(running)
        return out[1:]

    def forward(self, x):
        # x: (batch, channels, features)
        regions = []
        for i in range(self.num_regions):
            start = self._cum_idx[i]
            end = self._cum_idx[i + 1] if i < self.num_regions - 1 else None
            regions.append(x[:, start:end, :].mean(dim=1) if end is not None
                           else x[:, start:, :].mean(dim=1))
        return torch.stack(regions, dim=1)


class SHINE(nn.Module):
    """Top-level model: FPA → F2G → Local Graphs → PDG → Classifier."""

    def __init__(self, args, dataset):
        super().__init__()

        num_classes = dataset.n_classes
        in_channels, num_eeg_channels, num_timesteps = (
            dataset.in_channels, dataset.n_channels, dataset.n_timesteps,
        )
        sampling_rate = dataset.sampling_rate

        # -- Manuscript hyperparameters (Sec. III, Table I) --
        num_temporal_filters = args.num_temporal_filters
        region_dim = args.region_dim
        dropout_rate = args.dropout_rate
        window_size_samples = args.window_size_samples
        window_stride_ratio = args.window_stride_ratio
        variance_window_samples = args.variance_window_samples
        variance_stride_ratio = args.variance_stride_ratio
        self.num_gcn_layers = args.num_gcn_layers
        self.alpha = args.alpha            # PDG decay rate (manuscript α, Eq. 11)
        self.total_epochs = args.epochs    # PDG decay schedule horizon T

        self.channels_per_region = args.channels_per_region
        self.channel_reorder_idx = args.channel_reorder_idx
        self.num_regions = len(self.channels_per_region)
        self.num_eeg_channels = num_eeg_channels

        # -- FPA: three multiscale temporal convolutions (s ∈ {1,2,3}, λ=0.5) --
        self.fpa_window_fractions = [0.5, 0.25, 0.125]
        self.fpa_branch_1 = self._make_fpa_branch(
            in_channels, num_temporal_filters, sampling_rate,
            self.fpa_window_fractions[0], window_size_samples, window_stride_ratio,
            variance_window_samples, variance_stride_ratio,
        )
        self.fpa_branch_2 = self._make_fpa_branch(
            in_channels, num_temporal_filters, sampling_rate,
            self.fpa_window_fractions[1], window_size_samples, window_stride_ratio,
            variance_window_samples, variance_stride_ratio,
        )
        self.fpa_branch_3 = self._make_fpa_branch(
            in_channels, num_temporal_filters, sampling_rate,
            self.fpa_window_fractions[2], window_size_samples, window_stride_ratio,
            variance_window_samples, variance_stride_ratio,
        )
        self.bn_pre_fusion = nn.BatchNorm2d(num_temporal_filters)
        self.bn_post_fusion = nn.BatchNorm2d(num_temporal_filters)

        # -- FPA fusion block (Eq. 4) --
        self.fusion_block = nn.Sequential(
            nn.Conv2d(num_temporal_filters, num_temporal_filters, kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2)),
        )

        # -- Local Graphs: per-channel learnable W and b (Eq. 6) --
        feature_dim = self._fpa_output_feature_dim((in_channels, num_eeg_channels, num_timesteps))
        self.local_W = nn.Parameter(
            torch.FloatTensor(num_eeg_channels, feature_dim), requires_grad=True,
        )
        nn.init.xavier_uniform_(self.local_W)
        self.local_b = nn.Parameter(
            torch.zeros((1, num_eeg_channels, 1), dtype=torch.float32), requires_grad=True,
        )

        # -- Region aggregation (mean over channels in each region, Eq. 7) --
        self.region_aggregator = RegionAggregator(self.channels_per_region)

        # -- PDG: learnable region-region mask M + K stacked GCN layers --
        self.region_mask = nn.Parameter(
            torch.FloatTensor(self.num_regions, self.num_regions), requires_grad=True,
        )
        nn.init.xavier_uniform_(self.region_mask)
        self.bn_regions = nn.BatchNorm1d(self.num_regions)
        self.bn_pdg_concat = nn.BatchNorm1d(self.num_regions * self.num_gcn_layers)

        self.pdg_layers = nn.ModuleList()
        for k in range(self.num_gcn_layers):
            in_dim = feature_dim if k == 0 else region_dim
            self.pdg_layers.append(GCNLayer(in_dim, region_dim))

        # -- Classifier (Eq. 15) --
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_regions * region_dim * self.num_gcn_layers, num_classes),
        )

    @staticmethod
    def _make_fpa_branch(in_channels, num_filters, sampling_rate, window_fraction,
                         window_size_samples, window_stride_ratio,
                         variance_window_samples, variance_stride_ratio):
        kernel = (1, int(window_fraction * sampling_rate))
        return nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=kernel, stride=(1, 1), padding='same'),
            LearnableVarianceWindow(
                dim=-1,
                length=window_size_samples,
                step=int(window_stride_ratio * window_size_samples),
                variance_window_samples=variance_window_samples,
                variance_stride_ratio=variance_stride_ratio,
            ),
        )

    def _fpa_output_feature_dim(self, input_size):
        """Run a single dummy pass through FPA + fusion to derive the per-channel feature dim."""
        in_ch, n_chan, n_time = input_size
        dummy = torch.ones((1, in_ch, n_chan, n_time))
        out = torch.cat([
            self.fpa_branch_1(dummy),
            self.fpa_branch_2(dummy),
            self.fpa_branch_3(dummy),
        ], dim=-1)
        out = self.bn_pre_fusion(out)
        out = self.fusion_block(out)
        out = self.bn_post_fusion(out)
        out = out.permute(0, 2, 1, 3).reshape(out.size(0), out.size(2), -1)
        return out.size(-1)

    def forward(self, x, epoch=None, train=True, *vars, **kwargs):
        # F2G — channel reorder into 8 functional regions (Eq. 5)
        x = x[:, :, self.channel_reorder_idx, :]

        # FPA — multiscale conv + sliding-window learnable variance + concat
        out = torch.cat([
            self.fpa_branch_1(x),
            self.fpa_branch_2(x),
            self.fpa_branch_3(x),
        ], dim=-1)
        out = self.bn_pre_fusion(out)
        out = self.fusion_block(out)
        out = self.bn_post_fusion(out)
        out = out.permute(0, 2, 1, 3).reshape(out.size(0), out.size(2), -1)

        # Local Graphs — learnable per-channel weighting + ReLU + region-mean
        out = self._local_graph_filter(out, self.local_W)
        out = self.region_aggregator.forward(out)

        # PDG — region-region adjacency + stacked GCN with layer-wise decay
        adj = self._compute_region_adjacency(out)
        out = self.bn_regions(out)

        decayed = []
        for k, gcn in enumerate(self.pdg_layers):
            out = gcn(out, adj)
            out = out * self._layerwise_decay(k, epoch, train)
            decayed.append(out)

        decayed = torch.cat([d.unsqueeze(1) for d in decayed], dim=1)
        decayed = decayed.flatten(start_dim=1, end_dim=2)
        out = self.bn_pdg_concat(decayed).view(decayed.size(0), -1)

        # Classifier
        return self.classifier(out)

    # ---------- module helpers ----------

    def _local_graph_filter(self, x, W):
        """Local Graphs: U = ReLU(W ⊙ x − b)  (Eq. 6)."""
        W = W.unsqueeze(0).repeat(x.size(0), 1, 1)
        return F.relu(torch.mul(x, W) - self.local_b)

    def _compute_region_adjacency(self, h, self_loop=True):
        """PDG region-region adjacency: ReLU(M ⊙ ⟨h, h⟩) + I, then degree-normalised (Eq. 8-10)."""
        adj = torch.bmm(h, h.permute(0, 2, 1))                             # ⟨h, h⟩
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.region_mask + self.region_mask.transpose(0, 1)))
        if self_loop:
            adj = adj + torch.eye(num_nodes, device=h.device)
        rowsum = adj.sum(dim=-1)
        rowsum = torch.where(rowsum == 0, torch.ones_like(rowsum), rowsum)
        d_inv_sqrt = torch.diag_embed(rowsum.pow(-0.5))
        return torch.bmm(torch.bmm(d_inv_sqrt, adj), d_inv_sqrt)

    def _layerwise_decay(self, k, epoch, train):
        """f(k, t) = exp(−α · (1 − t/T) · k)  (manuscript Eq. 11)."""
        if train and epoch is not None:
            effective_alpha = torch.tensor(self.alpha) * (1 - epoch / self.total_epochs)
        else:
            effective_alpha = torch.tensor(self.alpha)
        return torch.exp(-effective_alpha * k)
