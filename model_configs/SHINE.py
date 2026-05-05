"""SHINE training configuration.

Hyperparameter names map onto the manuscript notation:
  ``num_temporal_filters``   = F (Eq. 1, FPA per-branch output channels)
  ``window_size_samples``    = q (FPA fusion-block average pool length, samples)
  ``window_stride_ratio``    = stride ratio for the FPA fusion average pool
  ``variance_window_ms``     = w in milliseconds (FPA learnable-variance window)
  ``variance_window_samples``= w in samples (computed from _ms × sampling rate)
  ``variance_stride_ratio``  = l/w (FPA learnable-variance window stride ratio)
  ``region_dim``             = d (PDG GCN per-region embedding dim, Eq. 12)
  ``num_gcn_layers``         = K (PDG depth, Eq. 13)
  ``alpha``                  = α (PDG decay rate, Eq. 11)
"""
import math


class Configs:
    """Hyperparameters and training knobs for SHINE on the open/close datasets."""

    def __init__(self):
        # --------- model architecture (manuscript-named) ---------
        self.num_classes = 2
        self.num_temporal_filters = 64           # F in Eq. 1
        self.region_dim = 32                     # d in Eq. 12
        self.dropout_rate = 0.5
        self.window_size_samples = 32            # q
        self.window_stride_ratio = 0.25
        self.variance_window_ms = 250            # w (ms)
        self.variance_window_samples = math.floor(self.variance_window_ms / 1000 * 250)
        self.variance_stride_ratio = 0.1
        self.alpha = 0.4                         # α in Eq. 11
        self.num_gcn_layers = 2                  # K
        self.channel_reorder_idx, self.channels_per_region = self._build_region_grouping()

        # --------- training schedule ---------
        self.epochs = 100
        self.batch_size = 16 * 3
        self.loss = 'ce_F'
        self.clip_grad_norm = 0
        self.step_size = 5
        self.LS = True
        self.LS_rate = 0.1

        self.learning_rate = 1e-3
        self.scheduler = 'cosine'
        self.T_max = self.epochs
        self.decay_style = None
        self.warmup_percent = 0.1
        self.warmup_lr_initial = 0.001
        self.warmup_lr_max = 0.001

        self.optimizer = 'adamw'
        self.adam_beta = (0.9, 0.999)
        self.adam_weight_decay = 0.0
        self.adamw_beta = (0.9, 0.999)
        self.adamw_weight_decay = 0.1

        self.seed = 1

    def __repr__(self):
        return str(vars(self))

    @staticmethod
    def _open_close_channel_order():
        """61-channel order present in the open/close datasets (matches data files on disk)."""
        return [
            'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
            'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4',
            'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1',
            'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3',
            'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4',
            'FT8', 'F6', 'AF8', 'AF4', 'F2',
        ]

    @staticmethod
    def _functional_regions():
        """Eight neuro-functional regions used by F2G (manuscript Fig. 2)."""
        return {
            'Anterior Frontal':  ['AF7', 'Fp1', 'Fp2', 'AFz', 'AF8', 'AF3', 'AF4'],
            'Frontal':           ['F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
            'Frontocentral':     ['FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6'],
            'Central':           ['C1', 'C2', 'C3', 'Cz', 'C4', 'C5', 'C6'],
            'Centroparietal':    ['CP1', 'CP2', 'CP3', 'CPz', 'CP4', 'CP5', 'CP6'],
            'Parietal':          ['P1', 'P2', 'P3', 'P4', 'Pz', 'P5', 'P6', 'P7', 'P8'],
            'Temporal':          ['T7', 'T8', 'FT7', 'FT8', 'TP7', 'TP8', 'FT9', 'FT10'],
            'Occipital':         ['PO3', 'PO4', 'PO7', 'PO8', 'O1', 'O2', 'Oz', 'POz'],
        }

    def _build_region_grouping(self):
        """Compute (channel reorder indices, channels-per-region) for F2G."""
        original_order = self._open_close_channel_order()
        regions = list(self._functional_regions().values())
        reorder_idx = []
        channels_per_region = []
        for region in regions:
            channels_per_region.append(len(region))
            for ch in region:
                reorder_idx.append(original_order.index(ch))
        return reorder_idx, channels_per_region
