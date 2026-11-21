import torch, torch.nn as nn
class BeatNet(nn.Module):
    """Small 1-D CNN on a 256-sample beat window, plus 4 RR-interval features."""
    def __init__(self, n_cls=5, use_rr=True, ch=(16, 32, 64), in_ch=1, n_f=4):
        super().__init__(); self.use_rr = use_rr; layers = []; c0 = in_ch
        for c in ch:
            layers += [nn.Conv1d(c0, c, 7, padding=3), nn.BatchNorm1d(c), nn.ReLU(),
                       nn.Conv1d(c, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2)]; c0 = c
        self.conv = nn.Sequential(*layers); self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(c0 + (n_f if use_rr else 0), 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_cls))
    def forward(self, w, f):
        z = self.pool(self.conv(w if w.dim() == 3 else w.unsqueeze(1))).squeeze(-1)
        if self.use_rr: z = torch.cat([z, f], 1)
        return self.head(z)
