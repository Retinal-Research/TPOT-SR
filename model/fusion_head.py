import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionRefineHead(nn.Module):
    """Fuse native x@2048 with TPOT-enhanced@2048 at the same resolution."""

    def __init__(self, in_channels=6, n_feat=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, n_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feat, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x_hr, e_hr):
        residual = self.body(torch.cat([x_hr, e_hr], dim=1))
        return torch.clamp(e_hr + residual, 0, 1)


class GeneratorWithSR(nn.Module):
    def __init__(self, base_generator, work_size=2048, sr_feat=64):
        super().__init__()
        self.work_size = work_size
        self.base = base_generator
        for param in self.base.parameters():
            param.requires_grad = False
        self.fusion_head = FusionRefineHead(in_channels=6, n_feat=sr_feat)

    def forward(self, x_hr, x_lr):
        with torch.no_grad():
            enhanced_lr = torch.clamp(self.base(x_lr), 0, 1)

        target_size = x_hr.shape[-2:]
        enhanced_hr = F.interpolate(
            enhanced_lr,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        output = self.fusion_head(x_hr, enhanced_hr)
        return output, enhanced_lr, enhanced_hr