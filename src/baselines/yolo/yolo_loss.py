import torch
import torch.nn as nn
import torch.nn.functional as F

class YoloLoss(nn.Module):
    def __init__(self, S, B, C, λ_coord=5., λ_noobj=0.5):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.λ_coord = λ_coord
        self.λ_noobj = λ_noobj
        self.ce = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, pred, target):
        """
        pred, target: [B, S, S, C + 5B]
        x,y,w,h,conf: sigmoid tartományban (0–1)
        """
        Bsz = pred.size(0)
        obj_mask    = target[..., self.C+4::5]            # [B,S,S,B]
        noobj_mask  = 1.0 - obj_mask

        # --- koordináták ---
        p_coord = pred[..., self.C:self.C+4*self.B].reshape(Bsz, self.S, self.S, self.B, 4)
        t_coord = target[..., self.C:self.C+4*self.B].reshape_as(p_coord)

        xy_loss = F.mse_loss(
            obj_mask.unsqueeze(-1) * p_coord[..., :2],
            obj_mask.unsqueeze(-1) * t_coord[..., :2],
            reduction='sum'
        )
        wh_loss = F.mse_loss(
            obj_mask.unsqueeze(-1) * torch.sqrt(p_coord[..., 2:4].clamp(min=1e-6)),
            obj_mask.unsqueeze(-1) * torch.sqrt(t_coord[..., 2:4].clamp(min=1e-6)),
            reduction='sum'
        )

        # --- confidence ---
        p_conf = pred[..., self.C+4::5]          # [B,S,S,B]
        t_conf = obj_mask                       # GT: 1 ott, ahol obj
        pos_loss = (obj_mask * (p_conf - 1)**2).sum()
        neg_loss = (self.λ_noobj * noobj_mask * p_conf**2).sum()
        conf_loss = pos_loss + neg_loss

        # --- class (logit + CE) ---
        p_cls = pred[..., :self.C]              # [B,S,S,C]
        t_cls = target[..., :self.C].argmax(-1) # [B,S,S]  one-hot → index
        class_loss = self.ce(
            p_cls.view(-1, self.C),
            t_cls.view(-1)
        )

        total = (self.λ_coord * (xy_loss + wh_loss) +
                 conf_loss + class_loss) / Bsz
        return total