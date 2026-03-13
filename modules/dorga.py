# modules/dorga_model.py : DORGA-Full 모델 정의
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# ═══════════════════════════════════════════════════════════
# DORGA 구성 요소
# ═══════════════════════════════════════════════════════════

class ConvPatchImportance(nn.Module):
    def __init__(self, grid, embed_dim, use_pos_bias=True, temp_max=5.0):
        super().__init__()
        self.grid = grid
        self.P = grid * grid
        self.norm = nn.LayerNorm(embed_dim)
        self.dwconv = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
        self.pwconv = nn.Conv2d(embed_dim, 1, 1)
        self.use_pos = use_pos_bias
        if use_pos_bias:
            self.pos_bias = nn.Parameter(torch.zeros(self.P))
        self.temperature = nn.Parameter(torch.tensor(2.0))
        self.temp_max = temp_max

    def forward(self, patch_tok):
        B, P, C = patch_tok.shape
        g = self.grid
        x = self.norm(patch_tok).view(B, g, g, C).permute(0, 3, 1, 2)
        x = F.gelu(self.dwconv(x))
        s = self.pwconv(x).flatten(2).squeeze(1)
        if self.use_pos:
            s = s + self.pos_bias.view(1, self.P).expand(B, -1)
        temp = self.temperature.clamp(min=0.1, max=self.temp_max)
        s_sm = torch.softmax(s / temp, dim=1)
        return s_sm * float(self.P), -(s_sm * (s_sm + 1e-8).log()).sum(dim=1)


class PoolerBox(nn.Module):
    def __init__(self, img_size=512, patch_size=16, num_regions=4, eps=1e-6):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_regions = num_regions
        self.eps = eps
        grid = img_size // patch_size
        self.grid = grid
        P = grid * grid
        pb = []
        for py in range(grid):
            for px in range(grid):
                y0, x0 = py * patch_size, px * patch_size
                pb.append((y0, x0, y0 + patch_size, x0 + patch_size))
        self.register_buffer(
            "patch_boxes",
            torch.tensor(pb, dtype=torch.float32).view(P, 4),
            persistent=False,
        )
        self.patch_area = patch_size * patch_size

    def _weights_area(self, rel):
        pix = rel * float(self.img_size)
        pix[..., 0::2] = pix[..., 0::2].clamp(0, self.img_size)
        pix[..., 1::2] = pix[..., 1::2].clamp(0, self.img_size)
        patches = self.patch_boxes.view(1, 1, -1, 4)
        rois = pix.unsqueeze(2)
        y0 = torch.maximum(rois[..., 0], patches[..., 0])
        x0 = torch.maximum(rois[..., 1], patches[..., 1])
        y1 = torch.minimum(rois[..., 2], patches[..., 2])
        x1 = torch.minimum(rois[..., 3], patches[..., 3])
        return ((y1 - y0).clamp(min=0) * (x1 - x0).clamp(min=0)) / float(self.patch_area)

    def forward(self, rel, patch_tokens, *, s_patch=None):
        B, P, C = patch_tokens.shape
        w_area = self._weights_area(rel)
        if s_patch is not None:
            s = torch.nan_to_num(s_patch, nan=1.0, posinf=3.0, neginf=0.0)
            s_mean = s.mean(dim=1, keepdim=True).clamp_min(self.eps)
            base = (s / s_mean).clamp(0.0, 3.0)
        else:
            base = torch.ones(B, P, device=patch_tokens.device)
        w_base = torch.nan_to_num(w_area * base.unsqueeze(1), nan=0.0)
        w = w_base / w_base.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return torch.einsum("brp,bpc->brc", w, patch_tokens), w, w_base


class SharedProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, z):
        return F.normalize(self.proj(z), p=2, dim=-1)


class ROISpecificProjector(nn.Module):
    def __init__(self, in_dim, out_dim, num_regions=4):
        super().__init__()
        self.num_regions = num_regions
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, out_dim),
            )
            for _ in range(num_regions)
        ])

    def forward(self, x):
        return F.normalize(
            torch.stack([self.nets[r](x[:, r, :]) for r in range(self.num_regions)], dim=1),
            p=2,
            dim=-1,
        )


class PatternPredictor(nn.Module):
    def __init__(self, in_dim, num_patterns, num_regions=4, num_classes=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + num_regions * num_classes, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, num_patterns),
        )

    def forward(self, cls_token, logits):
        return F.softmax(
            self.mlp(torch.cat([cls_token, logits.detach().flatten(1)], dim=-1)),
            dim=-1,
        )


class DynamicPrior(nn.Module):
    def __init__(self, pi_global, pi_patterns, smoothing=0.05):
        super().__init__()
        C = pi_patterns.shape[3]
        pp = (1 - smoothing) * pi_patterns + smoothing / C
        pg = (1 - smoothing) * pi_global + smoothing / C
        self.register_buffer(
            "pi_patterns", pp / pp.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        )
        self.register_buffer(
            "pi_global", pg / pg.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        )

    def forward(self, rho):
        d = torch.einsum("bk,kijcd->bijcd", rho, self.pi_patterns)
        return d / d.sum(dim=-2, keepdim=True).clamp(min=1e-8)


class DOGAN(nn.Module):
    def __init__(self, dim, num_classes, num_regions=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.C = num_classes
        self.R = num_regions
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.att_scale = nn.Parameter(torch.tensor(10.0))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.att_dropout = nn.Dropout(dropout)
        self.step_size = nn.Parameter(torch.tensor(0.0))

    def _l2n(self, x, eps=1e-8):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    def _slerp(self, h, target, t, eps=1e-7):
        cos_t = (h * target).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)
        theta = torch.acos(cos_t)
        sin_t = torch.sin(theta)
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        close = (theta.abs() < eps) | (sin_t.abs() < eps)
        ch = torch.sin((1 - t) * theta) / sin_t.clamp_min(eps)
        ct = torch.sin(t * theta) / sin_t.clamp_min(eps)
        return self._l2n(
            torch.where(close, self._l2n((1 - t) * h + t * target), ch * h + ct * target)
        )

    def forward(self, u, v, rho, pi_prior, class_anchors, **kw):
        B, R, D = u.shape
        C = self.C
        dev = u.device
        h = self._l2n(u)
        a = self._l2n(class_anchors)

        with torch.no_grad():
            p = F.one_hot(
                torch.einsum("brd,cd->brc", h, a).argmax(-1), C
            ).float()

        pi = pi_prior.detach().clone()
        pi[:, torch.eye(R, device=dev, dtype=torch.bool)] = torch.eye(C, device=dev)

        q_ij = torch.einsum("bijcd,bjd->bijc", pi, p)
        qa = self.q_proj(v).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        ka = self.k_proj(v).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        alpha = self.att_dropout(
            F.softmax((qa @ ka.transpose(-2, -1)) * self.att_scale, dim=-1)
        ).mean(dim=1)

        qbar = torch.einsum("bij,bijc->bic", alpha, q_ij)
        ci = torch.arange(C, device=dev, dtype=qbar.dtype)
        mu = (qbar * ci).sum(-1)
        var = (qbar * (ci - mu.unsqueeze(-1)).pow(2)).sum(-1)
        xi = (1.0 - var / (((C - 1) ** 2) / 4.0)).clamp(0, 1)

        A = torch.einsum("brc,cd->brd", qbar, a.detach())
        return {"h": self._slerp(h, self._l2n(A), torch.sigmoid(self.step_size) * xi)}


class AngularClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale_s2 = nn.Parameter(torch.tensor(5.0).log())
        self.logit_scale_s3 = nn.Parameter(torch.tensor(5.0).log())

    def forward(self, x, anchors, stage="s3"):
        x_n = F.normalize(x, dim=-1)
        a_n = F.normalize(anchors, dim=-1)
        theta = torch.acos(
            torch.einsum("...d,cd->...c", x_n, a_n).clamp(-1 + 1e-7, 1 - 1e-7)
        )
        scale = (
            self.logit_scale_s2 if stage == "s2" else self.logit_scale_s3
        ).exp().clamp(1.0, 10.0)
        return scale * (-theta)


# ═══════════════════════════════════════════════════════════
# DORGA-Full 메인 모델
# ═══════════════════════════════════════════════════════════

class DORGAFullModel(nn.Module):
    def __init__(self, vit, R=4, C=5, K=7, proj_dim=768,
                 pi_global=None, pi_patterns=None):
        super().__init__()
        self.vit = vit
        self.R = R
        self.C = C
        self.K = K
        self.proj_dim = proj_dim

        isz = vit.patch_embed.img_size
        isz = isz[0] if isinstance(isz, (list, tuple)) else isz
        psz = vit.patch_embed.patch_size
        psz = psz[0] if isinstance(psz, (list, tuple)) else psz

        self.patch_importance = ConvPatchImportance(grid=isz // psz, embed_dim=768)
        self.pooler = PoolerBox(img_size=isz, patch_size=psz, num_regions=R)
        self.pattern_predictor = PatternPredictor(768, K, R, C)
        self.dynamic_prior = DynamicPrior(pi_global, pi_patterns, smoothing=0.05)
        self.shared_projector = SharedProjector(768, proj_dim)
        self.roi_specific_projector = ROISpecificProjector(768, proj_dim, R)

        angles = torch.linspace(0, math.pi, C)
        anch = torch.zeros(C, proj_dim)
        anch[:, 0] = torch.cos(angles)
        anch[:, 1] = torch.sin(angles)
        self.class_anchors = nn.Parameter(anch)

        self.cls_head = AngularClassifier()
        self.gnn = DOGAN(proj_dim, C, R, num_heads=4, dropout=0.1)

    def _forward_tokens(self, x):
        vit = self.vit
        B = x.shape[0]
        x = vit.patch_embed(x)
        cls = vit.cls_token.expand(B, -1, -1)
        if getattr(vit, "pos_embed", None) is not None:
            x = x + vit.pos_embed[:, 1 : 1 + x.shape[1]]
            cls = cls + vit.pos_embed[:, :1]
        x = vit.pos_drop(torch.cat([cls, x], dim=1))
        for blk in vit.blocks:
            x = blk(x)
        return vit.norm(x)

    def forward(self, x, rel):
        seq = self._forward_tokens(x)
        cls_tok, ptok = seq[:, 0, :], seq[:, 1:, :]

        s, _ = self.patch_importance(ptok)
        z, _, _ = self.pooler(rel=rel, patch_tokens=ptok, s_patch=s)

        u = self.shared_projector(z)
        v = self.roi_specific_projector(z)

        logits_s2 = self.cls_head(u, self.class_anchors, stage="s2")
        rho = self.pattern_predictor(cls_tok.detach(), logits_s2.detach())
        pi = self.dynamic_prior(rho)

        gnn_out = self.gnn(
            u=u, v=v, rho=rho, pi_prior=pi, class_anchors=self.class_anchors
        )
        logits_s3 = self.cls_head(gnn_out["h"], self.class_anchors, stage="s3")

        return {"logits_s3": logits_s3, "u": u, "h": gnn_out["h"]}


# ═══════════════════════════════════════════════════════════
# ViT 팩토리 함수
# ═══════════════════════════════════════════════════════════

def vit_base_patch16_512(**kwargs):
    from timm.models.vision_transformer import VisionTransformer

    return VisionTransformer(
        img_size=512,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )