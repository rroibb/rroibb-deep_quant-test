import torch
import torch.nn as nn
import numpy as np
from .base import BaseDeepModel


class MultiModalFusionModel(BaseDeepModel):
    def __init__(self, lstm_dim=128, transformer_dim=64, cnn_dim=64,
                 xgb_dim=32, hidden_dim=128, num_regimes=3, output_dim=1):
        super().__init__(name='MultiModalFusion')
        self.output_dim = output_dim

        input_dim = lstm_dim + transformer_dim + cnn_dim + xgb_dim * 3

        self.regime_embedding = nn.Embedding(num_regimes, 16)
        input_dim += 16

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.feature_projectors = nn.ModuleDict({
            'lstm': nn.Linear(1, lstm_dim),
            'transformer': nn.Linear(1, transformer_dim),
            'cnn': nn.Linear(1, cnn_dim),
            'xgb_trend': nn.Linear(1, xgb_dim),
            'xgb_mean_reversion': nn.Linear(1, xgb_dim),
            'xgb_neutral': nn.Linear(1, xgb_dim),
        })

        self.attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True, dropout=0.1
        )
        self.attn_proj = nn.Linear(128, 64)

        self.regime_gate = nn.Sequential(
            nn.Linear(16, 8),
            nn.Sigmoid(),
        )

    def forward(self, lstm_out, transformer_out, cnn_out,
                xgb_trend=None, xgb_rev=None, xgb_neutral=None, regime_idx=None):
        modality_scores = {
            'lstm': lstm_out,
            'transformer': transformer_out,
            'cnn': cnn_out,
        }
        for name, val in [('xgb_trend', xgb_trend), ('xgb_mean_reversion', xgb_rev), ('xgb_neutral', xgb_neutral)]:
            if val is not None:
                modality_scores[name] = val

        projected = []
        proj_names = []
        for name, score in modality_scores.items():
            if score.numel() == 0:
                score = torch.zeros(1, device=self.fusion[0].weight.device)
            if score.dim() == 0:
                score = score.unsqueeze(0)
            if score.dim() == 1:
                score = score.unsqueeze(1)
            proj = self.feature_projectors[name](score)
            projected.append(proj)
            proj_names.append(name)

        scores_flat = torch.stack([p.mean(dim=-1) for p in projected], dim=1)
        stacked = scores_flat.unsqueeze(-1).expand(-1, -1, 128)
        attn_out, _ = self.attention(stacked, stacked, stacked)
        attn_feat = attn_out.mean(dim=1)

        combined = torch.cat([p for p in projected], dim=-1)

        if regime_idx is not None:
            if regime_idx.dim() == 0:
                regime_idx = regime_idx.unsqueeze(0)
            reg_emb = self.regime_embedding(regime_idx.long())
            gates = self.regime_gate(reg_emb)
            combined_mod = []
            for i, name in enumerate(proj_names):
                combined_mod.append(projected[i] * gates[:, i:i+1])
            combined = torch.cat(combined_mod, dim=-1)
            combined = torch.cat([combined, reg_emb], dim=-1)
        else:
            combined = torch.cat([combined, attn_feat], dim=-1)

        out = self.fusion(combined)
        if self.output_dim == 1:
            out = out.squeeze(-1)
        return out

    def fuse_predictions(self, predictions_dict, regime_idx=None):
        batch = 1
        device = next(self.parameters()).device

        default_tensor = torch.zeros(batch, device=device)

        tensor_dict = {}
        for name in ['lstm', 'transformer', 'cnn', 'xgb_trend', 'xgb_mean_reversion', 'xgb_neutral']:
            val = predictions_dict.get(name, 0)
            if isinstance(val, (int, float, np.floating)):
                if np.isnan(val):
                    val = 0.0
                tensor_dict[name] = torch.tensor([val], dtype=torch.float32, device=device)
            elif isinstance(val, np.ndarray):
                val = np.nan_to_num(val, 0.0)
                tensor_dict[name] = torch.from_numpy(val).float().to(device)
            elif isinstance(val, torch.Tensor):
                tensor_dict[name] = torch.nan_to_num(val.to(device), 0.0)
            else:
                tensor_dict[name] = default_tensor

        if regime_idx is None:
            reg_tensor = None
        elif isinstance(regime_idx, (int, np.integer)):
            reg_tensor = torch.tensor(regime_idx, device=device)
        else:
            try:
                reg_tensor = torch.tensor(regime_idx, dtype=torch.long, device=device) if np.ndim(regime_idx) > 0 else default_tensor
            except:
                reg_tensor = default_tensor

        self.eval()
        with torch.no_grad():
            result = self.forward(
                tensor_dict['lstm'], tensor_dict['transformer'],
                tensor_dict['cnn'],
                xgb_trend=tensor_dict.get('xgb_trend', default_tensor),
                xgb_rev=tensor_dict.get('xgb_mean_reversion', default_tensor),
                xgb_neutral=tensor_dict.get('xgb_neutral', default_tensor),
                regime_idx=reg_tensor
            )
        return float(result.cpu().numpy().squeeze().item())
