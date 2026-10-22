"""η (期待値座標) と局所 Fisher 計量の抽出 — 監査済み実装.

監査 (2026-07-09) で確定した修正を反映:
  - V は pre-attention LayerNorm を通した hidden に W_v を適用する
    (フック検証で実 Value と厳密一致。LN を省くと cos 0.72–0.81 まで乖離)
  - GPT-NeoX の query_key_value はヘッド毎 [q,k,v] インターリーブで分割する
    (ブロック分割は cos ≈ 0 の誤った V を返す)
  - η と G は同じ attention 重みで計算する (平均と分散の分布を一致させる)
"""
import numpy as np
import torch

__all__ = ['extract_z_w', 'eta_and_metrics']


def _last_attention(model):
    """(最終ブロック, attention入力のLN, value取得関数) を返す."""
    cfg = model.config
    if hasattr(model, 'gpt_neox'):                      # Pythia / GPT-NeoX
        layer = model.gpt_neox.layers[-1]
        hs = layer.attention.head_size

        def get_v(h_ln):
            qkv = layer.attention.query_key_value(h_ln).squeeze(0)
            T = qkv.shape[0]
            return qkv.view(T, -1, 3 * hs)[..., 2 * hs:].reshape(T, -1)
        return layer, layer.input_layernorm, get_v
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):   # GPT-2
        layer = model.transformer.h[-1]
        D = cfg.hidden_size

        def get_v(h_ln):
            return layer.attn.c_attn(h_ln).squeeze(0)[:, 2 * D:]
        return layer, layer.ln_1, get_v
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):          # LLaMA系
        layer = model.model.layers[-1]

        def get_v(h_ln):
            v = layer.self_attn.v_proj(h_ln).squeeze(0)
            n_h, n_kv = cfg.num_attention_heads, getattr(cfg, 'num_key_value_heads',
                                                         cfg.num_attention_heads)
            if n_kv < n_h:                              # GQA: kv head を複製
                hd = v.shape[-1] // n_kv
                v = v.view(-1, n_kv, hd).repeat_interleave(n_h // n_kv, 1).reshape(v.shape[0], -1)
            return v
        return layer, layer.input_layernorm, get_v
    raise ValueError(f'unsupported architecture: {type(model).__name__}')


@torch.no_grad()
def extract_z_w(model, input_ids):
    """1系列から (V: (T,D), w: (T,)) を抽出。V は実 Value と厳密一致 (LN 適用済み).

    w は最終トークンが各トークンへ向ける attention のヘッド平均 (和1)。
    """
    layer, ln, get_v = _last_attention(model)
    out = model(input_ids=input_ids, output_hidden_states=True, output_attentions=True)
    h_in = out.hidden_states[-2]                        # 最終層への入力
    V = get_v(ln(h_in)).float()                         # ← LN を通す (監査発見 #3 の修正)
    sl = input_ids.shape[1]
    w = out.attentions[-1][0, :, sl - 1, :].mean(0).float()
    w = w / (w.sum() + 1e-12)
    return V, w, out


def eta_and_metrics(V, w, lam=1e-4):
    """attention 重みで一貫させた η, G_theta (共分散), G_eta (逆行列, float64).

    η 座標での Fisher 計量は G_eta = G_theta^{-1} (fisher.py の距離計算で使う)。
    """
    eta = torch.einsum('t,td->d', w, V)
    diff = (V - eta[None, :]).double()
    G_theta = torch.einsum('t,ti,tj->ij', w.double(), diff, diff)
    G_theta = G_theta + lam * torch.eye(G_theta.shape[0], dtype=torch.float64,
                                        device=V.device)
    G_eta = torch.linalg.inv(G_theta)
    return eta, G_theta.float(), G_eta.float()
