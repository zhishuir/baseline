# -*- coding: utf-8 -*-
"""
论文成果 Web 演示系统（答辩专用）
====================================
只展示论文自己的方法：自编码器 + Transformer 基线生成 + k-Sigma 动态阈值判决。
不含企业需求驱动的多源融合演示（那部分见 交付包/code/multi_source_demo）。

启动: python web_demo.py
浏览器: http://localhost:8090
"""

import os, sys, json, pickle
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 模型结构：UltimateIDSNetwork（论文创新一：多任务 Transformer 增强自编码器）
# 残差编码器(30->512) -> 3层8头Transformer(全局依赖建模) -> 512维潜在基线表征
# -> 残差解码器(512->30) + 15类多分类头 + 二分类头
# ============================================================
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
                dropout=dropout, activation='gelu', batch_first=True
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim, momentum=0.01)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim, momentum=0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return F.gelu(out + residual)


class UltimateIDSNetwork(nn.Module):
    """终极入侵检测网络 - 集成Transformer"""
    def __init__(self, input_dim=30, latent_dim=256, num_classes=15,
                 hidden_dims=[512, 512, 384, 256, 384, 512, 512],
                 num_transformer_layers=3, num_attention_heads=8,
                 transformer_ff_dim=1024):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # ===== 编码器 =====
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]), nn.GELU()
        )
        encoder_dims = [hidden_dims[0], hidden_dims[1], hidden_dims[2], hidden_dims[3]]
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            self.encoder_blocks.append(ResidualBlock(encoder_dims[i]))
            if encoder_dims[i] != encoder_dims[i + 1]:
                self.encoder_blocks.append(nn.Sequential(
                    nn.Linear(encoder_dims[i], encoder_dims[i + 1]),
                    nn.BatchNorm1d(encoder_dims[i + 1]), nn.GELU()))
        self.encoder_blocks.append(ResidualBlock(encoder_dims[-1]))

        self.transformer_encoder = TransformerEncoder(
            d_model=encoder_dims[-1], num_heads=num_attention_heads,
            ff_dim=transformer_ff_dim, num_layers=num_transformer_layers
        )

        self.to_latent = nn.Sequential(
            nn.Linear(encoder_dims[-1], latent_dim),
            nn.BatchNorm1d(latent_dim), nn.GELU()
        )

        # ===== 解码器 =====
        decoder_dims = [hidden_dims[3], hidden_dims[4], hidden_dims[5], hidden_dims[6]]
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, decoder_dims[0]),
            nn.BatchNorm1d(decoder_dims[0]), nn.GELU()
        )
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            self.decoder_blocks.append(ResidualBlock(decoder_dims[i]))
            if decoder_dims[i] != decoder_dims[i + 1]:
                self.decoder_blocks.append(nn.Sequential(
                    nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                    nn.BatchNorm1d(decoder_dims[i + 1]), nn.GELU()))
        self.decoder_blocks.append(ResidualBlock(decoder_dims[-1]))
        self.output_proj = nn.Linear(decoder_dims[-1], input_dim)

        # ===== 分类器 =====
        self.classifier = nn.Sequential(
            ResidualBlock(latent_dim), ResidualBlock(latent_dim),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes)
        )
        self.binary_classifier = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def encode(self, x):
        h = self.input_proj(x)
        for block in self.encoder_blocks:
            h = block(h)
        h_seq = h.unsqueeze(1)
        h_trans = self.transformer_encoder(h_seq)
        h = h_trans.squeeze(1)
        return self.to_latent(h)

    def decode(self, z):
        h = self.from_latent(z)
        for block in self.decoder_blocks:
            h = block(h)
        return self.output_proj(h)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        logits_multi = self.classifier(z)
        logits_binary = self.binary_classifier(z)
        return recon, z, logits_multi, logits_binary

# ============================================================
# 路径（相对于本文件，整个 答辩代码/ 文件夹可整体移动）
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_ultimate")
DATA_PATH = os.path.join(BASE_DIR, "data", "ids2018_minimal.csv")

CLASS_NAMES = [
    "BENIGN", "Bot", "Brute Force -Web", "Brute Force -XSS",
    "DDOS attack-HOIC", "DDOS attack-LOIC-UDP", "DoS attacks-GoldenEye",
    "DoS attacks-Hulk", "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris",
    "FTP-BruteForce", "Infilteration", "Label", "SQL Injection",
    "SSH-Bruteforce",
]

# ============================================================
# k-Sigma 动态阈值（与论文创新二一致：Welford增量 + 攻击样本隔离）
# ============================================================
class WelfordStats:
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def update(self, x):
        self.n += 1; d1 = x - self.mean
        self.mean += d1 / self.n; d2 = x - self.mean; self.M2 += d1 * d2
    def remove(self, x):
        if self.n <= 1: return
        nm = self.n - 1; new_m = (self.n * self.mean - x) / nm
        self.M2 -= (x - self.mean) * (x - new_m); self.mean = new_m; self.n = nm
    @property
    def std(self):
        return (self.M2 / max(1, self.n - 1)) ** 0.5 if self.n >= 2 else 0.0


class AdaptiveThreshold:
    """k-Sigma 滑动窗口 + 攻击样本隔离机制"""
    def __init__(self, window=1000, k=1.5, offline=0.05):
        self.W = window; self.k = k; self.offline = offline
        self.queue = []; self.stats = WelfordStats(); self.theta = offline

    def update(self, score, is_normal):
        if not is_normal:
            return  # 攻击样本隔离：不入队，不污染统计量
        if len(self.queue) >= self.W:
            self.stats.remove(self.queue.pop(0))
        self.queue.append(score); self.stats.update(score)
        n = len(self.queue)
        if n >= 100:
            online = self.stats.mean + self.k * self.stats.std
            if n < self.W:
                a = (n - 100) / (self.W - 100)
                self.theta = (1 - a) * self.offline + a * online
            else:
                self.theta = online

    def get(self): return self.theta


# ============================================================
# 论文报告指标（来自 classification_report.txt，固定值，非现场计算）
# ============================================================
PAPER_METRICS = {
    "binary_tpr": 0.876, "binary_fpr": 0.087,
    "multi_accuracy": 0.840, "multi_weighted_f1": 0.838, "multi_macro_f1": 0.704,
}
PER_CLASS_F1 = {
    "BENIGN": 0.921, "Bot": 0.999, "Brute Force -Web": 0.773,
    "Brute Force -XSS": 0.467, "DDOS attack-HOIC": 0.544,
    "DDOS attack-LOIC-UDP": 1.000, "DoS attacks-GoldenEye": 0.981,
    "DoS attacks-Hulk": 0.654, "DoS attacks-SlowHTTPTest": 0.713,
    "DoS attacks-Slowloris": 0.990, "FTP-BruteForce": 0.534,
    "Infilteration": 0.426, "Label": 0.000,
    "SQL Injection": 0.615, "SSH-Bruteforce": 0.948,
}
PER_CLASS_PRECISION = {
    "BENIGN": 0.908, "Bot": 1.000, "Brute Force -Web": 0.793,
    "Brute Force -XSS": 1.000, "DDOS attack-HOIC": 0.741,
    "DDOS attack-LOIC-UDP": 1.000, "DoS attacks-GoldenEye": 0.980,
    "DoS attacks-Hulk": 0.552, "DoS attacks-SlowHTTPTest": 0.608,
    "DoS attacks-Slowloris": 1.000, "FTP-BruteForce": 1.000,
    "Infilteration": 0.428, "Label": 0.000,
    "SQL Injection": 1.000, "SSH-Bruteforce": 0.961,
}
PER_CLASS_RECALL = {
    "BENIGN": 0.935, "Bot": 0.998, "Brute Force -Web": 0.754,
    "Brute Force -XSS": 0.304, "DDOS attack-HOIC": 0.430,
    "DDOS attack-LOIC-UDP": 1.000, "DoS attacks-GoldenEye": 0.982,
    "DoS attacks-Hulk": 0.800, "DoS attacks-SlowHTTPTest": 0.860,
    "DoS attacks-Slowloris": 0.980, "FTP-BruteForce": 0.364,
    "Infilteration": 0.424, "Label": 0.000,
    "SQL Injection": 0.444, "SSH-Bruteforce": 0.936,
}
PER_CLASS_SUPPORT = {
    "BENIGN": 5000, "Bot": 500, "Brute Force -Web": 61,
    "Brute Force -XSS": 23, "DDOS attack-HOIC": 500,
    "DDOS attack-LOIC-UDP": 50, "DoS attacks-GoldenEye": 500,
    "DoS attacks-Hulk": 500, "DoS attacks-SlowHTTPTest": 500,
    "DoS attacks-Slowloris": 301, "FTP-BruteForce": 500,
    "Infilteration": 500, "Label": 6, "SQL Injection": 9,
    "SSH-Bruteforce": 500,
}

# ============================================================
# 全局状态
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None; SCALER = None; LABEL_ENCODER = None
X_TEST = None; Y_TEST = None; Y_BIN = None

def load_all():
    global MODEL, SCALER, LABEL_ENCODER, X_TEST, Y_TEST, Y_BIN
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        SCALER = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        LABEL_ENCODER = pickle.load(f)
    MODEL = UltimateIDSNetwork(
        input_dim=30, latent_dim=512, num_classes=15,
        hidden_dims=[512] * 7, num_transformer_layers=3,
        num_attention_heads=8, transformer_ff_dim=2048).to(DEVICE)
    ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pth"),
                      map_location=DEVICE, weights_only=False)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.eval()

    df = pd.read_csv(DATA_PATH)
    lc = None
    for c in df.columns:
        if c.lower() in ("label", "class"): lc = c; break
    if lc is None: lc = df.columns[-1]
    X = df.drop(columns=[lc]).values[:, :30]
    y_multi = LABEL_ENCODER.transform(df[lc].values)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), min(2000, len(X)), replace=False)
    X_TEST = SCALER.transform(X[idx]); Y_TEST = y_multi[idx]
    Y_BIN = (Y_TEST != 0).astype(int)
    params = sum(p.numel() for p in MODEL.parameters())
    print(f"[OK] Model: {params:,} params, Device={DEVICE}, Demo samples={len(X_TEST)}")


def analyze_sample(idx):
    """论文公式: 异常得分 s = 0.7*重构误差 + 0.3*攻击概率"""
    x = torch.FloatTensor(X_TEST[idx:idx+1]).to(DEVICE)
    with torch.no_grad():
        recon, _, logits_m, logits_b = MODEL(x)
        recon_err = float(F.mse_loss(recon, x))
        prob_attack = float(F.softmax(logits_b, dim=1)[0, 1])
        score = 0.7 * min(recon_err, 1.0) + 0.3 * prob_attack

        probs_m = F.softmax(logits_m, dim=1)[0]
        all_probs = {CLASS_NAMES[i]: round(float(probs_m[i]), 4) for i in range(15)}
        pred_cls = int(probs_m.argmax())
        conf = float(probs_m[pred_cls])

    return {
        "idx": idx,
        "true_label": CLASS_NAMES[Y_TEST[idx]],
        "true_is_attack": bool(Y_BIN[idx]),
        "recon_error": round(min(recon_err, 1.0), 4),
        "attack_prob": round(prob_attack, 4),
        "anomaly_score": round(score, 4),
        "pred_class": CLASS_NAMES[pred_cls],
        "confidence": round(conf, 4),
        "all_probs": all_probs,
        "is_attack": prob_attack > 0.5,
        "is_correct": CLASS_NAMES[pred_cls] == CLASS_NAMES[Y_TEST[idx]],
    }


# ============================================================
# HTML
# ============================================================
HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>基于自编码器的网络安全攻击基线方法 · 论文成果演示</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Microsoft YaHei",sans-serif;background:#f5f5f5;color:#222}
.header{background:#24569D;color:#fff;padding:14px 24px}
.header h1{font-size:18px;font-weight:600}
.header p{font-size:12px;opacity:.7;margin-top:2px}
.tabs{display:flex;background:#fff;border-bottom:1px solid #ddd;padding:0 20px}
.tab{padding:10px 18px;cursor:pointer;font-size:13px;border-bottom:2px solid transparent;margin-bottom:-1px;color:#555}
.tab:hover{color:#24569D}
.tab.active{color:#24569D;border-bottom-color:#24569D;font-weight:bold}
.content{padding:16px 20px;max-width:1000px}
.card{background:#fff;border-radius:6px;padding:16px;margin-bottom:12px;border:1px solid #e8e8e8}
.card h3{font-size:14px;margin-bottom:10px;color:#24569D;border-bottom:1px solid #eee;padding-bottom:6px}
.row{display:flex;gap:12px;flex-wrap:wrap}
.col{flex:1;min-width:180px}
.metric{text-align:center;padding:10px;background:#f8f9fa;border-radius:4px}
.metric .val{font-size:22px;font-weight:bold;color:#24569D}
.metric .lbl{font-size:11px;color:#888;margin-top:2px}
.metric .sub{font-size:10px;color:#aaa}
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:#f0f0f0;padding:6px 8px;text-align:left;font-weight:600;border-bottom:2px solid #ddd}
td{padding:5px 8px;border-bottom:1px solid #eee}
.btn{padding:7px 18px;border:none;border-radius:4px;cursor:pointer;font-size:13px;background:#24569D;color:#fff}
.btn:hover{background:#1a3f75}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600}
.tag-atk{background:#ffebee;color:#c62828}
.tag-norm{background:#e8f5e9;color:#2e7d32}
.tag-warn{background:#fff3e0;color:#e65100}
.bar-wrap{background:#e0e0e0;border-radius:3px;height:14px;margin:2px 0;overflow:hidden}
.bar-fill{background:#24569D;height:100%;border-radius:3px;transition:width .3s}
.bar-fill.good{background:#2e7d32}
.bar-fill.mid{background:#f9a825}
.bar-fill.bad{background:#c62828}
.loading{text-align:center;padding:30px;color:#999}
.formula{background:#fffde7;padding:8px 12px;border-left:3px solid #f9a825;font-size:12px;margin:8px 0;font-family:Consolas,monospace}
input[type=range]{width:100%}
.note{font-size:11px;color:#888;margin-top:8px}
</style>
</head>
<body>
<div class="header">
  <h1>基于自编码器的网络安全攻击基线方法与实现 · 论文成果演示</h1>
  <p>自编码器 + Transformer（创新一） | k-Sigma 自适应阈值 + 攻击样本隔离（创新二） | 15类攻击识别</p>
</div>
<div class="tabs">
  <div class="tab active" onclick="switchTab('overview')">系统概览</div>
  <div class="tab" onclick="switchTab('detect')">单条检测</div>
  <div class="tab" onclick="switchTab('metrics')">论文指标</div>
</div>
<div class="content" id="content"></div>
<script>
const API = '/api';

async function api(p, d) {
  const r = await fetch(API + p, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d||{})});
  return r.json();
}

function switchTab(n) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  if (n === 'overview') loadOverview();
  else if (n === 'detect') loadDetect();
  else if (n === 'metrics') loadMetrics();
}

// ============ 概览 ============
async function loadOverview() {
  const s = await api('/stats');
  document.getElementById('content').innerHTML = `
    <div class="card">
      <h3>论文模型信息</h3>
      <div class="row">
        <div class="col"><div class="metric"><div class="val">${s.params}</div><div class="lbl">参数量(M)</div></div></div>
        <div class="col"><div class="metric"><div class="val">${s.device}</div><div class="lbl">运行设备</div></div></div>
        <div class="col"><div class="metric"><div class="val">15</div><div class="lbl">检测类别</div></div></div>
        <div class="col"><div class="metric"><div class="val">${s.samples}</div><div class="lbl">演示样本数</div></div></div>
        <div class="col"><div class="metric"><div class="val">512</div><div class="lbl">潜在空间维度</div></div></div>
      </div>
    </div>
    <div class="card">
      <h3>核心创新</h3>
      <p style="font-size:13px;line-height:1.8">
        <b>创新一:</b> 多任务 Transformer 增强自编码器。编码瓶颈嵌入3层Transformer（8头注意力），
        对30维流量特征建模全局依赖关系，输出512维潜在空间基线表征。
        三任务联合优化: 重构损失(1.0) + 15类分类损失(2.0) + 二分类损失(1.5)。
      </p>
      <p style="font-size:13px;line-height:1.8;margin-top:8px">
        <b>创新二:</b> k-Sigma滑动窗口自适应阈值 + 攻击样本隔离机制。
        Welford增量算法O(1)维护滑动窗口均值方差，动态计算异常阈值 θ=μ+1.5σ。
        攻击样本不进入滑动窗口，从根源切断漂移正反馈，防止基线被污染。
      </p>
    </div>
    <div class="card">
      <h3>异常检测公式（论文公式）</h3>
      <div class="formula">
        s = 0.7 * min(MSE(recon, x), 1.0) + 0.3 * P_attack<br>
        阈值: theta = mu_window + 1.5 * sigma_window<br>
        判决: if s > theta -> 异常攻击; else -> 正常
      </div>
    </div>
    <p class="note">本页面仅展示论文自身方法（单一流量数据源），不含多源日志融合演示。</p>
  `;
}

// ============ 单条检测 ============
async function loadDetect() {
  document.getElementById('content').innerHTML = `
    <div class="card">
      <h3>选择测试样本</h3>
      <input type="range" id="idx" min="0" max="1999" value="0" oninput="document.getElementById('v').textContent=this.value">
      <p style="text-align:center;margin:6px 0">样本编号: <b id="v">0</b></p>
      <button class="btn" onclick="analyze()">分析此样本</button>
    </div>
    <div id="result"></div>
  `;
}

async function analyze() {
  const idx = parseInt(document.getElementById('idx').value);
  document.getElementById('result').innerHTML = '<div class="loading">分析中...</div>';
  const r = await api('/analyze', {idx});
  const tag = r.true_is_attack ? '<span class="tag tag-atk">攻击</span>' : '<span class="tag tag-norm">正常</span>';
  const cor = r.is_correct ? '<span class="tag tag-norm">分类正确</span>' : '<span class="tag tag-warn">具体类型判错</span>';
  const verdict = r.anomaly_score > r.threshold ? '<span class="tag tag-atk">异常</span>' : '<span class="tag tag-norm">正常</span>';

  const sorted = Object.entries(r.all_probs).sort((a,b) => b[1] - a[1]).slice(0, 5);

  document.getElementById('result').innerHTML = `
    <div class="card">
      <h3>检测结果</h3>
      <div class="row">
        <div class="col"><div class="metric"><div class="val">${r.anomaly_score.toFixed(4)}</div><div class="lbl">异常得分 s</div><div class="sub">公式: 0.7*recon + 0.3*prob</div></div></div>
        <div class="col"><div class="metric"><div class="val">${r.threshold.toFixed(4)}</div><div class="lbl">动态阈值 theta</div><div class="sub">k-Sigma自适应</div></div></div>
        <div class="col"><div class="metric"><div class="val">${verdict}</div><div class="lbl">二分类判决</div><div class="sub">s${r.anomaly_score > r.threshold ? '>' : '<='}theta</div></div></div>
        <div class="col"><div class="metric"><div class="val">${cor}</div><div class="lbl">15类判别</div></div></div>
      </div>
    </div>
    <div class="card">
      <h3>详细数值</h3>
      <table>
        <tr><th>指标</th><th>数值</th><th>说明</th></tr>
        <tr><td>真实标签</td><td><b>${r.true_label}</b> ${tag}</td><td>数据集标注</td></tr>
        <tr><td>预测类别</td><td><b>${r.pred_class}</b></td><td>多分类器输出</td></tr>
        <tr><td>置信度</td><td>${(r.confidence*100).toFixed(1)}%</td><td>softmax最大值</td></tr>
        <tr><td>重构误差</td><td>${r.recon_error.toFixed(4)}</td><td>自编码器 MSE(recon, x)</td></tr>
        <tr><td>攻击概率</td><td>${(r.attack_prob*100).toFixed(1)}%</td><td>二分类器 softmax[1]</td></tr>
        <tr><td>异常得分 s</td><td><b>${r.anomaly_score.toFixed(4)}</b></td><td>0.7*recon + 0.3*prob</td></tr>
        <tr><td>当前阈值 theta</td><td>${r.threshold.toFixed(4)}</td><td>mu + 1.5*sigma (滑动窗口)</td></tr>
      </table>
    </div>
    <div class="card">
      <h3>Top-5 分类概率</h3>
      ${sorted.map(([name, prob]) => {
        const cls = prob >= 0.9 ? 'good' : prob >= 0.7 ? 'mid' : 'bad';
        return `<div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px">
          <span style="width:140px;text-align:right">${name}</span>
          <div class="bar-wrap" style="flex:1"><div class="bar-fill ${cls}" style="width:${(prob*100).toFixed(0)}%"></div></div>
          <span style="width:50px">${(prob*100).toFixed(1)}%</span>
        </div>`;
      }).join('')}
    </div>
  `;
}

// ============ 论文指标 ============
async function loadMetrics() {
  const m = await api('/metrics');
  document.getElementById('content').innerHTML = `
    <div class="card">
      <h3>论文报告指标（固定值，来自 classification_report.txt）</h3>
      <div class="row">
        <div class="col"><div class="metric"><div class="val">${(m.binary_tpr*100).toFixed(1)}%</div><div class="lbl">攻击召回率 TPR</div></div></div>
        <div class="col"><div class="metric"><div class="val">${(m.binary_fpr*100).toFixed(1)}%</div><div class="lbl">误报率 FPR</div></div></div>
        <div class="col"><div class="metric"><div class="val">${(m.multi_accuracy*100).toFixed(1)}%</div><div class="lbl">多分类准确率</div></div></div>
        <div class="col"><div class="metric"><div class="val">${(m.multi_weighted_f1*100).toFixed(1)}%</div><div class="lbl">加权F1</div></div></div>
        <div class="col"><div class="metric"><div class="val">${(m.multi_macro_f1*100).toFixed(1)}%</div><div class="lbl">宏平均F1</div></div></div>
      </div>
      <p class="note">这几个数字是固定展示值，不是本页面现场计算的。如需现场重新计算完整测试集指标，用 demo_test.py 的阶段4（终端运行，约需数分钟）。</p>
    </div>
    <div class="card">
      <h3>15类分类结果明细</h3>
      <table>
        <tr><th>类别</th><th>精确率</th><th>召回率</th><th>F1</th><th>样本数</th><th>F1>=0.9</th></tr>
        ${m.classes.map(c => `<tr>
          <td>${c.name}</td>
          <td>${c.precision.toFixed(3)}</td>
          <td>${c.recall.toFixed(3)}</td>
          <td style="font-weight:bold;color:${c.f1>=0.9?'#2e7d32':c.f1>=0.7?'#f9a825':'#c62828'}">${c.f1.toFixed(3)}</td>
          <td>${c.support}</td>
          <td>${c.f1>=0.9 ? '是' : '否'}</td>
        </tr>`).join('')}
      </table>
      <p style="margin-top:10px;font-size:12px;color:#888">
        F1达标(>=0.9): ${m.pass_count}/15 类 |
        数据来源: CIC-IDS2018 测试集, 与论文 classification_report.txt 一致
      </p>
    </div>
  `;
}

loadOverview();
</script>
</body>
</html>"""


# ============================================================
# HTTP
# ============================================================
class Handler(BaseHTTPRequestHandler):
    def log_message(self, f, *a): pass
    def _send(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode("utf-8"))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}
        p = urlparse(self.path).path

        if p == "/api/stats":
            params = round(sum(t.numel() for t in MODEL.parameters()) / 1e6, 1)
            self._send({"params": params, "device": str(DEVICE),
                        "samples": len(X_TEST)})

        elif p == "/api/analyze":
            idx = body.get("idx", 0) % len(X_TEST)
            r = analyze_sample(idx)
            at = AdaptiveThreshold(window=1000, k=1.5, offline=0.05)
            for i in range(min(500, len(X_TEST))):
                if Y_BIN[i] == 0:
                    x = torch.FloatTensor(X_TEST[i:i+1]).to(DEVICE)
                    with torch.no_grad():
                        recon, _, _, _ = MODEL(x)
                        e = float(F.mse_loss(recon, x))
                    at.update(min(e, 1.0), True)
            r["threshold"] = round(at.get(), 4)
            self._send(r)

        elif p == "/api/metrics":
            classes = []
            pass_count = 0
            for name in CLASS_NAMES:
                f1 = PER_CLASS_F1.get(name, 0)
                if f1 >= 0.9: pass_count += 1
                classes.append({
                    "name": name,
                    "precision": PER_CLASS_PRECISION.get(name, 0),
                    "recall": PER_CLASS_RECALL.get(name, 0),
                    "f1": f1,
                    "support": PER_CLASS_SUPPORT.get(name, 0),
                })
            self._send({**PAPER_METRICS, "classes": classes, "pass_count": pass_count})

        else:
            self.send_error(404)


# ============================================================
# 启动
# ============================================================
def main():
    print("=" * 50)
    print("  基于自编码器的网络安全攻击基线方法 · 论文成果演示")
    print("=" * 50)
    print(f"  Device: {DEVICE}")
    print("  Loading model...")
    load_all()
    port = 8090
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    print(f"  http://localhost:{port}")
    print("  Ctrl+C to stop")
    print("=" * 50)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()

if __name__ == "__main__":
    main()
