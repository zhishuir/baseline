# -*- coding: utf-8 -*-
"""
训练脚本（论文创新一：基线生成方法）
====================================
三任务联合优化：重构损失(1.0) + 15类多分类损失(2.0, FocalLoss) + 二分类损失(1.5, FocalLoss)
数据-特征-梯度三层协同处理类别不均衡：加权采样 + Mixup + FocalLoss

已从原始实验脚本中移除：GPU 温度监控（与算法无关的硬件监控代码）、
Visualizer 可视化类（训练曲线/混淆矩阵等图片生成代码，不在工程代码走查范围内）。
模型结构定义已拆分到 model.py。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from tqdm import tqdm
import json
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================

CONFIG = {
    'data_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ids2018_minimal.csv'),

    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,

    'latent_dim': 512,  # ✅ 256→512，增大潜在空间
    'hidden_dims': [512, 512, 512, 512, 512, 512, 512],  # ✅ 统一512维度

    'num_transformer_layers': 3,  # ✅ 2→3层
    'num_attention_heads': 8,
    'transformer_dim': 512,  
    'transformer_ff_dim': 2048, 
    'batch_size': 256,  
    'num_epochs': 800,  # 
    'learning_rate': 0.0003,
    'weight_decay': 1e-4,
    'patience': 999, 

    'lambda_recon': 1.0,
    'lambda_classify': 2.0,  
    'lambda_binary': 1.5,
    'lambda_contrast': 0.0,

    'focal_alpha': 0.15,
    'focal_gamma': 4.0,

    'mixup_alpha': 0.2,  
    'use_targeted_smote': False,     

    'threshold_sigma': 1.5,
    'threshold_window': 1000,


    'model_dir': 'models_ultimate',
    'results_dir': 'results_ultimate',
}
# 创建目录
os.makedirs(CONFIG['model_dir'], exist_ok=True)
os.makedirs(CONFIG['results_dir'], exist_ok=True)

# ==================== 模型结构 ====================
# UltimateIDSNetwork（论文创新一：多任务 Transformer 增强自编码器）
# 残差编码器(30->512) -> 3层8头Transformer(全局依赖建模) -> 512维潜在基线表征
# -> 残差解码器(512->30) + 15类多分类头 + 二分类头

class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, d_model, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

# ==================== 残差块 ====================

class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
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
        out += residual
        out = F.gelu(out)
        return out

# ==================== 终极网络架构 ====================

class UltimateIDSNetwork(nn.Module):
    """终极入侵检测网络 - 集成Transformer"""

    def __init__(self, input_dim=30, latent_dim=256, num_classes=15,
                 hidden_dims=[512, 512, 384, 256, 384, 512, 512],
                 num_transformer_layers=3, num_attention_heads=8,
                 transformer_ff_dim=1024):
        super(UltimateIDSNetwork, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # ========== 编码器 ==========

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU()
        )

        # 残差编码器 - 渐进式降维
        encoder_dims = [hidden_dims[0], hidden_dims[1], hidden_dims[2], hidden_dims[3]]
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            self.encoder_blocks.append(ResidualBlock(encoder_dims[i]))
            if encoder_dims[i] != encoder_dims[i+1]:
                self.encoder_blocks.append(nn.Sequential(
                    nn.Linear(encoder_dims[i], encoder_dims[i+1]),
                    nn.BatchNorm1d(encoder_dims[i+1]),
                    nn.GELU()
                ))
        self.encoder_blocks.append(ResidualBlock(encoder_dims[-1]))

        # Transformer编码器（在最小维度上操作）
        self.transformer_encoder = TransformerEncoder(
            d_model=encoder_dims[-1],  # 使用编码器最后的维度
            num_heads=num_attention_heads,
            ff_dim=transformer_ff_dim,
            num_layers=num_transformer_layers
        )

        # 编码到潜在空间
        self.to_latent = nn.Sequential(
            nn.Linear(encoder_dims[-1], latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU()
        )

        # ========== 解码器 ==========

        # 从潜在空间解码
        decoder_dims = [hidden_dims[3], hidden_dims[4], hidden_dims[5], hidden_dims[6]]
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, decoder_dims[0]),
            nn.BatchNorm1d(decoder_dims[0]),
            nn.GELU()
        )

        # 残差解码器 - 渐进式升维
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            self.decoder_blocks.append(ResidualBlock(decoder_dims[i]))
            if decoder_dims[i] != decoder_dims[i+1]:
                self.decoder_blocks.append(nn.Sequential(
                    nn.Linear(decoder_dims[i], decoder_dims[i+1]),
                    nn.BatchNorm1d(decoder_dims[i+1]),
                    nn.GELU()
                ))
        self.decoder_blocks.append(ResidualBlock(decoder_dims[-1]))

        # 输出重构
        self.output_proj = nn.Linear(decoder_dims[-1], input_dim)

        # ========== 分类器 ==========

        # 多分类器（14类攻击+1类正常）
        self.classifier = nn.Sequential(
            ResidualBlock(latent_dim),
            ResidualBlock(latent_dim),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes)
        )

        # 二分类器（正常/攻击）
        self.binary_classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def encode(self, x):
        """编码"""
        # 输入投影
        h = self.input_proj(x)

        # 残差编码（包含自动升降维）
        for block in self.encoder_blocks:
            h = block(h)

        # Transformer编码（添加序列维度）
        h_seq = h.unsqueeze(1)  # [batch, 1, dim]
        h_trans = self.transformer_encoder(h_seq)
        h = h_trans.squeeze(1)  # [batch, dim]

        # 到潜在空间
        z = self.to_latent(h)

        return z

    def decode(self, z):
        """解码"""
        h = self.from_latent(z)

        for block in self.decoder_blocks:
            h = block(h)

        recon = self.output_proj(h)
        return recon

    def forward(self, x):
        """前向传播"""
        # 编码
        z = self.encode(x)

        # 解码
        recon = self.decode(z)

        # 分类
        logits_multi = self.classifier(z)
        logits_binary = self.binary_classifier(z)

        return recon, z, logits_multi, logits_binary

# ==================== 数据集 ====================

class IDSDataset(Dataset):
    """入侵检测数据集"""
    
    def __init__(self, features, labels, labels_binary):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.labels_binary = torch.LongTensor(labels_binary)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.labels_binary[idx]

# ==================== 损失函数 ====================

class FocalLoss(nn.Module):
    """Focal Loss with Label Smoothing"""
    
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, smoothing=0.05):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        
        # Label smoothing
        targets_one_hot = F.one_hot(targets, n_classes).float()
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = -(targets_smooth * log_probs).sum(dim=1)
        
        # 应用类别权重
        if self.weight is not None:
            ce_loss = ce_loss * self.weight[targets]
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class SupervisedContrastiveLoss(nn.Module):
    """监督对比学习损失"""
    
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        # 归一化
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除对角线
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算损失
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-6)
        loss = -mean_log_prob_pos.mean()
        
        return loss

def mixup_data(x, y, alpha=0.2):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

# ==================== 自适应阈值模块 ====================

class AdaptiveThresholdModule:
    """自适应阈值模块"""
    
    def __init__(self, window_size=1000, sigma_multiplier=2.0):
        self.window_size = window_size
        self.sigma_multiplier = sigma_multiplier
        self.normal_scores = []
        self.threshold = 0.5
    
    def update(self, score, is_normal):
        """更新阈值"""
        if is_normal:
            self.normal_scores.append(float(score))
            
            if len(self.normal_scores) > self.window_size:
                self.normal_scores.pop(0)
            
            if len(self.normal_scores) >= 100:
                mean = np.mean(self.normal_scores)
                std = np.std(self.normal_scores)
                self.threshold = mean + self.sigma_multiplier * std
    
    def get_threshold(self):
        return self.threshold

# ==================== 训练器 ====================

class UltimateTrainer:
    """终极训练器"""

    def __init__(self, model, train_loader, val_loader, test_loader, config, device,
                 class_weights_tensor=None):   # ✅ 新增参数
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # 优化器 - 分层学习率
        # ✅ Transformer层用更小学习率，分类器层用正常学习率
        transformer_params = list(model.transformer_encoder.parameters())
        transformer_ids = set(id(p) for p in transformer_params)
        other_params = [p for p in model.parameters() if id(p) not in transformer_ids]

        self.optimizer = torch.optim.AdamW([
            {'params': transformer_params, 'lr': config['learning_rate'] * 0.3},
            {'params': other_params,       'lr': config['learning_rate']},
        ], weight_decay=config['weight_decay'])

        # 学习率调度 - Warmup + CosineAnnealing
        # ✅ 前10个epoch线性预热，之后余弦衰减
        def lr_lambda(epoch):
            warmup_epochs = 10
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            progress = (epoch - warmup_epochs) / (config['num_epochs'] - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

        # 损失函数 - 带类别权重的FocalLoss
        # ✅ 传入类别权重
        if class_weights_tensor is not None:
            cw_multi = class_weights_tensor.to(device)
        else:
            cw_multi = None
        
        self.focal_loss = FocalLoss(
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma'],
            weight=cw_multi
        )
        
        # 二分类FocalLoss（单独用2类权重，不传类别权重）
        self.focal_loss_binary = FocalLoss(
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma'],
            weight=None          # ✅ 二分类不用15类权重
        )
        self.contrast_loss = SupervisedContrastiveLoss()
        self.recon_loss = nn.MSELoss()

        # 自适应阈值
        self.adaptive_threshold = AdaptiveThresholdModule(
            window_size=config['threshold_window'],
            sigma_multiplier=config['threshold_sigma']
        )

        # 训练历史
        self.history = {
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'train_acc_binary': [], 'val_acc_binary': [], 'test_acc_binary': [],
            'train_f1_binary': [], 'val_f1_binary': [], 'test_f1_binary': [],
            'train_acc_multi': [], 'val_acc_multi': [], 'test_acc_multi': [],
            'train_f1_multi': [], 'val_f1_multi': [], 'test_f1_multi': [],
            'learning_rates': [], 'adaptive_thresholds': [],
        }

        self.best_f1 = 0
        self.patience_counter = 0
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds_binary = []
        all_labels_binary = []
        all_preds_multi = []
        all_labels_multi = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for features, labels_multi, labels_binary in pbar:
            features = features.to(self.device)
            labels_multi = labels_multi.to(self.device)
            labels_binary = labels_binary.to(self.device)

            # Mixup - 对小样本类别更频繁使用
            mixup_prob = 0.5
            if labels_multi.min() > 0:
                unique_labels = torch.unique(labels_multi)
                for label in unique_labels:
                    if label > 0:
                        mixup_prob = 0.8
                        break

            if self.config['mixup_alpha'] > 0 and np.random.random() < mixup_prob:
                features, labels_a, labels_b, lam = mixup_data(
                    features, labels_multi, self.config['mixup_alpha']
                )

                # 前向传播
                recon, z, logits_multi, logits_binary = self.model(features)

                # 混合损失
                loss_cls_a = self.focal_loss(logits_multi, labels_a)
                loss_cls_b = self.focal_loss(logits_multi, labels_b)
                loss_cls = lam * loss_cls_a + (1 - lam) * loss_cls_b
            else:
                # 前向传播
                recon, z, logits_multi, logits_binary = self.model(features)
                loss_cls = self.focal_loss(logits_multi, labels_multi)
            
            # 重构损失
            loss_recon = self.recon_loss(recon, features)
            
            # 对比学习损失
            loss_contrast = self.contrast_loss(z, labels_binary)
            
            # 二分类损失
            loss_binary = self.focal_loss_binary(logits_binary, labels_binary)
            
            # 总损失
            loss = (
                self.config['lambda_recon'] * loss_recon +
                self.config['lambda_classify'] * loss_cls +
                self.config['lambda_binary'] * loss_binary +
                self.config['lambda_contrast'] * loss_contrast
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += float(loss.item())
            
            # 预测
            preds_binary = torch.argmax(logits_binary, dim=1)
            preds_multi = torch.argmax(logits_multi, dim=1)
            
            all_preds_binary.extend(preds_binary.cpu().numpy())
            all_labels_binary.extend(labels_binary.cpu().numpy())
            all_preds_multi.extend(preds_multi.cpu().numpy())
            all_labels_multi.extend(labels_multi.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算指标
        avg_loss = total_loss / len(self.train_loader)
        acc_binary = accuracy_score(all_labels_binary, all_preds_binary)
        f1_binary = f1_score(all_labels_binary, all_preds_binary, average='weighted')
        acc_multi = accuracy_score(all_labels_multi, all_preds_multi)
        f1_multi = f1_score(all_labels_multi, all_preds_multi, average='weighted')
        
        return avg_loss, acc_binary, f1_binary, acc_multi, f1_multi
    
    @torch.no_grad()
    def evaluate(self, loader, update_threshold=False):
        """评估"""
        self.model.eval()
        total_loss = 0
        all_preds_binary = []
        all_labels_binary = []
        all_preds_multi = []
        all_labels_multi = []
        all_probs_binary = []
        all_latents = []
        
        for features, labels_multi, labels_binary in tqdm(loader, desc='Evaluating'):
            features = features.to(self.device)
            labels_multi = labels_multi.to(self.device)
            labels_binary = labels_binary.to(self.device)
            
            # 前向传播
            recon, z, logits_multi, logits_binary = self.model(features)
            
            # 损失
            loss_recon = self.recon_loss(recon, features)
            loss_cls = self.focal_loss(logits_multi, labels_multi)
            loss_contrast = self.contrast_loss(z, labels_binary)
            loss_binary = self.focal_loss_binary(logits_binary, labels_binary)
            
            loss = (
                self.config['lambda_recon'] * loss_recon +
                self.config['lambda_classify'] * loss_cls +
                self.config['lambda_binary'] * loss_binary +
                self.config['lambda_contrast'] * loss_contrast
            )
            
            total_loss += float(loss.item())
            
            # 预测
            probs_binary = F.softmax(logits_binary, dim=1)
            preds_binary = torch.argmax(logits_binary, dim=1)
            preds_multi = torch.argmax(logits_multi, dim=1)
            
            all_probs_binary.extend(probs_binary[:, 1].cpu().numpy())  # 攻击概率
            all_preds_binary.extend(preds_binary.cpu().numpy())
            all_labels_binary.extend(labels_binary.cpu().numpy())
            all_preds_multi.extend(preds_multi.cpu().numpy())
            all_labels_multi.extend(labels_multi.cpu().numpy())
            all_latents.extend(z.cpu().numpy())
            
            # 更新自适应阈值
            if update_threshold:
                for i in range(len(features)):
                    recon_error = torch.mean((recon[i] - features[i]) ** 2).item()
                    is_normal = (labels_binary[i].item() == 0)
                    self.adaptive_threshold.update(recon_error, is_normal)
        
        # 计算指标
        avg_loss = total_loss / len(loader)
        acc_binary = accuracy_score(all_labels_binary, all_preds_binary)
        f1_binary = f1_score(all_labels_binary, all_preds_binary, average='weighted')
        acc_multi = accuracy_score(all_labels_multi, all_preds_multi)
        f1_multi = f1_score(all_labels_multi, all_preds_multi, average='weighted')
        
        return {
            'loss': avg_loss,
            'acc_binary': acc_binary,
            'f1_binary': f1_binary,
            'acc_multi': acc_multi,
            'f1_multi': f1_multi,
            'preds_binary': all_preds_binary,
            'labels_binary': all_labels_binary,
            'preds_multi': all_preds_multi,
            'labels_multi': all_labels_multi,
            'probs_binary': all_probs_binary,
            'latents': np.array(all_latents),
        }
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*80)
        print("开始训练")
        print("="*80)
        
        for epoch in range(self.config['num_epochs']):
            # 训练
            train_loss, train_acc_bin, train_f1_bin, train_acc_multi, train_f1_multi = self.train_epoch()
            
            # 验证
            val_results = self.evaluate(self.val_loader, update_threshold=True)
            
            # 测试
            test_results = self.evaluate(self.test_loader)
            
            # 学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            
            # 记录历史
            self.history['train_loss'].append(float(train_loss))
            self.history['val_loss'].append(float(val_results['loss']))
            self.history['test_loss'].append(float(test_results['loss']))
            self.history['train_acc_binary'].append(float(train_acc_bin))
            self.history['val_acc_binary'].append(float(val_results['acc_binary']))
            self.history['test_acc_binary'].append(float(test_results['acc_binary']))
            self.history['train_f1_binary'].append(float(train_f1_bin))
            self.history['val_f1_binary'].append(float(val_results['f1_binary']))
            self.history['test_f1_binary'].append(float(test_results['f1_binary']))
            self.history['train_acc_multi'].append(float(train_acc_multi))
            self.history['val_acc_multi'].append(float(val_results['acc_multi']))
            self.history['test_acc_multi'].append(float(test_results['acc_multi']))
            self.history['train_f1_multi'].append(float(train_f1_multi))
            self.history['val_f1_multi'].append(float(val_results['f1_multi']))
            self.history['test_f1_multi'].append(float(test_results['f1_multi']))
            self.history['learning_rates'].append(float(current_lr))
            self.history['adaptive_thresholds'].append(float(self.adaptive_threshold.get_threshold()))
            
            # 打印结果
            print(f"\n📊 Epoch {epoch+1}/{self.config['num_epochs']}")
            print(f"   训练: Loss={train_loss:.4f}, Binary(Acc={train_acc_bin:.4f}, F1={train_f1_bin:.4f}), Multi(Acc={train_acc_multi:.4f}, F1={train_f1_multi:.4f})")
            print(f"   验证: Loss={val_results['loss']:.4f}, Binary(Acc={val_results['acc_binary']:.4f}, F1={val_results['f1_binary']:.4f}), Multi(Acc={val_results['acc_multi']:.4f}, F1={val_results['f1_multi']:.4f})")
            print(f"   测试: Loss={test_results['loss']:.4f}, Binary(Acc={test_results['acc_binary']:.4f}, F1={test_results['f1_binary']:.4f}), Multi(Acc={test_results['acc_multi']:.4f}, F1={test_results['f1_multi']:.4f})")
            print(f"   学习率: {current_lr:.6f}")
            print(f"   自适应阈值: {self.adaptive_threshold.get_threshold():.4f}")
            
            # 早停
            if val_results['f1_multi'] > self.best_f1:
                self.best_f1 = val_results['f1_multi']
                self.patience_counter = 0
                
                # 保存最佳模型
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_f1': self.best_f1,
                    'threshold': self.adaptive_threshold.get_threshold(),
                }, f"{self.config['model_dir']}/best_model.pth")
                
                print(f"   ✅ 新最佳模型! F1={self.best_f1:.4f}")
            else:
                self.patience_counter += 1
                print(f"   ⏳ 耐心值: {self.patience_counter}/{self.config['patience']}")
                
           #     if self.patience_counter >= self.config['patience']:
            #        print(f"\n⚠️  早停触发 (patience={self.config['patience']})")
             #       break
        
        # 保存历史
        with open(f"{self.config['model_dir']}/history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n✅ 训练完成！")
        
        return test_results

def main():
    """主函数"""
    
    print("="*80)

    print("="*80)
    print(f" 数据集: {CONFIG['data_path']}")
    print(f" 数据划分: 训练{CONFIG['train_ratio']*100:.0f}% | 验证{CONFIG['val_ratio']*100:.0f}% | 测试{CONFIG['test_ratio']*100:.0f}%")
    print(f"  Transformer层数: {CONFIG['num_transformer_layers']}")
    print(f"  注意力头数: {CONFIG['num_attention_heads']}")
    print(f" 批大小: {CONFIG['batch_size']}")
    print(f" Epoch数: {CONFIG['num_epochs']}")
    print("="*80)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    if device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # ========== 步骤1: 加载数据 ==========
    print("\n" + "="*80)
    print("步骤1: 加载和预处理数据")
    print("="*80)
    
    df = pd.read_csv(CONFIG['data_path'])
    print(f" 加载数据: {len(df)}条记录, {len(df.columns)}个特征")
    print(f"   类别分布:\n{df['Label'].value_counts()}")
    
    # 特征和标签
    X = df.drop('Label', axis=1).values
    y_str = df['Label'].values
    
    # 标签编码
    label_encoder = LabelEncoder()
    y_multi = label_encoder.fit_transform(y_str)
    y_binary = (y_str != 'BENIGN').astype(int)
    
    print(f"标签编码完成")
    print(f"   类别: {label_encoder.classes_}")
    print(f"   正常/攻击分布: {np.bincount(y_binary)}")
    
    # 数据归一化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f" 数据归一化完成")
    
    # 数据划分：训练/验证/测试
    X_temp, X_test, y_multi_temp, y_multi_test, y_binary_temp, y_binary_test = train_test_split(
        X_scaled, y_multi, y_binary,
        test_size=CONFIG['test_ratio'],
        random_state=42,
        stratify=y_multi
    )
    
    val_size_adjusted = CONFIG['val_ratio'] / (CONFIG['train_ratio'] + CONFIG['val_ratio'])
    X_train, X_val, y_multi_train, y_multi_val, y_binary_train, y_binary_val = train_test_split(
        X_temp, y_multi_temp, y_binary_temp,
        test_size=val_size_adjusted,
        random_state=42,
        stratify=y_multi_temp
    )
    
    print(f" 数据划分完成")
    print(f"   训练集: {len(X_train)}条")
    print(f"   验证集: {len(X_val)}条")
    print(f"   测试集: {len(X_test)}条")


# ========== 针对性SMOTE（只对低召回率大类）==========
    if CONFIG.get('use_targeted_smote', False):
        from imblearn.over_sampling import SMOTE
        
        print("\n🎯 针对性SMOTE（只扩充低召回率类别）")
        
        # 找到需要扩充的类别索引
        target_indices = []
        for cls_name in CONFIG['targeted_smote_classes']:
            idx = np.where(label_encoder.classes_ == cls_name)[0]
            if len(idx) > 0:
                target_indices.append(int(idx[0]))
        
        if target_indices:
            counts_before = np.bincount(y_multi_train)
            
            # 只对这些类别进行SMOTE
            sampling_strategy = {}
            for idx in target_indices:
                if counts_before[idx] < CONFIG['targeted_smote_target']:
                    sampling_strategy[idx] = CONFIG['targeted_smote_target']
            
            if sampling_strategy:
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=42,
                    k_neighbors=5
                )
                X_train, y_multi_train = smote.fit_resample(X_train, y_multi_train)
                y_binary_train = (y_multi_train != 0).astype(int)
                
                counts_after = np.bincount(y_multi_train)
                print(f"   ✅ SMOTE完成，扩充类别：")
                for idx in target_indices:
                    cls = label_encoder.classes_[idx]
                    print(f"      {cls:<25}: {counts_before[idx]:>5} → {counts_after[idx]:>5}")
                print(f"   总训练样本: {len(y_multi_train)}")
    else:
        print("\n✅ 使用原始数据 + 加权采样")
    
    # 创建数据集
    train_dataset = IDSDataset(X_train, y_multi_train, y_binary_train)
    val_dataset   = IDSDataset(X_val,   y_multi_val,   y_binary_val)
    test_dataset  = IDSDataset(X_test,  y_multi_test,  y_binary_test)
    
    # 计算类别权重（用于FocalLoss）
    class_counts = np.bincount(y_multi_train)
    class_weights_np = np.zeros_like(class_counts, dtype=float)
    for i, count in enumerate(class_counts):
            if count < 200:  # 小样本类别
                class_weights_np[i] = 1.0 / np.sqrt(count)
            else:  # 大样本类别
                class_weights_np[i] = 1.0 / np.log1p(count)
    class_weights_np = class_weights_np / class_weights_np.sum() * len(class_weights_np)
    class_weights_tensor = torch.FloatTensor(class_weights_np)
    
    print(f"\n   各类样本数统计（前5）:")
    for i in range(min(5, len(label_encoder.classes_))):
        print(f"   {label_encoder.classes_[i]:<25}: {class_counts[i]:>5}")
    print(f"   BENIGN权重: {class_weights_np[0]:.3f}")
    
    # 采样权重（用于WeightedRandomSampler）
    sample_weights = class_weights_np[y_multi_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=sampler,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ DataLoader创建完成")

    # ========== 步骤2: 创建模型 ==========
    print("\n" + "="*80)
    print("步骤2: 创建终极网络模型")
    print("="*80)
    
    model = UltimateIDSNetwork(
        input_dim=X_train.shape[1],
        latent_dim=CONFIG['latent_dim'],
        num_classes=len(label_encoder.classes_),
        hidden_dims=CONFIG['hidden_dims'],
        num_transformer_layers=CONFIG['num_transformer_layers'],
        num_attention_heads=CONFIG['num_attention_heads'],
        transformer_ff_dim=CONFIG['transformer_ff_dim']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型创建完成")
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # ========== 步骤3: 训练 ==========
    print("\n" + "="*80)
    print("步骤3: 开始训练")
    print("="*80)


    trainer = UltimateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=CONFIG,
        device=device,
        class_weights_tensor=class_weights_tensor,
    )
    
    test_results = trainer.train()
    
    # ========== 步骤4: 保存 ==========
    print("\n" + "="*80)
    print("步骤4: 保存模型和配置")
    print("="*80)
    
    # 保存scaler和label_encoder
    with open(f"{CONFIG['model_dir']}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print("保存scaler.pkl")
    
    with open(f"{CONFIG['model_dir']}/label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✅ 保存label_encoder.pkl")
    
    # 保存配置
    with open(f"{CONFIG['model_dir']}/config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print("保存config.json")
    

    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print(f"模型保存: {CONFIG['model_dir']}/")
    print(f"结果保存: {CONFIG['results_dir']}/")
    print(f"最终性能:")
    print(f"二分类: Acc={test_results['acc_binary']:.4f}, F1={test_results['f1_binary']:.4f}")
    print(f"多分类: Acc={test_results['acc_multi']:.4f}, F1={test_results['f1_multi']:.4f}")
    print("="*80)

if __name__ == '__main__':
    main()
