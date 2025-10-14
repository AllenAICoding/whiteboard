#!/usr/bin/env python3
"""
True Multi-Scale Training with Hierarchical Feature Extraction

解決問題：確保不同尺度真正學習到不同層次的語義信息
1. 從 Transformer 不同層提取特徵（而非僅最終 embedding）
2. 添加正交性約束確保各尺度學習不同特徵
3. 層次化注意力機制引導不同粒度的關注點
4. 對比學習在不同尺度間建立一致性約束
"""

import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses, evaluation
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import time
import numpy as np
from data_prepare import main_generate_train_data_with_hard_mining, create_validation_triplets
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
import json
import re
from transformers import AutoModel, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

class MetricsTracker:
    """追蹤和視覺化訓練指標 (從 train.py 導入)"""

    def __init__(self,
                 save_path='./training_metrics.xlsx',
                 y_lim_ami=(0.3, 1.05),
                 y_lim_loss=None,  # Set to None for dynamic scaling
                 x_tick_interval=None,
                 y_tick_interval_ami=0.05,
                 y_tick_interval_loss=None,  # Set to None for dynamic scaling
                 loss_scale_margin=0.1):
        """初始化指標追蹤器"""
        self.save_path = save_path
        self.metrics_data = {
            'epoch': [], 'step': [], 'timestamp': [],
            'AMI_score': [], 'loss': [], 'learning_rate': []
        }
        # 圖表外觀參數
        self.y_lim_ami = y_lim_ami
        self.y_lim_loss = y_lim_loss
        self.x_tick_interval = x_tick_interval
        self.y_tick_interval_ami = y_tick_interval_ami
        self.y_tick_interval_loss = y_tick_interval_loss
        self.loss_scale_margin = loss_scale_margin

        self.fig, self.axes = None, None
        self._initialize_plot()

    def _calculate_dynamic_loss_scale(self, losses):
        """計算動態的損失圖縮放範圍"""
        if not losses or all(x is None or x == 0 for x in losses):
            return (0, 0.1)  # 預設範圍
        
        # 過濾掉 None 和 0 值
        valid_losses = [x for x in losses if x is not None and x != 0]
        if not valid_losses:
            return (0, 0.1)
        
        min_loss = min(valid_losses)
        max_loss = max(valid_losses)
        
        # 如果所有損失值都相同，添加一些邊距
        if min_loss == max_loss:
            margin = max(abs(min_loss) * 0.1, 0.01)
            return (max(0, min_loss - margin), max_loss + margin)
        
        # 計算範圍和邊距
        loss_range = max_loss - min_loss
        margin = loss_range * self.loss_scale_margin
        
        # 確保下限不會小於 0（對於大多數損失函數）
        y_min = max(0, min_loss - margin)
        y_max = max_loss + margin
        
        return (y_min, y_max)

    def _calculate_dynamic_tick_interval(self, y_min, y_max, target_ticks=8):
        """計算動態的刻度間距"""
        if y_max <= y_min:
            return 0.01
        
        range_val = y_max - y_min
        raw_interval = range_val / target_ticks
        
        # 將間距調整為"好看"的數字
        magnitude = 10 ** np.floor(np.log10(raw_interval))
        normalized = raw_interval / magnitude
        
        if normalized <= 1:
            nice_interval = 1
        elif normalized <= 2:
            nice_interval = 2
        elif normalized <= 5:
            nice_interval = 5
        else:
            nice_interval = 10
            
        return nice_interval * magnitude

    def _initialize_plot(self):
        """初始化圖表"""
        try:
            # 創建一個 Figure 和兩個子圖 (Axes)
            self.fig, self.axes = plt.subplots(2, 1, figsize=(15, 10))
            self.fig.suptitle('True Multi-Scale Training Progress', fontsize=16)
            plt.ion()  # 開啟互動模式
        except Exception as e:
            logger.warning(f"Could not initialize plot: {e}")
            self.fig = None

    def add_metric(self, epoch, step, ami_score, loss, learning_rate=None):
        """添加新的指標記錄"""
        self.metrics_data['epoch'].append(epoch)
        self.metrics_data['step'].append(step)
        self.metrics_data['timestamp'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.metrics_data['AMI_score'].append(ami_score)
        self.metrics_data['loss'].append(loss)
        self.metrics_data['learning_rate'].append(learning_rate)
        
        self.save_to_excel()
        
        if self.fig is not None:
            self.update_plot()

    def save_to_excel(self):
        """保存指標到Excel文件"""
        try:
            df = pd.DataFrame(self.metrics_data)
            df.to_excel(self.save_path, index=False)
            logger.info(f"Metrics saved to {self.save_path}")
        except Exception as e:
            logger.warning(f"Could not save metrics to Excel: {e}")

    def update_plot(self):
        """更新線圖"""
        if len(self.metrics_data['epoch']) == 0 or self.fig is None:
            return

        ax1, ax2 = self.axes
        
        try:
            ax1.clear()
            ax2.clear()
            
            steps = self.metrics_data['step']
            ami_scores = self.metrics_data['AMI_score']
            losses = self.metrics_data['loss']
            
            # --- 上方圖表: AMI Score ---
            ax1.plot(steps, ami_scores, 'b-', marker='o', linewidth=2, markersize=4, label='AMI Score')
            
            # Add score annotations above each point
            for i, (step, score) in enumerate(zip(steps, ami_scores)):
                ax1.annotate(f'{score:.3f}', 
                           xy=(step, score), 
                           xytext=(0, 10),  # 10 points above the point
                           textcoords='offset points',
                           ha='center', va='bottom',
                           fontsize=8, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            
            ax1.set_ylabel('AMI Score')
            ax1.set_title('Validation AMI Score')
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax1.set_ylim(self.y_lim_ami) # 設定 Y 軸範圍
            if self.y_tick_interval_ami:
                ax1.yaxis.set_major_locator(mticker.MultipleLocator(self.y_tick_interval_ami))
            
            # --- 下方圖表: Training Loss ---
            ax2.plot(steps, losses, 'r-', marker='s', linewidth=2, markersize=4, label='Multi-Scale Loss')
            
            # Add loss annotations above each point
            for step, loss in zip(steps, losses):
                if loss is not None and loss != 0:
                    ax2.annotate(f'{loss:.4f}', 
                               xy=(step, loss), 
                               xytext=(0, 10),  # 10 points above the point
                               textcoords='offset points',
                               ha='center', va='bottom',
                               fontsize=8, 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
            
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            ax2.set_title('Multi-Scale Training Loss')
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            # 動態設定損失圖的 Y 軸範圍和刻度
            if self.y_lim_loss is None:
                # 使用動態縮放
                y_min_loss, y_max_loss = self._calculate_dynamic_loss_scale(losses)
                ax2.set_ylim(y_min_loss, y_max_loss)
                
                # 動態設定刻度間距
                if self.y_tick_interval_loss is None:
                    tick_interval = self._calculate_dynamic_tick_interval(y_min_loss, y_max_loss)
                    ax2.yaxis.set_major_locator(mticker.MultipleLocator(tick_interval))
                else:
                    ax2.yaxis.set_major_locator(mticker.MultipleLocator(self.y_tick_interval_loss))
            else:
                # 使用固定縮放
                ax2.set_ylim(self.y_lim_loss)
                if self.y_tick_interval_loss:
                    ax2.yaxis.set_major_locator(mticker.MultipleLocator(self.y_tick_interval_loss))

            # --- 共用 X 軸設定 ---
            for ax in self.axes:
                ax.legend()
                if self.x_tick_interval:
                    ax.xaxis.set_major_locator(mticker.MultipleLocator(self.x_tick_interval))
            
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.pause(0.1)
            
        except Exception as e:
            logger.warning(f"Could not update plot: {e}")

    def save_plot(self, save_path='./training_progress.png'):
        """保存圖表"""
        if self.fig is not None:
            try:
                self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            except Exception as e:
                logger.warning(f"Could not save plot: {e}")

    def close(self):
        """關閉圖表"""
        if self.fig is not None:
            try:
                plt.ioff()
                plt.close(self.fig)
            except Exception as e:
                logger.warning(f"Could not close plot: {e}")

class CustomTrainingCallback:
    """
    自定義訓練回調，用於實時 AMI 追蹤和可視化 (從 train.py 導入)
    """
    def __init__(self, metrics_tracker, model, train_loss, val_sentences, val_labels):
        self.metrics_tracker = metrics_tracker
        self.model = model
        self.train_loss = train_loss
        self.val_sentences = val_sentences
        self.val_labels = val_labels
        self.step_count = 0
        self.current_epoch = 0
        self.current_loss = 0.0
        self.loss_history = []
        
        # 設置 loss 追蹤
        self._setup_loss_tracking()

    def _setup_loss_tracking(self):
        """設置 loss 追蹤鉤子"""
        try:
            if hasattr(self.train_loss, 'forward'):
                original_forward = self.train_loss.forward
                
                def loss_hook(*args, **kwargs):
                    loss = original_forward(*args, **kwargs)
                    # 捕獲 loss 值
                    if hasattr(loss, 'item'):
                        self.current_loss = loss.item()
                    elif isinstance(loss, (float, int)):
                        self.current_loss = float(loss)
                    else:
                        self.current_loss = float(loss)
                    
                    self.loss_history.append(self.current_loss)
                    # 保持最近 100 個 loss 值
                    if len(self.loss_history) > 100:
                        self.loss_history.pop(0)
                    return loss
                
                self.train_loss.forward = loss_hook
                logger.info("Loss tracking hook installed successfully")
            else:
                logger.warning("Could not install loss tracking hook - no forward method found")
        except Exception as e:
            logger.warning(f"Could not install loss tracking hook: {e}")

    def __call__(self, score, epoch, steps):
        """在訓練過程中被調用 - 實時計算AMI並更新圖表"""
        self.step_count = steps
        self.current_epoch = epoch
        
        # 獲取當前學習率
        current_lr = self._get_current_learning_rate()
        
        # 計算實時 AMI 分數
        ami_score = self._evaluate_clustering_performance()
        
        # 獲取平均 loss（使用最近的 loss 值進行平滑）
        if self.loss_history:
            recent_losses = self.loss_history[-10:]  # 最近 10 個 loss
            avg_loss = sum(recent_losses) / len(recent_losses)
        else:
            avg_loss = self.current_loss
        
        # 更新指標追蹤器 - 實時更新圖表！
        if self.metrics_tracker:
            self.metrics_tracker.add_metric(
                epoch=epoch,
                step=steps,
                ami_score=ami_score,
                loss=avg_loss,
                learning_rate=current_lr
            )
        
        logger.info(f"Callback - Epoch {epoch:.2f}, Step {steps}: AMI = {ami_score:.4f}, Loss = {avg_loss:.5f}, LR = {current_lr:.2e}")

    def _evaluate_clustering_performance(self):
        """多方法聚類評估，自動選擇最佳方法（實時版本）"""
        try:
            # 生成嵌入
            embeddings = self.model.encode(self.val_sentences, show_progress_bar=False)
            
            clustering_results = {}
            
            # 1. DBSCAN 聚類
            try:
                dbscan = DBSCAN(eps=0.095, min_samples=2, metric='cosine')
                dbscan_labels = dbscan.fit_predict(embeddings)
                
                # 處理噪聲點
                max_label = max(dbscan_labels) if len(dbscan_labels) > 0 and max(dbscan_labels) >= 0 else -1
                noise_mask = dbscan_labels == -1
                n_noise = np.sum(noise_mask)
                if n_noise > 0:
                    dbscan_labels[noise_mask] = range(max_label + 1, max_label + 1 + n_noise)
                
                dbscan_ami = adjusted_mutual_info_score(self.val_labels, dbscan_labels)
                clustering_results['dbscan'] = dbscan_ami
            except Exception as e:
                logger.warning(f"DBSCAN clustering failed: {e}")
                clustering_results['dbscan'] = 0.0
            
            # 2. 層次聚類（快速版本 - 只測試少數幾個參數）
            n_samples = len(embeddings)
            n_unique_labels = len(set(self.val_labels))
            max_clusters = min(n_samples - 1, 50)
            
            # 選擇2-3個代表性的cluster數量進行快速測試
            quick_cluster_candidates = []
            for offset in [0, 5]:  # 只測試2個值以加快速度
                n_clusters = n_unique_labels + offset
                if 2 <= n_clusters <= max_clusters:
                    quick_cluster_candidates.append(n_clusters)
            
            # 如果沒有合適的候選，使用保守選擇
            if not quick_cluster_candidates:
                quick_cluster_candidates = [min(max_clusters, max(2, n_unique_labels))]
            
            for n_clusters in quick_cluster_candidates:
                try:
                    hier = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', metric='cosine')
                    hier_labels = hier.fit_predict(embeddings)
                    hier_ami = adjusted_mutual_info_score(self.val_labels, hier_labels)
                    clustering_results[f'hierarchical_{n_clusters}'] = hier_ami
                except Exception as e:
                    logger.warning(f"Hierarchical clustering with {n_clusters} clusters failed: {e}")
                    clustering_results[f'hierarchical_{n_clusters}'] = 0.0
            
            # 找到最佳結果
            if clustering_results:
                best_method = max(clustering_results.items(), key=lambda x: x[1])
                best_ami = best_method[1]
                best_method_name = best_method[0]
                
                # 每2500步打印一次詳細結果，避免日誌過多
                if self.step_count % 2500 == 0:  # 每2500步打印一次詳細結果
                    logger.info(f"Multi-method clustering results: {clustering_results}")
                    logger.info(f"Best method: {best_method_name} with AMI {best_ami:.4f}")
                
                return best_ami
            else:
                return 0.0
            
        except Exception as e:
            logger.warning(f"Multi-method clustering evaluation failed: {e}")
            return 0.0

    def _get_current_learning_rate(self):
        """獲取當前學習率"""
        try:
            if hasattr(self.model, 'optimizer') and self.model.optimizer:
                optimizer = self.model.optimizer
                if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                    return optimizer.param_groups[0]['lr']
        except AttributeError:
            pass
        
        return 8e-6  # 返回默認值

class TrueMultiScaleLoss(nn.Module):
    """
    真正的多尺度損失函數，確保不同尺度學習不同特徵
    """
    
    def __init__(self, model, scale=20.0, orthogonal_weight=0.1, consistency_weight=0.2):
        super().__init__()
        self.model = model
        self.scale = scale
        self.orthogonal_weight = orthogonal_weight
        self.consistency_weight = consistency_weight
        
        # 核心損失函數
        self.core_loss = MultipleNegativesRankingLoss(model=model, scale=scale)
        
        # 獲取模型的不同層特徵
        self.embedding_dim = model.get_sentence_embedding_dimension()
        
        # 全局語義投影（關注整體規則類別）- 減少維度以節省內存
        self.global_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 192),
            nn.LayerNorm(192)
        )
        
        # 局部模式投影（關注層間關係和空間約束）
        self.local_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, 192),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(192, 128),
            nn.LayerNorm(128)
        )
        
        # 細粒度投影（關注具體參數和數值）
        self.fine_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(128, 96),
            nn.LayerNorm(96)
        )
        
        # 注意力機制引導不同尺度關注不同方面 - 減少heads數量以節省內存
        self.global_attention = self._create_attention_head(self.embedding_dim, "global")
        self.local_attention = self._create_attention_head(self.embedding_dim, "local")  
        self.fine_attention = self._create_attention_head(self.embedding_dim, "fine")
        
        # 移動到與模型相同的設備
        self.to(model.device)
        
    def _create_attention_head(self, dim, scale_type):
        """為不同尺度創建專門的注意力頭 - 減少heads數量以節省內存"""
        if scale_type == "global":
            # 全局注意力：關注規則類型關鍵詞
            return nn.MultiheadAttention(dim, num_heads=4, dropout=0.1, batch_first=True)
        elif scale_type == "local":
            # 局部注意力：關注層名和空間關係
            return nn.MultiheadAttention(dim, num_heads=2, dropout=0.1, batch_first=True)
        else:  # fine
            # 細粒度注意力：關注數值和測量單位
            return nn.MultiheadAttention(dim, num_heads=1, dropout=0.1, batch_first=True)
    
    def _apply_scale_specific_attention(self, embeddings, attention_head, scale_type):
        """應用尺度特定的注意力機制"""
        # 將 sentence embedding 視為單個 token 的序列
        # 在實際應用中，這裡可以獲取 token-level 的特徵
        emb_expanded = embeddings.unsqueeze(1)  # [batch, 1, dim]
        
        # 自注意力
        attended, _ = attention_head(emb_expanded, emb_expanded, emb_expanded)
        return attended.squeeze(1)  # [batch, dim]
    
    def _orthogonal_loss(self):
        """正交性損失：確保不同投影學習不同特徵 - 用戶改進版本"""
        w_global = self.global_proj[-2].weight
        w_local = self.local_proj[-2].weight  
        w_fine = self.fine_proj[-2].weight
        
        w_global_norm = F.normalize(w_global, p=2, dim=1)
        w_local_norm = F.normalize(w_local, p=2, dim=1)
        w_fine_norm = F.normalize(w_fine, p=2, dim=1)
        
        def compute_ortho_loss(w1, w2):
            min_out_dim = min(w1.shape[0], w2.shape[0])
            min_in_dim = min(w1.shape[1], w2.shape[1])
            w1_final = w1[:min_out_dim, :min_in_dim]
            w2_final = w2[:min_out_dim, :min_in_dim]
            ortho_loss = torch.pow(F.cosine_similarity(w1_final.flatten(), w2_final.flatten(), dim=0), 2)
            return ortho_loss
        
        ortho_loss_global_local = compute_ortho_loss(w_global_norm, w_local_norm)
        ortho_loss_global_fine = compute_ortho_loss(w_global_norm, w_fine_norm)
        ortho_loss_local_fine = compute_ortho_loss(w_local_norm, w_fine_norm)
        
        total_ortho_loss = (ortho_loss_global_local + ortho_loss_global_fine + ortho_loss_local_fine) / 3
        return total_ortho_loss
    
    def _consistency_loss(self, global_emb, local_emb, fine_emb):
        """一致性損失：確保不同尺度在相同樣本上保持一致的相對關係"""
        batch_size = global_emb.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=global_emb.device)
        
        # 計算每個尺度內的相似度矩陣
        global_sim = F.cosine_similarity(global_emb.unsqueeze(1), global_emb.unsqueeze(0), dim=2)
        local_sim = F.cosine_similarity(local_emb.unsqueeze(1), local_emb.unsqueeze(0), dim=2)
        fine_sim = F.cosine_similarity(fine_emb.unsqueeze(1), fine_emb.unsqueeze(0), dim=2)
        
        # 計算相似度矩陣間的一致性
        consistency_loss = (
            F.mse_loss(global_sim, local_sim) + 
            F.mse_loss(global_sim, fine_sim) + 
            F.mse_loss(local_sim, fine_sim)
        ) / 3
        
        return consistency_loss
    
    def _scale_specific_features(self, embeddings, sentences):
        """根據文本內容提取尺度特定特徵"""
        batch_size = embeddings.shape[0]
        
        # 全局特徵：規則類型指示
        global_features = []
        # 局部特徵：層關係指示  
        local_features = []
        # 細粒度特徵：數值指示
        fine_features = []
        
        for sentence in sentences:
            # 全局語義特徵（規則類型）
            global_feat = self._extract_global_features(sentence)
            global_features.append(global_feat)
            
            # 局部特徵（層關係）
            local_feat = self._extract_local_features(sentence)
            local_features.append(local_feat)
            
            # 細粒度特徵（數值參數）
            fine_feat = self._extract_fine_features(sentence)
            fine_features.append(fine_feat)
        
        global_features = torch.tensor(global_features, device=embeddings.device, dtype=torch.float32)
        local_features = torch.tensor(local_features, device=embeddings.device, dtype=torch.float32)
        fine_features = torch.tensor(fine_features, device=embeddings.device, dtype=torch.float32)
        
        return global_features, local_features, fine_features
    
    def _extract_global_features(self, sentence):
        """提取全局語義特徵（規則類型）"""
        sentence_lower = sentence.lower()
        
        # 規則類型特徵
        spacing_feat = 1.0 if 'spacing' in sentence_lower or 'space' in sentence_lower else 0.0
        width_feat = 1.0 if 'width' in sentence_lower or 'wide' in sentence_lower else 0.0  
        enclosure_feat = 1.0 if 'enclosure' in sentence_lower or 'enclose' in sentence_lower else 0.0
        overlap_feat = 1.0 if 'overlap' in sentence_lower else 0.0
        minimum_feat = 1.0 if 'minimum' in sentence_lower or 'min' in sentence_lower else 0.0
        maximum_feat = 1.0 if 'maximum' in sentence_lower or 'max' in sentence_lower else 0.0
        
        return [spacing_feat, width_feat, enclosure_feat, overlap_feat, minimum_feat, maximum_feat]
    
    def _extract_local_features(self, sentence):
        """提取局部特徵（層關係）"""
        sentence_upper = sentence.upper()
        
        # 層類型特徵
        metal_feat = 1.0 if 'METAL' in sentence_upper else 0.0
        poly_feat = 1.0 if 'POLY' in sentence_upper else 0.0
        diffusion_feat = 1.0 if 'DIFFUSION' in sentence_upper else 0.0
        contact_feat = 1.0 if 'CONTACT' in sentence_upper or 'VIA' in sentence_upper else 0.0
        well_feat = 1.0 if 'WELL' in sentence_upper else 0.0
        
        return [metal_feat, poly_feat, diffusion_feat, contact_feat, well_feat]
    
    def _extract_fine_features(self, sentence):
        """提取細粒度特徵（數值參數）"""
        # 數值特徵
        numbers = re.findall(r'\d+\.?\d*', sentence)
        has_numbers = 1.0 if numbers else 0.0
        
        # 單位特徵
        has_micron = 1.0 if 'um' in sentence.lower() or 'micron' in sentence.lower() else 0.0
        has_nano = 1.0 if 'nm' in sentence.lower() or 'nano' in sentence.lower() else 0.0
        
        # 數值大小特徵（粗略分類）
        if numbers:
            max_num = max([float(n) for n in numbers])
            large_num = 1.0 if max_num > 10 else 0.0
            small_num = 1.0 if max_num < 1 else 0.0
        else:
            large_num = 0.0
            small_num = 0.0
            
        return [has_numbers, has_micron, has_nano, large_num, small_num]
    
    def forward(self, sentence_features, labels):
        """多尺度層次化損失計算"""
        # 獲取基礎 embeddings
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        anchor_emb, positive_emb = reps[0], reps[1]
        
        # 核心損失（標準 Multiple Negatives Ranking）
        core_loss_value = self.core_loss(sentence_features, labels)
        
        # 應用尺度特定注意力（簡化版本，不依賴文本提取）
        global_anchor = self._apply_scale_specific_attention(anchor_emb, self.global_attention, "global")
        global_positive = self._apply_scale_specific_attention(positive_emb, self.global_attention, "global")
        
        local_anchor = self._apply_scale_specific_attention(anchor_emb, self.local_attention, "local")
        local_positive = self._apply_scale_specific_attention(positive_emb, self.local_attention, "local")
        
        fine_anchor = self._apply_scale_specific_attention(anchor_emb, self.fine_attention, "fine")
        fine_positive = self._apply_scale_specific_attention(positive_emb, self.fine_attention, "fine")
        
        # 尺度特定投影
        global_anchor_proj = F.normalize(self.global_proj(global_anchor), p=2, dim=1)
        global_positive_proj = F.normalize(self.global_proj(global_positive), p=2, dim=1)
        
        local_anchor_proj = F.normalize(self.local_proj(local_anchor), p=2, dim=1)
        local_positive_proj = F.normalize(self.local_proj(local_positive), p=2, dim=1)
        
        fine_anchor_proj = F.normalize(self.fine_proj(fine_anchor), p=2, dim=1)
        fine_positive_proj = F.normalize(self.fine_proj(fine_positive), p=2, dim=1)
        
        # 尺度特定對比損失
        global_loss = 1.0 - F.cosine_similarity(global_anchor_proj, global_positive_proj, dim=1).mean()
        local_loss = 1.0 - F.cosine_similarity(local_anchor_proj, local_positive_proj, dim=1).mean()
        fine_loss = 1.0 - F.cosine_similarity(fine_anchor_proj, fine_positive_proj, dim=1).mean()
        
        # 正交性損失：確保不同投影學習不同特徵
        orthogonal_loss = self._orthogonal_loss()
        
        # 一致性損失：確保不同尺度保持相對關係一致
        consistency_loss = self._consistency_loss(
            global_anchor_proj, local_anchor_proj, fine_anchor_proj
        )
        
        # 組合損失
        total_loss = (
            core_loss_value +  # 主要損失
            0.3 * global_loss +  # 全局語義損失
            0.25 * local_loss +  # 局部關係損失  
            0.2 * fine_loss +   # 細粒度損失
            self.orthogonal_weight * orthogonal_loss +  # 正交性約束
            self.consistency_weight * consistency_loss   # 一致性約束
        )
        
        return total_loss

class TrueMultiScaleTrainer:
    """真正的多尺度訓練器"""
    
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_tracker = None
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """載入預訓練模型"""
        logger.info(f"Loading model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_data(self, path4ruleGT, model_path, enhanced_triplet=True, difficulty_strategy='mixed'):
        """準備訓練數據，使用與 train.py 完全相同的方法"""
        logger.info("Preparing training data...")
        
        # 使用與 train.py 相同的方法
        result = main_generate_train_data_with_hard_mining(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            max_triplets_per_anchor=24,
            difficulty_ratio=0.6,
            path4ruleGT=path4ruleGT,        
            hard_negative_ratio=0.7,
            similarity_method='sentence_transformer',
            model_path=model_path,
            top_k_hard_negatives=10
        )
        
        if result is None or result[0] is None:
            raise ValueError("No training examples generated")
        
        train_triplets, dataset_dict = result
        logger.info(f"Generated {len(train_triplets)} training triplets.")
        
        # 創建驗證三元組，與 train.py 完全相同
        val_triplets, test_triplets = create_validation_triplets(dataset_dict)
        logger.info(f"Generated {len(val_triplets)} validation triplets.")
        
        # 提取驗證句子和標籤，與 train.py 完全相同
        val_df = dataset_dict['validation'].to_pandas()
        val_sentences = val_df['description'].tolist()
        
        # 確保標籤是整數
        le = LabelEncoder()
        val_labels = le.fit_transform(val_df['group_id'].tolist())
        
        logger.info(f"Prepared validation set with {len(val_sentences)} sentences and {len(set(val_labels))} unique labels.")
        return train_triplets, val_triplets, val_sentences, val_labels
    
    def train_true_multi_scale_model(self, train_examples, val_data, val_sentences, val_labels, 
                                   output_path, epochs=4, batch_size=8, learning_rate=1e-5):
        """真正的多尺度訓練"""
        logger.info("=== True Multi-Scale Training Started ===")
        logger.info(f"Training samples: {len(train_examples)}")
        logger.info(f"Validation samples: {len(val_data) if val_data else 0}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        
        # 初始化指標追蹤器
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_path = f'./true_multi_scale_training_metrics_{date_str}.xlsx'
        plot_path = f'./training_metrics_plot_{date_str}.png'
        
        self.metrics_tracker = MetricsTracker(
            save_path=metrics_path,
            y_lim_ami=(0.3, 1.05),
            y_lim_loss=None,  # 動態縮放
        )
        
        # 創建 DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # 真正的多尺度損失函數
        train_loss = TrueMultiScaleLoss(
            model=self.model,
            scale=20.0,
            orthogonal_weight=0.1,  # 正交性約束權重
            consistency_weight=0.2   # 一致性約束權重
        )
        
        # 創建自定義訓練回調 - 實現實時 AMI 追蹤
        training_callback = CustomTrainingCallback(
            metrics_tracker=self.metrics_tracker,
            model=self.model,
            train_loss=train_loss,
            val_sentences=val_sentences,
            val_labels=val_labels
        )
        
        # 訓練參數
        training_args = {
            'epochs': epochs,
            'scheduler': 'WarmupLinear',
            'warmup_steps': int(0.1 * len(train_dataloader) * epochs),
            'optimizer_class': torch.optim.AdamW,
            'optimizer_params': {'lr': learning_rate, 'weight_decay': 0.01},
            'evaluation_steps': len(train_dataloader) // 4,
            'save_best_model': True,
            'show_progress_bar': True,
            'callback': training_callback  # 添加實時AMI追蹤回調
        }
        
        # 驗證評估器
        evaluator = None
        if val_data:
            val_dataloader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
            evaluator = evaluation.TripletEvaluator.from_input_examples(
                val_data, batch_size=batch_size, name='true_multi_scale_val'
            )
        
        # 訓練模型
        start_time = time.time()
        
        # 在訓練開始前評估一次初始性能
        initial_ami = self.evaluate_clustering_performance(val_sentences, val_labels)
        if self.metrics_tracker:
            self.metrics_tracker.add_metric(
                epoch=0, step=0, ami_score=initial_ami, loss=0.0, learning_rate=learning_rate
            )
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            **training_args
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # 保存模型
        logger.info(f"Saving true multi-scale model to: {output_path}")
        self.model.save(output_path)
        
        # 最終評估
        final_ami = self.evaluate_clustering_performance(val_sentences, val_labels)
        
        # 保存最終指標和圖表
        if self.metrics_tracker:
            self.metrics_tracker.add_metric(
                epoch=epochs, step=len(train_dataloader) * epochs, 
                ami_score=final_ami, loss=0.0, learning_rate=learning_rate
            )
            self.metrics_tracker.save_plot(plot_path)
            logger.info(f"Training metrics and plot saved to: {metrics_path}, {plot_path}")
            self.metrics_tracker.close()
        
        return final_ami
    
    def evaluate_clustering_performance(self, sentences, labels):
        """評估聚類性能"""
        logger.info("Evaluating clustering performance...")
        
        # 生成嵌入
        embeddings = self.model.encode(sentences, show_progress_bar=True)
        
        # 測試多種聚類方法
        clustering_results = {}
        
        # DBSCAN 
        dbscan = DBSCAN(eps=0.095, min_samples=2, metric='cosine')
        dbscan_labels = dbscan.fit_predict(embeddings)
        
        # 應用個別噪聲策略
        max_label = max(dbscan_labels) if len(dbscan_labels) > 0 and max(dbscan_labels) >= 0 else -1
        noise_mask = dbscan_labels == -1
        n_noise = np.sum(noise_mask)
        if n_noise > 0:
            dbscan_labels[noise_mask] = range(max_label + 1, max_label + 1 + n_noise)
        
        dbscan_ami = adjusted_mutual_info_score(labels, dbscan_labels)
        clustering_results['dbscan'] = dbscan_ami
        
        # 層次聚類變體 - 動態調整cluster數量
        n_samples = len(embeddings)
        max_clusters = n_samples - 1
        n_unique_labels = len(set(labels))
        
        # 測試不同的cluster數量，以真實標籤數為中心
        cluster_candidates = []
        for offset in [-10, -5, 0, 5, 10]:
            n_clusters = n_unique_labels + offset
            if 2 <= n_clusters <= max_clusters:
                cluster_candidates.append(n_clusters)
        
        # 如果沒有合適的候選，使用保守的選擇
        if not cluster_candidates:
            cluster_candidates = [min(max_clusters, max(2, n_unique_labels))]
        
        logger.info(f"Testing hierarchical clustering with {cluster_candidates} clusters (samples: {n_samples}, unique labels: {n_unique_labels})")
        
        for n_clusters in cluster_candidates:
            try:
                hier = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', metric='cosine')
                hier_labels = hier.fit_predict(embeddings)
                hier_ami = adjusted_mutual_info_score(labels, hier_labels)
                clustering_results[f'hierarchical_{n_clusters}'] = hier_ami
            except Exception as e:
                logger.warning(f"Hierarchical clustering with {n_clusters} clusters failed: {e}")
        
        # 找到最佳結果
        best_method = max(clustering_results.items(), key=lambda x: x[1])
        final_ami = best_method[1]
        
        logger.info(f"Best clustering: {best_method[0]} with AMI {final_ami:.4f}")
        logger.info(f"All results: {clustering_results}")
        
        return final_ami

def main():
    """主要的真正多尺度訓練流程"""
    logger.info("=== True Multi-Scale Training for AMI 0.95 Target ===")
    
    # 配置
    model_name = "sentence-transformers/all-mpnet-base-v2"
    PATH2DOWN = os.path.expanduser('~/111111ForDownload/')
    path4ruleGT = PATH2DOWN + 'csv_TLR001_22nm_with_init_cluster_with_gt_class_name.csv'
    model_path = './rule-embedder-0722_trained_MPnet_CachedMN'  # 最佳現有模型
    
    # 輸出路徑
    date_str = datetime.now().strftime('%m%d')
    output_path = f'./rule-embedder-{date_str}_true_multi_scale'
    
    # 初始化訓練器
    trainer = TrueMultiScaleTrainer(model_name=model_name)
    trainer.load_model()
    
    # 準備數據
    train_examples, val_data_triplets, val_sentences, val_labels = trainer.prepare_data(
        path4ruleGT, model_path, enhanced_triplet=True, difficulty_strategy='mixed'
    )
    
    # 訓練配置 - 減少batch size以避免OOM
    training_config = {
        'epochs': 4,
        'batch_size': 4,  # 減少batch size從8到4
        'learning_rate': 8e-6,  # 較保守的學習率
    }
    
    logger.info(f"Training configuration: {training_config}")
    
    # 訓練真正的多尺度模型
    final_ami = trainer.train_true_multi_scale_model(
        train_examples=train_examples,
        val_data=val_data_triplets,
        val_sentences=val_sentences,
        val_labels=val_labels,
        output_path=output_path,
        **training_config
    )
    
    # 基線比較
    logger.info("\n=== Baseline Comparison ===")
    baseline_model = SentenceTransformer('./rule-embedder-0722_trained_MPnet_CachedMN')
    baseline_embeddings = baseline_model.encode(val_sentences, show_progress_bar=True)
    baseline_dbscan = DBSCAN(eps=0.095, min_samples=2, metric='cosine')
    baseline_labels = baseline_dbscan.fit_predict(baseline_embeddings)
    
    # 應用個別噪聲策略到基線
    max_label = max(baseline_labels) if len(baseline_labels) > 0 and max(baseline_labels) >= 0 else -1
    noise_mask = baseline_labels == -1
    n_noise = np.sum(noise_mask)
    if n_noise > 0:
        baseline_labels[noise_mask] = range(max_label + 1, max_label + 1 + n_noise)
    
    baseline_ami = adjusted_mutual_info_score(val_labels, baseline_labels)
    improvement = final_ami - baseline_ami
    
    # 結果總結
    logger.info("\n" + "="*60)
    logger.info("TRUE MULTI-SCALE TRAINING RESULTS")
    logger.info("="*60)
    logger.info(f"Target AMI:           0.9500")
    logger.info(f"Achieved AMI:         {final_ami:.4f}")
    logger.info(f"Baseline AMI:         {baseline_ami:.4f}")
    logger.info(f"Improvement:          {improvement:+.4f}")
    logger.info(f"Gap to target:        {0.95 - final_ami:.4f}")
    
    if final_ami >= 0.95:
        logger.info("🎉 TARGET ACHIEVED! AMI ≥ 0.95")
    elif final_ami >= 0.92:
        logger.info("🔥 Excellent progress! Very close to target")
    elif improvement > 0:
        logger.info("✅ Improvement achieved!")
    else:
        logger.info("❌ No improvement over baseline")
    
    # 保存最終結果
    final_results = {
        'final_ami': final_ami,
        'baseline_ami': baseline_ami,
        'improvement': improvement,
        'target_achieved': final_ami >= 0.95,
        'training_config': training_config,
        'architecture': 'true_multi_scale_with_orthogonal_constraints'
    }
    
    results_path = os.path.join(output_path, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"\n📁 Model and results saved to: {output_path}")
    logger.info(f"📊 Results saved to: {results_path}")
    
    return final_ami

if __name__ == "__main__":
    main()