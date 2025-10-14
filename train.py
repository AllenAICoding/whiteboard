import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses, evaluation
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import logging
import os
import time
import numpy as np
from data_prepare import main_generate_train_data,create_validation_triplets,main_generate_train_data_with_hard_mining
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime

# 設置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ],
    force=True  # Force reconfiguration of logging
)
logger = logging.getLogger(__name__)

class EmbeddingTrainer:
    """嵌入模型訓練器"""
    
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_tracker = None
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """加載預訓練模型"""
        logger.info(f"Loading model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_data(self, path4ruleGT, model_path, enhanced_triplet=True, difficulty_strategy='mixed'):
        """準備訓練數據"""
        logger.info("Preparing training data...")
        
        # 使用改進的數據預處理函數
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
        
        # 創建驗證三元組
        val_triplets, test_triplets = create_validation_triplets(dataset_dict)
        logger.info(f"Generated {len(val_triplets)} validation triplets.")
        
        # 提取驗證句子和標籤
        val_df = dataset_dict['validation'].to_pandas()
        val_sentences = val_df['description'].tolist()
        
        # 確保標籤是整數
        le = LabelEncoder()
        val_labels = le.fit_transform(val_df['group_id'].tolist())
        
        logger.info(f"Prepared validation set with {len(val_sentences)} sentences and {len(set(val_labels))} unique labels.")
        return train_triplets, val_triplets, val_sentences, val_labels
    
    def configure_training(self, train_data, val_data=None, loss_type='triplet'):
        """配置訓練參數"""
        
        # 動態調整批次大小 - MS Loss benefits from larger batches
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if loss_type in ['multi_similarity', 'cached_multiple_negatives']:  # These losses benefit from larger batches
                if loss_type == 'cached_multiple_negatives':
                    # Cached loss can handle larger batch sizes, try 16 for better performance
                    if gpu_memory >= 16:
                        batch_size = 24  # Slightly larger for high-end GPUs
                    elif gpu_memory >= 8:
                        batch_size = 16  # Target batch size for mid-range GPUs
                    else:
                        batch_size = 8   # Conservative for lower-end GPUs
                else:
                    # MS Loss benefits from larger batches for better hard negative mining
                    if gpu_memory >= 16:
                        batch_size = 16
                    elif gpu_memory >= 8:
                        batch_size = 8
                    else:
                        batch_size = 4
            else:
                # Standard batch sizes for other loss functions
                if gpu_memory >= 16:
                    batch_size = 16
                elif gpu_memory >= 8:
                    batch_size = 8
                else:
                    batch_size = 4
        else:
            if loss_type == 'cached_multiple_negatives':
                batch_size = 16  # Target batch size for cached loss
            elif loss_type == 'multi_similarity':
                batch_size = 8
            else:
                batch_size = 4
        
        logger.info(f"Using batch size: {batch_size}")
        
        # 創建 DataLoader
        train_dataloader = DataLoader(
            train_data, 
            shuffle=True, 
            batch_size=batch_size
        )
        
        val_dataloader = None
        if val_data:
            val_dataloader = DataLoader(
                val_data, 
                shuffle=False, 
                batch_size=batch_size
            )
        
        # 選擇損失函數
        if loss_type == 'circle':
            # Circle Loss - check if available in sentence-transformers
            try:
                train_loss = losses.CircleLoss(
                    model=self.model,
                    scale=256,  # Temperature parameter (higher = more confident)
                    margin=0.25  # Margin parameter
                )
                logger.info("Using Circle Loss (scale=256, margin=0.25)")
            except (AttributeError, ImportError):
                logger.warning("Circle Loss not available in this sentence-transformers version. Falling back to Multiple Negatives Ranking Loss.")
                train_loss = losses.MultipleNegativesRankingLoss(
                    model=self.model,
                    scale=20.0
                )
                logger.info("Using Multiple Negatives Ranking Loss (scale=20.0) as Circle Loss fallback")
        elif loss_type == 'multiple_negatives':
            # Multiple Negatives Ranking Loss - good for clustering
            train_loss = losses.MultipleNegativesRankingLoss(
                model=self.model,
                scale=20.0  # Temperature parameter
            )
            logger.info("Using Multiple Negatives Ranking Loss (scale=20.0)")
        elif loss_type == 'cached_multiple_negatives':
            # Cached Multiple Negatives Ranking Loss - allows larger effective batch sizes
            try:
                train_loss = losses.CachedMultipleNegativesRankingLoss(
                    model=self.model,
                    scale=20.0,  # Temperature parameter
                    mini_batch_size=16  # Smaller chunks to save memory
                )
                logger.info("Using Cached Multiple Negatives Ranking Loss (scale=20.0, mini_batch_size=16)")
            except (AttributeError, ImportError):
                logger.warning("CachedMultipleNegativesRankingLoss not available. Falling back to Multiple Negatives Ranking Loss.")
                train_loss = losses.MultipleNegativesRankingLoss(
                    model=self.model,
                    scale=20.0
                )
                logger.info("Using Multiple Negatives Ranking Loss (scale=20.0) as fallback")
        elif loss_type == 'cosine':
            # Cosine Similarity Loss - simple but effective
            train_loss = losses.CosineSimilarityLoss(
                model=self.model
            )
            logger.info("Using Cosine Similarity Loss")
        elif loss_type == 'multi_similarity':
            # Multi-Similarity Loss - automatic hard negative mining
            try:
                from pytorch_metric_learning import losses as pml_losses
                
                # Create MS Loss wrapper that extends sentence-transformers loss base class
                from sentence_transformers.losses import Loss
                
                class MultiSimilarityLossWrapper(Loss):
                    def __init__(self, model, alpha=2.0, beta=50.0, base=0.5):
                        super().__init__(model)
                        self.ms_loss = pml_losses.MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)
                    
                    def forward(self, sentence_features, labels=None):
                        # Get embeddings for all texts in the batch
                        # sentence_features is a list of dictionaries
                        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
                        
                        # For triplets: reps[0] = anchors, reps[1] = positives, reps[2] = negatives
                        if len(reps) == 3:
                            # Triplet format: interleave anchor, positive, negative
                            batch_size = reps[0].size(0)
                            embeddings = torch.stack([reps[0], reps[1], reps[2]], dim=1).view(-1, reps[0].size(1))
                            
                            # Create labels for MS Loss: [0, 0, 1, 1, 1, 2, 2, 2, ...]
                            labels = torch.arange(batch_size).repeat_interleave(3).to(embeddings.device)
                            # Set negative labels to different class
                            labels[2::3] = labels[2::3] + batch_size
                        else:
                            # Fallback: concatenate all embeddings
                            embeddings = torch.cat(reps, dim=0)
                            labels = torch.arange(embeddings.size(0)).to(embeddings.device)
                        
                        # Compute MS Loss
                        return self.ms_loss(embeddings, labels)
                
                train_loss = MultiSimilarityLossWrapper(
                    model=self.model,
                    alpha=2.0,  # Controls the strength of positive pairs
                    beta=50.0,  # Controls the strength of negative pairs
                    base=0.5   # Base threshold for mining
                )
                logger.info("Using Multi-Similarity Loss Wrapper (alpha=2.0, beta=50.0, base=0.5)")
            except ImportError:
                logger.warning("pytorch-metric-learning not available. Falling back to Multiple Negatives Ranking Loss.")
                train_loss = losses.MultipleNegativesRankingLoss(
                    model=self.model,
                    scale=20.0
                )
                logger.info("Using Multiple Negatives Ranking Loss (scale=20.0) as Multi-Similarity Loss fallback")
        else:
            # Default: Triplet Loss
            train_loss = losses.TripletLoss(
                model=self.model,
                distance_metric=losses.TripletDistanceMetric.COSINE,
                triplet_margin=0.5
            )
            logger.info("Using Triplet Loss (cosine distance, margin=0.5)")
        
        # 設置評估器
        evaluator = None
        if val_data:
            evaluator = evaluation.TripletEvaluator.from_input_examples(
                val_data, 
                name='validation'
            )
        
        return train_dataloader, val_dataloader, train_loss, evaluator
    
    def train(self, train_dataloader, train_loss, evaluator=None, 
              num_epochs=3, learning_rate=2e-5, output_path='./trained-embedder',**kwargs):
        """執行訓練（修復版，包含正確的callback使用）"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # 初始化指標追蹤器
        self.metrics_tracker = MetricsTracker(
            save_path=f'{output_path}/training_metrics.xlsx'
        )
        
        warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
        
        logger.info("Starting training...")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Warmup steps: {warmup_steps}")
        logger.info(f"Output path: {output_path}")
        
        # 創建自定義訓練回調
        training_callback = CustomTrainingCallback(
            evaluator=evaluator,
            metrics_tracker=self.metrics_tracker,
            model=self.model,
            train_loss=train_loss
        )
        
        # 訓練配置
        training_args = {
            'train_objectives': [(train_dataloader, train_loss)],
            'epochs': num_epochs,
            'warmup_steps': warmup_steps,
            'output_path': output_path,
            'optimizer_params': {'lr': learning_rate},
            'max_grad_norm': kwargs.get('max_grad_norm',2.0),
            'show_progress_bar': True,
            'save_best_model': True,
            'checkpoint_save_steps': 500,
            'checkpoint_save_total_limit': 3,
            'callback': training_callback,  # 添加回調
        }
        
        # 如果有評估器，添加評估配置
        if evaluator:
            training_args['evaluator'] = evaluator
            training_args['evaluation_steps'] = 500
        
        # 開始訓練
        start_time = time.time()
        
        try:
            self.model.fit(**training_args)
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # 保存最終圖表
            if self.metrics_tracker:
                self.metrics_tracker.save_plot(f'{output_path}/training_progress.png')
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # 關閉圖表
            if self.metrics_tracker:
                self.metrics_tracker.close()
    
    def evaluate_model(self, model_path, test_sentences=None):
        """評估訓練後的模型"""
        logger.info("Evaluating trained model...")
        
        # 加載訓練後的模型
        trained_model = SentenceTransformer(model_path)
        
        # 如果沒有提供測試句子，使用默認的
        if test_sentences is None:
            test_sentences = [
                "DIFFUSION width [over TG_25 device] (exclude maximum width of NATIVE, MIS_IMPLANT_BLOCK, LDNMOS, LDPMOS, CAD_IO_ID_MARK (GDS No: 91(0)) and GATED_DIODE (GDS No: 22(29)))",
                "DIFFUSION maximum width",
                "DIFFUSION width [over TG_18 device] (exclude maximum width of NATIVE, MIS_IMPLANT_BLOCK, LDNMOS, LDPMOS, CAD_IO_ID_MARK (GDS No: 91(0)) and GATED_DIODE (GDS No: 22(29)))",
                "POLY1_CUSTOMER_DUMMY width",
                "N+ IMPLANT width",
                "PHANTOM_DIFFUSION width over TG",
            ]
        
        # 生成嵌入
        embeddings = trained_model.encode(test_sentences)
        
        # 計算相似度矩陣
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        logger.info("Similarity matrix:")
        for i, sent in enumerate(test_sentences):
            logger.info(f"{i}: {sent[:50]}...")
        
        # 打印相似度矩陣
        import numpy as np
        np.set_printoptions(precision=3, suppress=True)
        logger.info(f"\nSimilarity matrix:\n{similarity_matrix}")
        
        # 分析相似群組
        self._analyze_clusters(test_sentences, similarity_matrix)
        
        return trained_model, embeddings, similarity_matrix
    
    def _analyze_clusters(self, sentences, similarity_matrix, threshold=0.7):
        """分析相似群組"""
        logger.info(f"\nAnalyzing clusters with threshold {threshold}:")
        
        n = len(sentences)
        visited = [False] * n
        clusters = []
        
        for i in range(n):
            if not visited[i]:
                cluster = [i]
                visited[i] = True
                
                for j in range(i + 1, n):
                    if not visited[j] and similarity_matrix[i][j] >= threshold:
                        cluster.append(j)
                        visited[j] = True
                
                clusters.append(cluster)
        
        for idx, cluster in enumerate(clusters):
            logger.info(f"Cluster {idx + 1}:")
            for sent_idx in cluster:
                logger.info(f"  - {sentences[sent_idx][:60]}...")

class CustomTrainingCallback_deprecated:
    """自定義訓練回調，用於追蹤指標"""
    
    def __init__(self, evaluator, metrics_tracker, model, train_loss):
        self.evaluator = evaluator
        self.metrics_tracker = metrics_tracker
        self.model = model
        self.train_loss = train_loss
        self.step_count = 0
        self.current_epoch = 0
        self.current_loss = 0.0
        
    def __call__(self, score, epoch, steps):
        """在訓練過程中被調用"""
        self.step_count = steps
        self.current_epoch = epoch
        
        # 獲取當前學習率
        current_lr = self._get_current_learning_rate()
        
        # 如果有自定義評估器，獲取額外指標
        ami_score = 0.0
        if hasattr(self.evaluator, '__call__'):
            try:
                ami_score = self.evaluator(self.model, epoch=epoch, steps=steps)
            except:
                ami_score = score  # 使用默認分數
        else:
            ami_score = score
        
        # 更新指標追蹤器
        if self.metrics_tracker:
            self.metrics_tracker.add_metric(
                epoch=epoch,
                step=steps,
                ami_score=ami_score,
                loss=self.current_loss,
                learning_rate=current_lr
            )
        
        logger.info(f"Epoch {epoch}, Step {steps}: Score = {ami_score:.4f}, Loss = {self.current_loss:.4f}, LR = {current_lr:.2e}")
        
        return ami_score
    
    def set_current_loss(self, loss):
        """設置當前損失值"""
        self.current_loss = loss
    
    def _get_current_learning_rate(self):
        """獲取當前學習率"""
        try:
            # 嘗試從模型的優化器獲取學習率
            if hasattr(self.model, '_modules') and hasattr(self.model._modules, 'optimizer'):
                optimizer = self.model._modules.optimizer
                if optimizer and len(optimizer.param_groups) > 0:
                    return optimizer.param_groups[0]['lr']
        except:
            pass
        
        # 如果無法獲取，返回默認值
        return 2e-5

class CustomTrainingCallback:
    """
    自定義訓練回調，用於追蹤指標 (修正版 - 支持實際 loss 追蹤)
    """
    def __init__(self, evaluator, metrics_tracker, model, train_loss):
        self.evaluator = evaluator
        self.metrics_tracker = metrics_tracker
        self.model = model
        self.train_loss = train_loss
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
        """
        在訓練過程中被調用。
        重要：此方法不應該有返回值 (implicitly returns None)。
        """
        self.step_count = steps
        self.current_epoch = epoch
        
        # 獲取當前學習率
        current_lr = self._get_current_learning_rate()
        
        ami_score = score
        
        # 獲取平均 loss（使用最近的 loss 值進行平滑）
        if self.loss_history:
            recent_losses = self.loss_history[-10:]  # 最近 10 個 loss
            avg_loss = sum(recent_losses) / len(recent_losses)
        else:
            avg_loss = self.current_loss
        
        # 更新指標追蹤器 - 現在有真實的 loss 了！
        if self.metrics_tracker:
            self.metrics_tracker.add_metric(
                epoch=epoch,
                step=steps,
                ami_score=ami_score,
                loss=avg_loss,  # 現在是真實的 loss！
                learning_rate=current_lr
            )
        
        logger.info(f"Callback - Epoch {epoch:.2f}, Step {steps}: AMI = {ami_score:.4f}, Loss = {avg_loss:.5f}, LR = {current_lr:.2e}")

    def _get_current_learning_rate(self):
        """獲取當前學習率"""
        try:
            if hasattr(self.model, 'optimizer') and self.model.optimizer:
                optimizer = self.model.optimizer
                if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                    return optimizer.param_groups[0]['lr']
        except AttributeError:
            pass
        
        # 備用方法：嘗試從模型內部狀態獲取
        try:
            if hasattr(self.model, '_modules'):
                for name, module in self.model._modules.items():
                    if hasattr(module, 'optimizer'):
                        optimizer = module.optimizer
                        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                            return optimizer.param_groups[0]['lr']
        except:
            pass
        
        return 2e-5  # 返回默認值
    
class EmbeddingTrainerEnhanced(EmbeddingTrainer):
    """增強版的訓練器，包含更多評估指標"""
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        super().__init__(model_name)
        self.metrics_tracker = None

    def configure_training_enhanced(self, train_data, val_data=None, val_sentences=None, val_labels=None, clustering_method='auto', loss_type='triplet'):
        """配置訓練參數（增強版）"""
        
        # 不在這裡初始化指標追蹤器，避免重複創建
        # self.metrics_tracker = MetricsTracker()
        
        # 原始配置
        train_dataloader, val_dataloader, train_loss, _ = self.configure_training(
            train_data, val_data, loss_type=loss_type
        )
        
        # 創建增強評估器
        evaluator = None
        if val_data and val_sentences and (val_labels is not None and len(val_labels) > 0):
            logger.info("Configuring CustomEvaluator with validation sentences and labels.")
            # Use custom evaluator with metrics tracker
            evaluator = CustomEvaluator(
                sentences=val_sentences,
                true_labels=val_labels,
                name='validation_enhanced',
                clustering_method=clustering_method,
                metrics_tracker=self.metrics_tracker
            )
        elif val_data:
            logger.info("Configuring standard TripletEvaluator for validation.")
            evaluator = evaluation.TripletEvaluator.from_input_examples(
                val_data, 
                name='validation'
            )
        
        return train_dataloader, val_dataloader, train_loss, evaluator
    
    def evaluate_model_enhanced(self, model_path, test_sentences=None, test_labels=None, clustering_method='compare'):
        """增強版的模型評估"""
        
        logger.info("Enhanced model evaluation...")
        
        # 加載訓練後的模型
        trained_model = SentenceTransformer(model_path)
        
        # 如果沒有提供測試數據，使用默認的
        if test_sentences is None or test_labels is None:
            test_sentences, test_labels = self._get_default_test_data()
        
        # 使用增強評估器
        evaluator = EnhancedEvaluator(trained_model)
        metrics, predicted_labels = evaluator.comprehensive_evaluation(
            test_sentences, test_labels, clustering_method=clustering_method
        )
        
        # 生成詳細報告
        self._generate_evaluation_report(
            test_sentences, test_labels, predicted_labels, metrics
        )
        
        return trained_model, metrics, predicted_labels
    
    def _get_default_test_data(self):
        """獲取默認的測試數據"""
        test_sentences = [
            "DIFFUSION width [over TG_25 device] (exclude maximum width of NATIVE, MIS_IMPLANT_BLOCK, LDNMOS, LDPMOS, CAD_IO_ID_MARK (GDS No: 91(0)) and GATED_DIODE (GDS No: 22(29)))",
            "DIFFUSION maximum width",
            "DIFFUSION width [over TG_18 device] (exclude maximum width of NATIVE, MIS_IMPLANT_BLOCK, LDNMOS, LDPMOS, CAD_IO_ID_MARK (GDS No: 91(0)) and GATED_DIODE (GDS No: 22(29)))",
            "POLY1_CUSTOMER_DUMMY width",
            "N+ IMPLANT width",
            "PHANTOM_DIFFUSION width over TG",
        ]
        
        # 假設的標籤（您需要根據實際情況調整）
        test_labels = [33, 9, 33, 0, 0, 1]  # 0: DIFFUSION, 1: POLY, 2: IMPLANT
        
        return test_sentences, test_labels
    
    def _generate_evaluation_report(self, sentences, true_labels, predicted_labels, metrics):
        """生成評估報告"""
        
        logger.info("=== Detailed Evaluation Report ===")
        
        # 1. 整體指標
        logger.info("Overall Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # 2. 聚類結果分析
        logger.info("\nClustering Analysis:")
        unique_true = set(true_labels)
        unique_pred = set(predicted_labels)
        
        logger.info(f"  True clusters: {len(unique_true)}")
        logger.info(f"  Predicted clusters: {len(unique_pred)}")
        
        # 3. 每個聚類的詳細信息
        for cluster_id in unique_pred:
            cluster_indices = [i for i, label in enumerate(predicted_labels) if label == cluster_id]
            logger.info(f"\nCluster {cluster_id} ({len(cluster_indices)} samples):")
            
            for idx in cluster_indices[:3]:  # 只顯示前3個
                logger.info(f"  - {sentences[idx][:60]}... (true: {true_labels[idx]})")
            
            if len(cluster_indices) > 3:
                logger.info(f"  ... and {len(cluster_indices) - 3} more samples")

# =======================
import matplotlib.ticker as mticker
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import evaluation

class EnhancedEvaluator:
    """增強的評估器，包含多種評估指標"""
    
    def __init__(self, model):
        self.model = model
        
    def evaluate_clustering_metrics(self, sentences, true_labels, clustering_method='auto', n_clusters=None):
        """計算聚類相關的評估指標
        
        Args:
            sentences: 輸入句子
            true_labels: 真實標籤
            clustering_method: 聚類方法 ('kmeans', 'dbscan', 'hierarchical', 'gmm', 'auto')
            n_clusters: 聚類數（對於需要指定群數的算法）
        """
        
        # 生成嵌入
        embeddings = self.model.encode(sentences)
        
        # 如果沒有指定聚類數，使用真實標籤的數量
        if n_clusters is None:
            n_clusters = len(set(true_labels))
        
        # 選擇聚類算法
        if clustering_method == 'auto':
            # 自動選擇：如果不知道群數，優先使用 DBSCAN
            clustering_method = 'dbscan' if n_clusters is None else 'kmeans'
        
        predicted_labels = self._perform_clustering(embeddings, clustering_method, n_clusters)
        
        # 計算各種指標
        metrics = {}
        metrics['clustering_method'] = clustering_method
        metrics['n_predicted_clusters'] = len(set(predicted_labels)) if -1 not in predicted_labels else len(set(predicted_labels)) - 1
        
        # 如果 DBSCAN 產生了噪音點（-1），需要特別處理
        if -1 in predicted_labels:
            noise_ratio = np.sum(predicted_labels == -1) / len(predicted_labels)
            metrics['noise_ratio'] = noise_ratio
            logger.info(f"DBSCAN noise ratio: {noise_ratio:.3f}")
        
        # 計算評估指標（排除噪音點）
        if len(set(predicted_labels)) > 1:  # 至少有兩個群
            # 1. AMI (Adjusted Mutual Information)
            metrics['AMI'] = adjusted_mutual_info_score(true_labels, predicted_labels)
            
            # 2. NMI (Normalized Mutual Information)
            metrics['NMI'] = normalized_mutual_info_score(true_labels, predicted_labels)
            
            # 3. Silhouette Score (排除噪音點)
            if -1 in predicted_labels:
                mask = predicted_labels != -1
                if np.sum(mask) > 1:
                    metrics['Silhouette'] = silhouette_score(embeddings[mask], predicted_labels[mask])
                else:
                    metrics['Silhouette'] = -1  # 無法計算
            else:
                metrics['Silhouette'] = silhouette_score(embeddings, predicted_labels)
            
            # 4. Calinski-Harabasz Index (排除噪音點)
            if -1 in predicted_labels:
                mask = predicted_labels != -1
                if np.sum(mask) > 1:
                    metrics['Calinski_Harabasz'] = calinski_harabasz_score(embeddings[mask], predicted_labels[mask])
                else:
                    metrics['Calinski_Harabasz'] = 0
            else:
                metrics['Calinski_Harabasz'] = calinski_harabasz_score(embeddings, predicted_labels)
            
            # 5. Davies-Bouldin Index (越小越好，排除噪音點)
            if -1 in predicted_labels:
                mask = predicted_labels != -1
                if np.sum(mask) > 1:
                    metrics['Davies_Bouldin'] = davies_bouldin_score(embeddings[mask], predicted_labels[mask])
                else:
                    metrics['Davies_Bouldin'] = float('inf')
            else:
                metrics['Davies_Bouldin'] = davies_bouldin_score(embeddings, predicted_labels)
            
            # 6. 聚類純度
            metrics['Purity'] = self._calculate_purity(true_labels, predicted_labels)
        else:
            # 如果只有一個群或全是噪音，設置默認值
            logger.warning("Only one cluster found or all points are noise")
            metrics.update({
                'AMI': 0.0,
                'NMI': 0.0,
                'Silhouette': -1.0,
                'Calinski_Harabasz': 0.0,
                'Davies_Bouldin': float('inf'),
                'Purity': 0.0
            })
        
        return metrics, predicted_labels
  
    def _find_best_clustering_method(self, results):
        """根據評估指標找出最佳聚類方法"""
        
        best_score = -1
        best_method = None
        
        for method, result in results.items():
            if 'error' in result:
                continue
            
            metrics = result['metrics']
            
            # 綜合評分：AMI (40%) + Silhouette (30%) + Purity (30%)
            score = (
                metrics.get('AMI', 0) * 0.4 +
                max(metrics.get('Silhouette', -1), 0) * 0.3 +  # Silhouette 可能為負
                metrics.get('Purity', 0) * 0.3
            )
            
            if score > best_score:
                best_score = score
                best_method = method
        
        return best_method
        """計算聚類純度"""
        clusters = {}
        for i, pred_label in enumerate(predicted_labels):
            if pred_label not in clusters:
                clusters[pred_label] = []
            clusters[pred_label].append(true_labels[i])
        
        total_correct = 0
        total_samples = len(true_labels)
        
        for cluster_labels in clusters.values():
            # 找到該聚類中最常見的真實標籤
            most_common = max(set(cluster_labels), key=cluster_labels.count)
            correct_count = cluster_labels.count(most_common)
            total_correct += correct_count
        
        return total_correct / total_samples
    
    def comprehensive_evaluation(self, sentences, true_labels, clustering_method='auto', n_clusters=None):
        """綜合評估（支持多種聚類方法）"""
        
        logger.info("Starting comprehensive evaluation...")
        
        if clustering_method == 'compare':
            # 比較多種聚類方法
            clustering_results, best_method = self.compare_clustering_methods(
                sentences, true_labels
            )
            
            # 使用最佳方法的結果
            best_result = clustering_results[best_method]
            clustering_metrics = best_result['metrics']
            predicted_labels = best_result['predicted_labels']
            
            logger.info(f"Using best method: {best_method}")
            
        else:
            # 使用指定的聚類方法
            clustering_metrics, predicted_labels = self.evaluate_clustering_metrics(
                sentences, true_labels, clustering_method, n_clusters
            )
        
        # 2. 相似度指標
        similarity_metrics = self.evaluate_similarity_metrics(sentences, true_labels)
        
        # 3. 合併所有指標
        all_metrics = {**clustering_metrics, **similarity_metrics}
        
        # 4. 打印結果
        logger.info("=== Evaluation Results ===")
        logger.info("Clustering Metrics:")
        for key, value in clustering_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("Similarity Metrics:")
        for key, value in similarity_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return all_metrics, predicted_labels

# ==============0709=======================
    def _estimate_dbscan_eps(self, embeddings):
        """改進的 DBSCAN eps 參數估計"""
        from sklearn.neighbors import NearestNeighbors
        
        # 使用 k-distance 圖方法
        k = max(2, min(10, len(embeddings) // 20))  # 動態調整 k
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # 取第 k 個最近鄰的距離
        k_distances = distances[:, k-1]
        k_distances = np.sort(k_distances)
        
        # 使用更保守的 eps 估計
        # 嘗試不同的百分位數，選擇產生合理群數的
        percentiles = [60, 70, 80, 85, 90]
        
        for p in percentiles:
            eps_candidate = np.percentile(k_distances, p)
            if eps_candidate == 0.0:
                eps_candidate = 0.1 # 用一個非常小的值代替 0
            
            # 快速測試這個 eps 會產生多少群
            test_clusterer = DBSCAN(eps=eps_candidate, min_samples=2, metric='cosine')
            test_labels = test_clusterer.fit_predict(embeddings)
            n_clusters = len(set(test_labels)) - (1 if -1 in test_labels else 0)
            
            # 如果群數在合理範圍內，使用這個 eps
            if 2 <= n_clusters <= len(embeddings) // 3:
                logger.info(f"Selected eps: {eps_candidate:.4f} (percentile: {p}, clusters: {n_clusters})")
                return eps_candidate
        
        # 如果沒找到合適的，使用默認值
        eps = np.percentile(k_distances, 80)
        if eps == 0.0:
            eps = 0.1 # 用一個非常小的值代替 0

        logger.info(f"Using default eps: {eps:.4f}")
        return eps

    def evaluate_similarity_metrics(self, sentences, true_labels):
        """修復的相似度評估指標"""
        
        embeddings = self.model.encode(sentences)
        similarity_matrix = cosine_similarity(embeddings)
        
        metrics = {}
        
        # 檢查是否有足夠的數據
        if len(set(true_labels)) < 2:
            logger.warning("Only one class found in true labels")
            return {
                'Same_Class_Avg_Similarity': 0.0,
                'Diff_Class_Avg_Similarity': 0.0,
                'Similarity_Separation': 0.0,
                'Intra_Class_Std': 0.0,
                'Inter_Class_Std': 0.0
            }
        
        # 1. 同類樣本的平均相似度
        same_class_similarities = []
        diff_class_similarities = []
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                sim = similarity_matrix[i][j]
                if true_labels[i] == true_labels[j]:
                    same_class_similarities.append(sim)
                else:
                    diff_class_similarities.append(sim)
        
        # 處理空列表的情況
        if same_class_similarities:
            metrics['Same_Class_Avg_Similarity'] = np.mean(same_class_similarities)
            metrics['Intra_Class_Std'] = np.std(same_class_similarities)
        else:
            metrics['Same_Class_Avg_Similarity'] = 0.0
            metrics['Intra_Class_Std'] = 0.0
            logger.warning("No same-class pairs found")
        
        if diff_class_similarities:
            metrics['Diff_Class_Avg_Similarity'] = np.mean(diff_class_similarities)
            metrics['Inter_Class_Std'] = np.std(diff_class_similarities)
        else:
            metrics['Diff_Class_Avg_Similarity'] = 0.0
            metrics['Inter_Class_Std'] = 0.0
            logger.warning("No different-class pairs found")
        
        # 計算分離度
        if same_class_similarities and diff_class_similarities:
            metrics['Similarity_Separation'] = metrics['Same_Class_Avg_Similarity'] - metrics['Diff_Class_Avg_Similarity']
        else:
            metrics['Similarity_Separation'] = 0.0
        
        return metrics

    def _perform_clustering(self, embeddings, method, n_clusters):
        """改進的聚類執行方法"""
        
        if method == 'kmeans':
            # 確保 n_clusters 不超過樣本數
            n_clusters = min(n_clusters, len(embeddings) - 1)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            return clusterer.fit_predict(embeddings)
        
        elif method == 'dbscan':
            # 改進的 DBSCAN 參數
            # eps = self._estimate_dbscan_eps(embeddings)
            eps = 0.1
            
            # 動態調整 min_samples
            min_samples = max(2, min(10, len(embeddings) // 50))
            
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = clusterer.fit_predict(embeddings)
            
            labels = self._handle_dbscan_noise(labels, embeddings)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"DBSCAN result: {n_clusters} clusters, eps={eps:.4f}, min_samples={min_samples}")
            
            return labels
        
        elif method == 'hierarchical':
            n_clusters = min(n_clusters, len(embeddings) - 1)
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            return clusterer.fit_predict(embeddings)
        
        elif method == 'gmm':
            n_clusters = min(n_clusters, len(embeddings) - 1)
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
            return clusterer.fit_predict(embeddings)
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    def _handle_dbscan_noise(self, cluster_labels: np.ndarray, embeddings: np.ndarray, 
                            noise_strategy: str = 'individual', **kwargs) -> np.ndarray:
        """
        處理 DBSCAN 的 noise 點
        
        Args:
            cluster_labels: 原始聚類標籤
            embeddings: 嵌入向量
            noise_strategy: noise 處理策略
                - 'individual': 每個 noise 點自成一群
                - 'sub_cluster': 對 noise 點進行子聚類
                - 'nearest': 將 noise 點分配給最近的聚類
                
        Returns:
            處理後的聚類標籤
        """
        # 找到 noise 點 (標籤為 -1)
        noise_mask = cluster_labels == -1
        noise_indices = np.where(noise_mask)[0]
        
        if len(noise_indices) == 0:
            logger.info("No noise points found in DBSCAN results")
            return cluster_labels
        
        logger.info(f"Found {len(noise_indices)} noise points, applying strategy: {noise_strategy}")
        
        # 創建新的標籤數組
        new_labels = cluster_labels.copy()
        
        # 獲取當前最大的聚類標籤
        max_label = np.max(cluster_labels[cluster_labels != -1]) if np.any(cluster_labels != -1) else -1
        next_label = max_label + 1
        
        if noise_strategy == 'individual':
            # 每個 noise 點自成一群
            for i, noise_idx in enumerate(noise_indices):
                new_labels[noise_idx] = next_label + i
            logger.info(f"Assigned {len(noise_indices)} individual clusters to noise points")
            
        elif noise_strategy == 'sub_cluster':
            # 對 noise 點進行子聚類
            if len(noise_indices) > 1:
                noise_embeddings = embeddings[noise_indices]
                
                # 使用 KMeans 對 noise 點進行子聚類
                # 自動確定聚類數 (最多不超過 noise 點數量的一半)
                n_sub_clusters = min(max(1, len(noise_indices) // 3), 5)
                
                if n_sub_clusters > 1:
                    try:
                        sub_clusterer = KMeans(n_clusters=n_sub_clusters, random_state=42, n_init=10)
                        sub_labels = sub_clusterer.fit_predict(noise_embeddings)
                        
                        # 重新編碼子聚類標籤
                        for i, noise_idx in enumerate(noise_indices):
                            new_labels[noise_idx] = next_label + sub_labels[i]
                        
                        logger.info(f"Created {n_sub_clusters} sub-clusters for noise points")
                    except:
                        # 如果子聚類失敗，回退到個別處理
                        for i, noise_idx in enumerate(noise_indices):
                            new_labels[noise_idx] = next_label + i
                        logger.warning("Sub-clustering failed, using individual strategy")
                else:
                    # 只有一個子聚類，所有 noise 點歸為一群
                    for noise_idx in noise_indices:
                        new_labels[noise_idx] = next_label
                    logger.info("All noise points assigned to single cluster")
            else:
                # 只有一個 noise 點
                new_labels[noise_indices[0]] = next_label
                
        elif noise_strategy == 'nearest':
            # 將 noise 點分配給最近的聚類
            if np.any(cluster_labels != -1):
                # 獲取非 noise 點的聚類中心
                unique_labels = np.unique(cluster_labels[cluster_labels != -1])
                cluster_centers = {}
                
                for label in unique_labels:
                    cluster_mask = cluster_labels == label
                    cluster_embeddings = embeddings[cluster_mask]
                    cluster_centers[label] = np.mean(cluster_embeddings, axis=0)
                
                # 為每個 noise 點找到最近的聚類
                for noise_idx in noise_indices:
                    noise_embedding = embeddings[noise_idx]
                    max_similarity = -1
                    nearest_cluster = next_label
                    
                    for label, center in cluster_centers.items():
                        similarity = cosine_similarity([noise_embedding], [center])[0][0]
                        if similarity > max_similarity:
                            max_similarity = similarity
                            nearest_cluster = label
                    
                    new_labels[noise_idx] = nearest_cluster
                
                logger.info(f"Assigned noise points to nearest clusters")
            else:
                # 沒有非 noise 聚類，所有 noise 點自成一群
                for noise_idx in noise_indices:
                    new_labels[noise_idx] = next_label
        
        else:
            raise ValueError(f"Unknown noise strategy: {noise_strategy}")
        
        return new_labels
    
    def _calculate_purity(self, true_labels, predicted_labels):
        """修復的純度計算"""
        
        if len(true_labels) == 0:
            return 0.0
        
        # 過濾掉噪音點 (-1)
        mask = np.array(predicted_labels) != -1
        if not np.any(mask):
            return 0.0
        
        filtered_true = np.array(true_labels)[mask]
        filtered_pred = np.array(predicted_labels)[mask]
        
        clusters = {}
        for i, pred_label in enumerate(filtered_pred):
            if pred_label not in clusters:
                clusters[pred_label] = []
            clusters[pred_label].append(filtered_true[i])
        
        total_correct = 0
        total_samples = len(filtered_true)
        
        for cluster_labels in clusters.values():
            if cluster_labels:  # 確保不是空列表
                most_common = max(set(cluster_labels), key=cluster_labels.count)
                correct_count = cluster_labels.count(most_common)
                total_correct += correct_count
        
        return total_correct / total_samples if total_samples > 0 else 0.0

    def compare_clustering_methods(self, sentences, true_labels, methods=['kmeans', 'hierarchical', 'dbscan']):
        """改進的聚類方法比較"""
        
        logger.info("Comparing different clustering methods...")
        
        results = {}
        embeddings = self.model.encode(sentences)
        n_true_clusters = len(set(true_labels))
        
        # 重新排序方法，將 DBSCAN 放在最後
        methods = [m for m in methods if m != 'dbscan'] + (['dbscan'] if 'dbscan' in methods else [])
        
        for method in methods:
            logger.info(f"Testing {method}...")
            
            try:
                if method == 'dbscan':
                    # DBSCAN 不需要指定群數
                    metrics, predicted_labels = self.evaluate_clustering_metrics(
                        sentences, true_labels, clustering_method=method
                    )
                    
                    # 檢查 DBSCAN 結果是否合理
                    n_pred_clusters = metrics.get('n_predicted_clusters', 0)
                    if n_pred_clusters > len(sentences) // 2:
                        logger.warning(f"DBSCAN produced too many clusters ({n_pred_clusters}), skipping")
                        continue
                        
                else:
                    # 其他方法使用真實群數
                    metrics, predicted_labels = self.evaluate_clustering_metrics(
                        sentences, true_labels, clustering_method=method, n_clusters=n_true_clusters
                    )
                
                results[method] = {
                    'metrics': metrics,
                    'predicted_labels': predicted_labels
                }
                
            except Exception as e:
                logger.warning(f"Failed to run {method}: {e}")
                results[method] = {'error': str(e)}
        
        # 找出最佳方法
        best_method = self._find_best_clustering_method(results)
        logger.info(f"Best clustering method: {best_method}")
        
        return results, best_method

class CustomEvaluator(evaluation.SentenceEvaluator):
    """自定義評估器，用於整合到 sentence-transformers 的訓練流程"""
    
    def __init__(self, sentences, true_labels, name="custom", clustering_method='auto', n_clusters=None, metrics_tracker=None):
        self.sentences = sentences
        self.true_labels = true_labels
        self.name = name
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters or len(set(true_labels))
        self.metrics_tracker = metrics_tracker
        self.current_epoch = 0
        self.current_step = 0
        self.current_loss = 0
        
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        """在訓練過程中被調用"""
        
        evaluator = EnhancedEvaluator(model)
        metrics, predicted_labels = evaluator.comprehensive_evaluation(
            self.sentences, self.true_labels, self.clustering_method, self.n_clusters
        )
        
        ami_score = metrics.get('AMI', 0.0)
        
        # 詳細分析驗證集聚類結果
        if epoch % 2 == 0 or ami_score < 0.1:  # 每2個epoch或AMI過低時分析
            self._analyze_validation_clustering(model, predicted_labels, metrics, epoch, steps)
        
        # 更新指標追蹤器
        if self.metrics_tracker:
            self.metrics_tracker.add_metric(
                epoch=epoch,
                step=steps,
                ami_score=ami_score,
                loss=self.current_loss
            )
            
        # 主要指標 (AMI) 用於模型選擇 - 詳細信息已在callback中顯示
        return ami_score
    
    def _analyze_validation_clustering(self, model, predicted_labels, metrics, epoch, steps):
        """分析驗證集聚類結果，診斷問題"""
        logger.info(f"\n=== Validation Clustering Analysis (Epoch {epoch}, Step {steps}) ===")
        
        # 1. 基本統計信息
        n_true_clusters = len(set(self.true_labels))
        n_pred_clusters = len(set(predicted_labels))
        n_noise_points = sum(1 for label in predicted_labels if label == -1)
        
        logger.info(f"True clusters: {n_true_clusters}, Predicted clusters: {n_pred_clusters}")
        logger.info(f"Noise points: {n_noise_points}/{len(predicted_labels)} ({n_noise_points/len(predicted_labels)*100:.1f}%)")
        
        # 2. 聚類分布分析
        from collections import Counter
        pred_cluster_sizes = Counter(predicted_labels)
        true_cluster_sizes = Counter(self.true_labels)
        
        # Convert numpy types to regular Python types for clean logging
        pred_sizes_clean = {int(k): int(v) for k, v in pred_cluster_sizes.items()}
        true_sizes_clean = {int(k): int(v) for k, v in true_cluster_sizes.most_common(10)}
        
        logger.info(f"Predicted cluster sizes: {dict(sorted(pred_sizes_clean.items()))}")
        logger.info(f"True cluster sizes (top 10): {true_sizes_clean}")
        
        # 3. 嵌入相似性分析
        embeddings = model.encode(self.sentences)
        self._analyze_embedding_similarity(embeddings, epoch)
        
        # 4. 問題診斷
        self._diagnose_clustering_problems(predicted_labels, metrics, embeddings, epoch)
        
        # 5. 每個真實cluster的預測分布
        self._analyze_cluster_confusion(predicted_labels, epoch)
        
        # 6. 保存驗證數據用於分析
        self._save_validation_analysis(embeddings, predicted_labels, metrics, epoch, steps)
        
        logger.info("=== End Analysis ===\n")
    
    def _save_validation_analysis(self, embeddings, predicted_labels, metrics, epoch, steps):
        """保存驗證集分析數據"""
        import pandas as pd
        import numpy as np
        import os
        
        # 創建保存目錄
        save_dir = f"./validation_analysis"
        os.makedirs(save_dir, exist_ok=True)
        
        # 準備數據
        analysis_data = {
            'sentence': self.sentences,
            'true_label': self.true_labels,
            'predicted_label': predicted_labels,
            'epoch': [epoch] * len(self.sentences),
            'step': [steps] * len(self.sentences),
            'ami_score': [metrics.get('AMI', 0.0)] * len(self.sentences)
        }
        
        # 添加嵌入維度
        for i in range(embeddings.shape[1]):
            analysis_data[f'embedding_dim_{i}'] = embeddings[:, i]
        
        # 計算每個句子的相似性統計
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        same_class_similarities = []
        diff_class_similarities = []
        
        for i in range(len(embeddings)):
            same_class_sims = []
            diff_class_sims = []
            
            for j in range(len(embeddings)):
                if i != j:
                    sim = similarity_matrix[i][j]
                    if self.true_labels[i] == self.true_labels[j]:
                        same_class_sims.append(sim)
                    else:
                        diff_class_sims.append(sim)
            
            same_class_similarities.append(np.mean(same_class_sims) if same_class_sims else 0)
            diff_class_similarities.append(np.mean(diff_class_sims) if diff_class_sims else 0)
        
        analysis_data['avg_same_class_similarity'] = same_class_similarities
        analysis_data['avg_diff_class_similarity'] = diff_class_similarities
        analysis_data['similarity_margin'] = [s - d for s, d in zip(same_class_similarities, diff_class_similarities)]
        
        # 創建DataFrame
        df = pd.DataFrame(analysis_data)
        
        # 保存到文件
        epoch_int = int(epoch)  # Convert to int to handle float epochs
        filename = f"validation_analysis_epoch_{epoch_int:02d}_step_{steps}.csv"
        filepath = os.path.join(save_dir, filename)
        df.to_csv(filepath, index=False)
        
        # 保存嵌入矩陣
        embedding_filename = f"embeddings_epoch_{epoch_int:02d}_step_{steps}.npy"
        embedding_filepath = os.path.join(save_dir, embedding_filename)
        np.save(embedding_filepath, embeddings)
        
        # 保存相似度矩陣
        similarity_filename = f"similarity_matrix_epoch_{epoch_int:02d}_step_{steps}.npy"
        similarity_filepath = os.path.join(save_dir, similarity_filename)
        np.save(similarity_filepath, similarity_matrix)
        
        # 保存聚類指標
        metrics_data = {
            'epoch': float(epoch),  # Ensure JSON serializable
            'step': int(steps),
            'ami_score': float(metrics.get('AMI', 0.0)),
            'n_true_clusters': len(set(self.true_labels)),
            'n_pred_clusters': len(set(predicted_labels)),
            'n_noise_points': sum(1 for label in predicted_labels if label == -1),
            'clustering_method': self.clustering_method
        }
        
        # 添加其他指標 - ensure JSON serializable
        for key, value in metrics.items():
            if key not in metrics_data:
                if isinstance(value, (np.integer, np.floating)):
                    metrics_data[key] = float(value)
                elif isinstance(value, np.ndarray):
                    metrics_data[key] = value.tolist()
                else:
                    metrics_data[key] = value
        
        metrics_filename = f"clustering_metrics_epoch_{epoch_int:02d}_step_{steps}.json"
        metrics_filepath = os.path.join(save_dir, metrics_filename)
        
        import json
        with open(metrics_filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"💾 Validation analysis saved:")
        logger.info(f"   - Data: {filepath}")
        logger.info(f"   - Embeddings: {embedding_filepath}")
        logger.info(f"   - Similarity matrix: {similarity_filepath}")
        logger.info(f"   - Metrics: {metrics_filepath}")
    
    def _analyze_embedding_similarity(self, embeddings, epoch):
        """分析嵌入相似性"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 計算同類和異類相似性
        same_class_similarities = []
        diff_class_similarities = []
        
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if self.true_labels[i] == self.true_labels[j]:
                    same_class_similarities.append(sim)
                else:
                    diff_class_similarities.append(sim)
        
        if same_class_similarities and diff_class_similarities:
            same_class_mean = np.mean(same_class_similarities)
            same_class_std = np.std(same_class_similarities)
            diff_class_mean = np.mean(diff_class_similarities)
            diff_class_std = np.std(diff_class_similarities)
            
            logger.info(f"Same_Class_Avg_Similarity: {same_class_mean:.4f}, Same_Class_Std: {same_class_std:.4f}")
            logger.info(f"Diff_Class_Avg_Similarity: {diff_class_mean:.4f}, Diff_Class_Std: {diff_class_std:.4f}")
            logger.info(f"Similarity_Margin: {same_class_mean - diff_class_mean:.4f}")
            
            # 檢查是否有嵌入坍縮
            if same_class_std < 0.01:
                logger.warning("⚠️  Embedding collapse detected! Same-class embeddings are too similar")
            if same_class_mean - diff_class_mean < 0.1:
                logger.warning("⚠️  Poor separation! Same-class and different-class similarities are too close")
    
    def _diagnose_clustering_problems(self, predicted_labels, metrics, embeddings, epoch):
        """診斷聚類問題"""
        logger.info("🔍 Clustering Problem Diagnosis:")
        
        n_pred_clusters = len(set(predicted_labels))
        n_true_clusters = len(set(self.true_labels))
        n_noise = sum(1 for label in predicted_labels if label == -1)
        
        # 1. 聚類數量問題
        if n_pred_clusters == 1:
            logger.warning("❌ Problem: Only 1 cluster predicted (clustering collapse)")
            logger.info("   Possible causes: eps too small, all embeddings too similar")
        elif n_pred_clusters > n_true_clusters * 2:
            logger.warning("❌ Problem: Too many clusters predicted (over-fragmentation)")
            logger.info("   Possible causes: eps too large, min_samples too small")
        elif n_pred_clusters < n_true_clusters * 0.5:
            logger.warning("❌ Problem: Too few clusters predicted (under-clustering)")
            logger.info("   Possible causes: eps too small, min_samples too large")
        
        # 2. 噪音點問題
        if n_noise > len(predicted_labels) * 0.3:
            logger.warning("❌ Problem: Too many noise points")
            logger.info("   Possible causes: eps too small, min_samples too large")
        
        # 3. 性能問題
        ami_score = metrics.get('AMI', 0.0)
        if ami_score < 0.1:
            logger.warning("❌ Problem: Very low AMI score")
            logger.info("   Possible causes: poor embedding quality, wrong clustering parameters")
        
        # 4. 建議解決方案
        logger.info("💡 Suggestions:")
        if n_pred_clusters == 1:
            logger.info("   - Reduce eps parameter for DBSCAN")
            logger.info("   - Check if learning rate is too high (causing embedding collapse)")
        elif n_noise > len(predicted_labels) * 0.3:
            logger.info("   - Increase eps parameter")
            logger.info("   - Decrease min_samples parameter")
        
    def _analyze_cluster_confusion(self, predicted_labels, epoch):
        """分析每個真實cluster的預測分布"""
        from collections import defaultdict, Counter
        
        # 為每個真實cluster統計預測分布
        true_to_pred_mapping = defaultdict(list)
        for i, (true_label, pred_label) in enumerate(zip(self.true_labels, predicted_labels)):
            true_to_pred_mapping[true_label].append(pred_label)
        
        # 找出問題最嚴重的clusters
        problem_clusters = []
        for true_label, pred_labels in true_to_pred_mapping.items():
            pred_counter = Counter(pred_labels)
            most_common_pred = pred_counter.most_common(1)[0]
            purity = most_common_pred[1] / len(pred_labels)
            
            if purity < 0.5:  # 純度低於50%
                problem_clusters.append((true_label, purity, pred_counter))
        
        if problem_clusters:
            logger.info(f"🔍 Problem clusters (purity < 50%):")
            for true_label, purity, pred_counter in sorted(problem_clusters, key=lambda x: x[1]):
                logger.info(f"   True cluster {true_label}: purity={purity:.2f}, predictions={dict(pred_counter)}")
                
    def set_current_loss(self, loss):
        """設置當前損失值"""
        self.current_loss = loss

class MetricsTracker:
    """追蹤和視覺化訓練指標 (增強版，可自訂圖表外觀)"""

    def __init__(self,
                 save_path='./training_metrics.xlsx',
                 y_lim_ami=(0.3, 1.05),
                 y_lim_loss=None,  # Set to None for dynamic scaling
                 x_tick_interval=None,
                 y_tick_interval_ami=0.05,
                 y_tick_interval_loss=None,  # Set to None for dynamic scaling
                 loss_scale_margin=0.1):
        """
        初始化指標追蹤器。

        Args:
            save_path (str): Excel 指標檔案的儲存路徑。
            y_lim_ami (tuple): AMI 分數圖 (上方圖表) 的 Y 軸範圍。
            y_lim_loss (tuple or None): 損失圖 (下方圖表) 的 Y 軸範圍。None 表示動態縮放。
            x_tick_interval (float, optional): X 軸主要刻度的間距。預設為自動。
            y_tick_interval_ami (float, optional): AMI 分數圖 Y 軸主要刻度的間距。
            y_tick_interval_loss (float, optional): 損失圖 Y 軸主要刻度的間距。None 表示動態設定。
            loss_scale_margin (float): 動態縮放時在最小/最大值之外的邊距比例。
        """
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
            self.fig.suptitle('Training Progress', fontsize=16)
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
            # 我們仍然繪製 loss，即使它可能是佔位符
            losses = self.metrics_data['loss']
            
            # --- 上方圖表: AMI Score ---
            ax1.plot(steps, ami_scores, 'b-', marker='o', linewidth=2, markersize=4, label='AMI Score')
            
            # Add score annotations above each point
            for i, (step, score) in enumerate(zip(steps, ami_scores)):
                ax1.annotate(f'{score:.2f}', 
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
                ax1.yaxis.set_major_locator(mticker.MultipleLocator(self.y_tick_interval_ami)) # 設定 Y 軸刻度間距
            
            # --- 下方圖表: Training Loss ---
            ax2.plot(steps, losses, 'r-', marker='s', linewidth=2, markersize=4, label='Training Loss')
            
            # Add loss annotations above each point
            for step, loss in zip(steps, losses):
                if loss is not None and loss != 0:
                    ax2.annotate(f'{loss:.3f}', 
                               xy=(step, loss), 
                               xytext=(0, 10),  # 10 points above the point
                               textcoords='offset points',
                               ha='center', va='bottom',
                               fontsize=8, 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
            
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss (Dynamic Scale)')
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
                    ax.xaxis.set_major_locator(mticker.MultipleLocator(self.x_tick_interval)) # 設定 X 軸刻度間距
            
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # 調整佈局以容納主標題
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

def main():
    """增強版的主訓練流程"""
    
    # 配置保持不變
    PATH2DOWN = os.path.expanduser('~/111111ForDownload/')
    # data_path = PATH2DOWN + 'csv_TLR001_22nm_with_init_cluster_with_gt_class_name.csv'
    data_path = PATH2DOWN + 'All_TLR__28_LOGIC_clustering_optimized_20250822_post_process_manualLabel.csv'
    
    config = {
        'model_name': 'sentence-transformers/all-mpnet-base-v2',
        'num_epochs': 10,
        'learning_rate': 1e-5,  # Reduced from 6e-5 to prevent embedding collapse
        'enhanced_triplet': True,
        'difficulty_strategy': 'mixed',
        'loss_type': 'cached_multiple_negatives',  # Options: 'triplet', 'circle','multi_similarity', 'multiple_negatives', 'cached_multiple_negatives', 'cosine'
        'output_path': f'./rule-embedder-{datetime.now().strftime("%m%d")}_trained_MPnet_CachedMN',
        'HNM_model_path':'rule-embedder-0722_trained_MPnet_CachedMN'
    }
    
    try:
        # 使用增強版訓練器
        trainer = EmbeddingTrainerEnhanced(model_name=config['model_name'])
        trainer.load_model()
        
        train_data, val_data_triplets, val_sentences, val_labels = trainer.prepare_data(
            path4ruleGT=data_path,
            model_path=config['HNM_model_path'],
        )
        
        # 驗證檢查
        if len(set(val_labels)) < 2:
            logger.error(f"FATAL: Validation set only has {len(set(val_labels))} unique labels. Check data generation.")
            raise ValueError("Insufficient class diversity in validation set.")
        else:
            logger.info(f"Verification successful: val_labels contains {len(set(val_labels))} unique classes.")

        # 配置訓練
        train_dataloader, val_dataloader, train_loss, evaluator = trainer.configure_training_enhanced(
            train_data=train_data,
            val_data=val_data_triplets,
            val_sentences=val_sentences,
            val_labels=val_labels,
            clustering_method='dbscan',
            loss_type=config['loss_type']
        )
        
        # 執行訓練
        trainer.train(
            train_dataloader=train_dataloader,
            train_loss=train_loss,
            evaluator=evaluator,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            output_path=config['output_path']
        )
        
        # 增強版評估
        trained_model, metrics, predicted_labels = trainer.evaluate_model_enhanced(
            model_path=config['output_path']
        )
        
        return trained_model, metrics
        
    except Exception as e:
        logger.error(f"Enhanced training pipeline failed: {e}")
        raise
    

if __name__ == "__main__":
    trained_model, metrics = main()
    logger.info("Enhanced training completed successfully!")
    