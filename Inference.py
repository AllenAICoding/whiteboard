import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
import logging
from typing import List, Dict, Tuple, Optional, Union
import pickle
from collections import defaultdict, Counter
import warnings
import glob
from pathlib import Path
import re
from datetime import datetime

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    input("HDBSCAN not available. Install with: pip install hdbscan")
    
warnings.filterwarnings('ignore')

# 設置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_model_config(model_path, best_params, best_ami, method='dbscan'):
    """Save or update model configuration with best parameters"""
    try:
        # Get model info
        model_info = get_model_info(model_path)
        model_name = model_info['name']
        
        # Path to config file (in parent directory)
        config_path = Path(__file__).parent.parent / "model_configs.json"
        
        # Load existing config or create new one
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {
                "models": {},
                "default_model": None
            }
        
        # Update model configuration
        config["models"][model_name] = {
            "path": model_path,
            "best_params": best_params,
            "best_ami": float(best_ami),
            "date_created": model_info['created_time'],
            "date_evaluated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "description": f"Auto-evaluated with {method.upper()}, AMI={best_ami:.4f}"
        }
        
        # Set as default if it's the best we've seen so far
        if config.get("default_model") is None or best_ami > 0.85:
            config["default_model"] = model_name
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved model config for {model_name} to {config_path}")
        logger.info(f"  - Best params: {best_params}")
        logger.info(f"  - Best AMI: {best_ami:.4f}")
        
    except Exception as e:
        logger.warning(f"Failed to save model config: {e}")
        # Don't fail the evaluation if config saving fails

def find_latest_model(base_dir: str = "./", model_pattern: str = "rule-embedder-*") -> str:
    """
    自動找到最新的模型路徑
    
    Args:
        base_dir: 搜索基礎目錄
        model_pattern: 模型目錄名稱模式
        
    Returns:
        最新模型的路徑
    """
    # 搜索所有匹配的模型目錄
    model_dirs = glob.glob(os.path.join(base_dir, model_pattern))
    
    if not model_dirs:
        raise FileNotFoundError(f"No models found with pattern '{model_pattern}' in directory '{base_dir}'")
    
    # 過濾出有效的模型目錄（包含必要的文件）
    valid_models = []
    for model_dir in model_dirs:
        if os.path.isdir(model_dir):
            # 檢查是否包含必要的模型文件
            required_files = ['config.json', 'model.safetensors', 'modules.json']
            if all(os.path.exists(os.path.join(model_dir, f)) for f in required_files):
                valid_models.append(model_dir)
    
    if not valid_models:
        raise FileNotFoundError(f"No valid models found. Models must contain: {required_files}")
    
    # 根據修改時間排序，找到最新的
    latest_model = max(valid_models, key=os.path.getmtime)
    
    logger.info(f"Found {len(valid_models)} valid models:")
    for model in sorted(valid_models, key=os.path.getmtime, reverse=True):
        mod_time = datetime.fromtimestamp(os.path.getmtime(model)).strftime('%Y-%m-%d %H:%M:%S')
        marker = " (LATEST)" if model == latest_model else ""
        logger.info(f"  {model} - {mod_time}{marker}")
    
    return latest_model

def find_latest_model_by_name(base_dir: str = "./", model_pattern: str = "rule-embedder-*") -> str:
    """
    根據名稱中的日期模式找到最新模型（備用方法）
    
    Args:
        base_dir: 搜索基礎目錄
        model_pattern: 模型目錄名稱模式
        
    Returns:
        最新模型的路徑
    """
    model_dirs = glob.glob(os.path.join(base_dir, model_pattern))
    
    if not model_dirs:
        raise FileNotFoundError(f"No models found with pattern '{model_pattern}' in directory '{base_dir}'")
    
    # 提取日期模式並排序
    date_pattern = r'(\d{4})(\d{2})(\d{2})'  # YYYYMMDD
    model_with_dates = []
    
    for model_dir in model_dirs:
        if os.path.isdir(model_dir):
            # 檢查模型有效性
            required_files = ['config.json', 'model.safetensors', 'modules.json']
            if all(os.path.exists(os.path.join(model_dir, f)) for f in required_files):
                # 提取日期
                basename = os.path.basename(model_dir)
                date_match = re.search(date_pattern, basename)
                if date_match:
                    date_str = date_match.group(0)
                    model_with_dates.append((model_dir, date_str, basename))
                else:
                    # 如果沒有日期，使用文件修改時間
                    model_with_dates.append((model_dir, "00000000", basename))
    
    if not model_with_dates:
        raise FileNotFoundError("No valid models found")
    
    # 按日期排序，取最新的
    latest_model = max(model_with_dates, key=lambda x: x[1])
    
    logger.info(f"Found {len(model_with_dates)} valid models:")
    for model_dir, date_str, basename in sorted(model_with_dates, key=lambda x: x[1], reverse=True):
        marker = " (LATEST)" if model_dir == latest_model[0] else ""
        logger.info(f"  {basename} - {date_str}{marker}")
    
    return latest_model[0]

def get_model_info(model_path: str) -> Dict:
    """
    獲取模型信息
    
    Args:
        model_path: 模型路徑
        
    Returns:
        模型信息字典
    """
    info = {
        'path': model_path,
        'name': os.path.basename(model_path),
        'created_time': datetime.fromtimestamp(os.path.getctime(model_path)).strftime('%Y-%m-%d %H:%M:%S'),
        'modified_time': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 嘗試讀取配置文件
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                info['config'] = config
        except:
            pass
    
    # 檢查是否有訓練記錄
    training_log_path = os.path.join(model_path, 'training_progress.png')
    if os.path.exists(training_log_path):
        info['has_training_log'] = True
    
    eval_path = os.path.join(model_path, 'eval')
    if os.path.exists(eval_path):
        info['has_evaluation'] = True
    
    return info

class RuleEmbeddingInference:
    """規則嵌入推理器"""

    def __init__(self, model_path: str, device: str = 'auto', config_path: str = None):
        """
        初始化推理器

        Args:
            model_path: 訓練好的模型路径
            device: 使用的設備 ('auto', 'cpu', 'cuda')
            config_path: 配置文件路徑，如果為None則使用默認路徑
        """
        self.model_path = model_path
        self.model = None
        self.device = device
        self.embeddings_cache = {}
        self.similarity_cache = {}

        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'inference_config.json')
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> dict:
        """
        加載配置文件

        Args:
            config_path: 配置文件路徑

        Returns:
            配置字典
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config
            else:
                logger.warning(f"Config file not found: {config_path}, using default values")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using default values")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """
        獲取默認配置

        Returns:
            默認配置字典
        """
        return {
            "model": {"device": "auto", "batch_size": 32, "show_progress": True},
            "clustering": {
                "default_method": "dbscan",
                "optimize_params": True,
                "kmeans": {"n_clusters": None, "random_state": 42, "n_init": 10},
                "dbscan": {"eps": 0.3, "min_samples": 2, "metric": "cosine"},
                "hdbscan": {"min_cluster_size": 5, "min_samples": None, "metric": "euclidean"},
                "hierarchical": {"n_clusters": None, "linkage": "ward"}
            },
            "similarity": {"default_metric": "cosine", "top_k": 10, "threshold": 0.5},
            "noise_handling": {"strategy": "individual"},
            "optimization": {
                "dbscan": {"eps_range": [0.05, 0.5], "min_samples_range": [2, 5]},
                "hdbscan": {"min_cluster_size_range": [2, 20], "min_samples_range": [1, 10]}
            }
        }
        
    def load_model(self):
        """加載訓練好的模型"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = SentenceTransformer(self.model_path)
            
            if self.device == 'auto':
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            logger.info(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def encode_rules(self, rules: List[str], batch_size: int = None,
                    show_progress: bool = None) -> np.ndarray:
        """
        對規則進行編碼
        
        Args:
            rules: 規則描述列表
            batch_size: 批次大小
            show_progress: 是否顯示進度
            
        Returns:
            規則嵌入向量
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Use config defaults if not specified
        if batch_size is None:
            batch_size = self.config.get('model', {}).get('batch_size', 32)
        if show_progress is None:
            show_progress = self.config.get('model', {}).get('show_progress', True)

        logger.info(f"Encoding {len(rules)} rules...")

        # 檢查緩存
        cache_key = hash(tuple(rules))
        if cache_key in self.embeddings_cache:
            logger.info("Using cached embeddings")
            return self.embeddings_cache[cache_key]

        # 批次編碼
        embeddings = self.model.encode(
            rules,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # 緩存結果
        self.embeddings_cache[cache_key] = embeddings
        
        logger.info(f"Encoding completed. Shape: {embeddings.shape}")
        return embeddings
    
    def compute_similarity_matrix(self, embeddings: np.ndarray, 
                                 metric: str = 'cosine') -> np.ndarray:
        """
        計算相似度矩陣
        
        Args:
            embeddings: 嵌入向量
            metric: 相似度指標 ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            相似度矩陣
        """
        if metric == 'cosine':
            return cosine_similarity(embeddings)
        elif metric == 'euclidean':
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(embeddings)
            # 轉換為相似度 (距離越小，相似度越高)
            return 1 / (1 + distances)
        elif metric == 'manhattan':
            from sklearn.metrics.pairwise import manhattan_distances
            distances = manhattan_distances(embeddings)
            return 1 / (1 + distances)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def find_similar_rules(self, query_rule: str, candidate_rules: List[str],
                          top_k: int = None, threshold: float = None) -> List[Dict]:
        """
        查找相似規則
        
        Args:
            query_rule: 查詢規則
            candidate_rules: 候選規則列表
            top_k: 返回前k個最相似的規則
            threshold: 相似度閾值
            
        Returns:
            相似規則列表，包含規則文本和相似度分數
        """
        # Use config defaults if not specified
        if top_k is None:
            top_k = self.config.get('similarity', {}).get('top_k', 10)
        if threshold is None:
            threshold = self.config.get('similarity', {}).get('threshold', 0.5)

        # 編碼所有規則
        all_rules = [query_rule] + candidate_rules
        embeddings = self.encode_rules(all_rules)

        # 計算相似度
        query_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]

        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        # 排序並過濾
        similar_rules = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                similar_rules.append({
                    'rule': candidate_rules[i],
                    'similarity': float(similarity),
                    'index': i
                })
        
        # 按相似度排序
        similar_rules.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_rules[:top_k]
        
    def _handle_dbscan_noise(self, cluster_labels: np.ndarray, embeddings: np.ndarray,
                            noise_strategy: str = None, **kwargs) -> np.ndarray:
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
        # Use config defaults if not specified
        if noise_strategy is None:
            noise_strategy = self.config.get('noise_handling', {}).get('strategy', 'individual')

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
                sub_cluster_divisor = self.config.get('noise_handling', {}).get('sub_cluster_divisor', 3)
                sub_cluster_max = self.config.get('noise_handling', {}).get('sub_cluster_max', 5)
                n_sub_clusters = min(max(1, len(noise_indices) // sub_cluster_divisor), sub_cluster_max)
                
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

    # 在 batch_inference 方法中添加 noise_strategy 參數
    def batch_inference(self, rules_file: str, output_file: str, 
                    clustering_method: str = 'kmeans', n_clusters: int = None,
                    noise_strategy: str = 'individual') -> Dict:
        """
        批次推理
        
        Args:
            rules_file: 規則文件路徑 (CSV)
            output_file: 輸出文件路徑
            clustering_method: 聚類方法
            n_clusters: 聚類數量
            noise_strategy: DBSCAN noise 處理策略
            
        Returns:
            推理結果
        """
        logger.info(f"Starting batch inference on {rules_file}")
        
        # 讀取數據
        df = pd.read_csv(rules_file)
        
        # 假設CSV有'description'列
        if 'Description' not in df.columns:
            raise ValueError("CSV file must contain 'Description' column")
        
        rules = df['Description'].tolist()
        
        # 執行聚類
        clustering_result = self.cluster_rules(
            rules, 
            method=clustering_method, 
            n_clusters=n_clusters,
            noise_strategy=noise_strategy  # 傳遞 noise_strategy 參數
        )
        
        # 添加聚類結果到DataFrame
        df['predicted_cluster'] = clustering_result['cluster_labels']
        
        # 計算相似度矩陣
        similarity_matrix = self.compute_similarity_matrix(clustering_result['embeddings'])
        
        # 保存結果
        output_data = {
            'rules_with_clusters': df.to_dict('records'),
            'cluster_stats': self._convert_numpy_types(clustering_result['cluster_stats']),
            'similarity_matrix': similarity_matrix.tolist(),
            'clustering_method': clustering_method,
            'n_clusters': clustering_result['n_clusters'],
            'noise_strategy': noise_strategy if clustering_method == 'dbscan' else None
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch inference completed. Results saved to {output_file}")
        return output_data, df
        
    def _find_optimal_clusters(self, embeddings: np.ndarray, max_k: int = 20) -> int:
        """使用肘部法則和輪廓分析找到最佳聚類數"""
        if max_k <= 1:
            return 1
            
        n_samples = len(embeddings)
        # 調整最大聚類數，但不超過樣本數的1/3
        max_k = min(max_k, n_samples // 3, 30)
        
        if max_k <= 1:
            return 1
            
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)  # 從2開始，因為輪廓分析需要至少2個聚類
        
        from sklearn.metrics import silhouette_score
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            inertias.append(kmeans.inertia_)
            
            # 計算輪廓係數
            if len(set(cluster_labels)) > 1:  # 確保有多個聚類
                sil_score = silhouette_score(embeddings, cluster_labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # 方法1: 肘部法則
        elbow_k = 2
        if len(inertias) >= 3:
            # 計算二階導數來找到肘部
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            
            if len(second_diffs) > 0:
                # 找到二階導數最大的點
                elbow_k = np.argmax(second_diffs) + 3  # +3因為從k=2開始
        
        # 方法2: 輪廓分析
        silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        # 方法3: 結合兩種方法
        # 如果輪廓係數在最優點附近都很高，選擇較大的k
        best_sil_score = max(silhouette_scores)
        good_k_candidates = []
        
        for i, score in enumerate(silhouette_scores):
            if score >= best_sil_score * 0.95:  # 在最優值的95%以內
                good_k_candidates.append(k_range[i])
        
        # 在好的候選中選擇
        if good_k_candidates:
            # 如果elbow_k在候選中，選擇它
            if elbow_k in good_k_candidates:
                optimal_k = elbow_k
            else:
                # 否則選擇輪廓係數最高的
                optimal_k = silhouette_k
        else:
            optimal_k = silhouette_k
        
        logger.info(f"Optimal cluster analysis: Elbow method suggests {elbow_k}, "
                   f"Silhouette analysis suggests {silhouette_k}, "
                   f"Final choice: {optimal_k}")
        
        return optimal_k
    
    def visualize_embeddings(self, rules: List[str], labels: List[int] = None,
                        method: str = 'tsne', save_path: str = None,
                        title: str = "Rule Embeddings Visualization") -> None:
        """
        可視化嵌入向量
        
        Args:
            rules: 規則列表
            labels: 標籤列表 (可選)
            method: 降維方法 ('tsne', 'pca')
            save_path: 保存路徑
            title: 圖表標題
        """
        logger.info(f"Visualizing embeddings using {method}")
        
        # 編碼規則
        embeddings = self.encode_rules(rules)
        
        # 降維
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(rules)-1))
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unsupported visualization method: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # 創建圖表
        plt.figure(figsize=(12, 8))
        
        if labels is not None:
            # 處理標籤數據，統一類型和處理缺失值
            processed_labels = self._process_labels_for_visualization(labels)
            
            unique_labels = sorted(set(processed_labels))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = np.array(processed_labels) == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                        c=[colors[i]], label=f'Group {label}', alpha=0.7, s=50)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=50)
        
        plt.title(title)
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()

    def _process_labels_for_visualization(self, labels: List[Union[int, str, float]]) -> List[str]:
        """
        處理標籤數據以便可視化
        
        Args:
            labels: 原始標籤列表
            
        Returns:
            處理後的標籤列表
        """
        processed_labels = []
        
        for label in labels:
            # 處理 NaN 值
            if pd.isna(label):
                processed_labels.append('Unknown')
            # 處理 None 值
            elif label is None:
                processed_labels.append('Unknown')
            # 將所有標籤轉換為字符串
            else:
                processed_labels.append(str(label))
        
        return processed_labels

    def analyze_rule_groups(self, rules: List[str], labels: List[int]) -> Dict:
        """
        分析規則組 (修復版本)
        
        Args:
            rules: 規則列表
            labels: 標籤列表
            
        Returns:
            分析結果字典
        """
        logger.info("Analyzing rule groups...")
        
        # 處理標籤
        processed_labels = self._process_labels_for_visualization(labels)
        
        # 編碼規則
        embeddings = self.encode_rules(rules)
        
        # 按組分析
        groups = defaultdict(list)
        for i, label in enumerate(processed_labels):
            groups[label].append({
                'rule': rules[i],
                'embedding': embeddings[i],
                'index': i
            })
        
        analysis = {}
        for label, items in groups.items():
            group_embeddings = np.array([item['embedding'] for item in items])
            group_rules = [item['rule'] for item in items]
            
            # 計算組內相似度
            if len(group_embeddings) > 1:
                similarities = cosine_similarity(group_embeddings)
                # 去除對角線
                mask = np.ones(similarities.shape, dtype=bool)
                np.fill_diagonal(mask, 0)
                intra_similarity = np.mean(similarities[mask])
                similarity_std = np.std(similarities[mask])
            else:
                intra_similarity = 1.0
                similarity_std = 0.0
            
            # 計算組中心
            centroid = np.mean(group_embeddings, axis=0)
            
            # 找到最代表性的規則（離中心最近的）
            centroid_similarities = cosine_similarity([centroid], group_embeddings)[0]
            most_representative_idx = np.argmax(centroid_similarities)
            
            analysis[label] = {
                'size': len(items),
                'rules': group_rules,
                'intra_similarity': float(intra_similarity),
                'similarity_std': float(similarity_std),
                'centroid': centroid.tolist(),  # Convert to list for JSON serialization
                'most_representative_rule': group_rules[most_representative_idx],
                'representative_score': float(centroid_similarities[most_representative_idx])
            }
        
        # 計算組間相似度
        inter_group_similarities = {}
        group_labels = list(groups.keys())
        
        for i, label1 in enumerate(group_labels):
            for j, label2 in enumerate(group_labels):
                if i < j:
                    centroid1 = np.array(analysis[label1]['centroid'])
                    centroid2 = np.array(analysis[label2]['centroid'])
                    similarity = cosine_similarity([centroid1], [centroid2])[0][0]
                    inter_group_similarities[f"{label1}-{label2}"] = float(similarity)
        
        result = {
            'group_analysis': analysis,
            'inter_group_similarities': inter_group_similarities,
            'total_groups': len(groups),
            'total_rules': len(rules)
        }
        
        logger.info(f"Analysis completed for {len(groups)} groups")
        return result

    def evaluate_clustering(self, rules: List[str], true_labels: List[int], 
                        predicted_labels: List[int]) -> Dict:
        """
        評估聚類效果 (修復版本)
        
        Args:
            rules: 規則列表
            true_labels: 真實標籤
            predicted_labels: 預測標籤
            
        Returns:
            評估指標字典
        """
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score,adjusted_mutual_info_score
        from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
        
        # 處理標籤
        processed_true_labels = self._process_labels_for_evaluation(true_labels)
        processed_pred_labels = self._process_labels_for_evaluation(predicted_labels)
        
        # 計算各種評估指標
        ari = adjusted_rand_score(processed_true_labels, processed_pred_labels)
        nmi = normalized_mutual_info_score(processed_true_labels, processed_pred_labels)
        ami = adjusted_mutual_info_score(processed_true_labels, processed_pred_labels)
        homogeneity = homogeneity_score(processed_true_labels, processed_pred_labels)
        completeness = completeness_score(processed_true_labels, processed_pred_labels)
        v_measure = v_measure_score(processed_true_labels, processed_pred_labels)
        
        # 計算純度
        purity = self._calculate_purity(processed_true_labels, processed_pred_labels)
        
        metrics = {
            'adjusted_rand_score': float(ari),
            'normalized_mutual_info': float(nmi),
            'adjusted_mutual_info': float(ami),
            'homogeneity': float(homogeneity),
            'completeness': float(completeness),
            'v_measure': float(v_measure),
            'purity': float(purity)
        }
        
        logger.info("Clustering evaluation metrics:")
        for metric, value in metrics.items():
            if metric == 'adjusted_mutual_info':
                logger.info("="*50)
                logger.info(f"  {metric}: {value:.4f}")
                logger.info("="*50)
            else:
                logger.info(f"  {metric}: {value:.4f}")
        
        return metrics

    def _process_labels_for_evaluation(self, labels: List[Union[int, str, float]]) -> List[int]:
        """
        處理標籤數據以便評估
        
        Args:
            labels: 原始標籤列表
            
        Returns:
            處理後的整數標籤列表
        """
        # 將標籤轉換為字符串並建立映射
        str_labels = []
        for label in labels:
            if pd.isna(label) or label is None:
                str_labels.append('Unknown')
                input('there is a None or NaN label, please check your data')
            else:
                str_labels.append(str(label))
        
        # 創建標籤到整數的映射
        unique_labels = sorted(set(str_labels))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        
        # 轉換為整數標籤
        int_labels = [label_to_int[label] for label in str_labels]
        
        return int_labels

    def _calculate_purity(self, true_labels: List[int], predicted_labels: List[int]) -> float:
        """計算聚類純度 (修復版本)"""
        total_samples = len(true_labels)
        if total_samples == 0:
            return 0.0
            
        cluster_label_counts = defaultdict(lambda: defaultdict(int))
        
        for true_label, pred_label in zip(true_labels, predicted_labels):
            cluster_label_counts[pred_label][true_label] += 1
        
        correct_assignments = 0
        for pred_label, true_label_counts in cluster_label_counts.items():
            # 每個聚類中最多的真實標籤數量
            max_count = max(true_label_counts.values()) if true_label_counts else 0
            correct_assignments += max_count
        
        return correct_assignments / total_samples

    def _convert_numpy_types(self, obj):
        """
        遞歸地將NumPy類型轉換為Python原生類型，以便JSON序列化
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
            
    def save_model_artifacts(self, save_dir: str) -> None:
            """保存模型相關文件"""
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存緩存
            cache_file = os.path.join(save_dir, 'embeddings_cache.pkl')
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'embeddings_cache': self.embeddings_cache,
                    'similarity_cache': self.similarity_cache
                }, f)
            
            # 保存配置
            config = {
                'model_path': self.model_path,
                'device': self.device,
                'cache_size': len(self.embeddings_cache)
            }
            
            config_file = os.path.join(save_dir, 'inference_config.json')
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Model artifacts saved to {save_dir}")

    def cluster_rules(self, rules: List[str], method: str = None,
                    n_clusters: int = None, optimize_params: bool = None, **kwargs) -> Dict:
        """
        對規則進行聚類，支持DBSCAN和HDBSCAN的參數優化

        Args:
            rules: 規則列表
            method: 聚類方法 ('kmeans', 'dbscan', 'hdbscan', 'hierarchical')
            n_clusters: 聚類數量 (適用於kmeans和hierarchical)
            optimize_params: 是否優化參數
            **kwargs: 其他聚類參數

        Returns:
            聚類結果字典
        """
        # Use config defaults if not specified
        if method is None:
            method = self.config.get('clustering', {}).get('default_method', 'dbscan')
        if optimize_params is None:
            optimize_params = self.config.get('clustering', {}).get('optimize_params', True)

        logger.info(f"Clustering {len(rules)} rules using {method}")

        # 編碼規則
        embeddings = self.encode_rules(rules)

        # 選擇聚類算法
        if method == 'kmeans':
            kmeans_config = self.config.get('clustering', {}).get('kmeans', {})
            if n_clusters is None:
                n_clusters = kmeans_config.get('n_clusters')
                if n_clusters is None:
                    # 自動確定最佳聚類數
                    n_clusters = self._find_optimal_clusters(embeddings, max_k=len(rules)//2)

            random_state = kmeans_config.get('random_state', 42)
            n_init = kmeans_config.get('n_init', 10)
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
            
        elif method == 'dbscan':
            dbscan_config = self.config.get('clustering', {}).get('dbscan', {})
            min_samples = kwargs.get('min_samples', dbscan_config.get('min_samples', 2))
            metric = kwargs.get('metric', dbscan_config.get('metric', 'cosine'))

            if optimize_params and 'eps' not in kwargs:
                # 使用AMI優化eps
                eps_range = kwargs.get('eps_range', tuple(dbscan_config.get('eps_range', [0.1, 1.0])))
                n_eps_values = kwargs.get('n_eps_values', dbscan_config.get('n_eps_values', 20))
                reference_method = kwargs.get('reference_method', dbscan_config.get('reference_method', 'kmeans'))

                eps_optimization = self.find_optimal_eps_with_ami(
                    embeddings,
                    min_samples=min_samples,
                    eps_range=eps_range,
                    n_eps_values=n_eps_values,
                    reference_method=reference_method
                )

                optimal_eps = eps_optimization['best_eps']
                logger.info(f"Optimal eps found: {optimal_eps:.4f} (AMI: {eps_optimization['best_ami']:.4f})")

                # 也嘗試膝蓋法作為參考
                knee_eps = self.find_optimal_eps_knee_method(embeddings, min_samples)
                logger.info(f"Knee method suggests eps: {knee_eps:.4f}")

                eps = optimal_eps
            else:
                eps = kwargs.get('eps', dbscan_config.get('eps', 0.3))

            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            
        elif method == 'hdbscan':
            if not HDBSCAN_AVAILABLE:
                raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")

            hdbscan_config = self.config.get('clustering', {}).get('hdbscan', {})

            if optimize_params and ('min_cluster_size' not in kwargs or 'min_samples' not in kwargs):
                # 使用AMI優化HDBSCAN參數
                min_cluster_size_range = kwargs.get('min_cluster_size_range',
                                                   tuple(hdbscan_config.get('min_cluster_size_range', [2, 20])))
                min_samples_range = kwargs.get('min_samples_range',
                                              tuple(hdbscan_config.get('min_samples_range', [1, 10])))
                reference_method = kwargs.get('reference_method', hdbscan_config.get('reference_method', 'kmeans'))

                hdbscan_optimization = self.find_optimal_hdbscan_params(
                    embeddings,
                    min_cluster_size_range=min_cluster_size_range,
                    min_samples_range=min_samples_range,
                    reference_method=reference_method
                )

                optimal_params = hdbscan_optimization['best_params']
                logger.info(f"Optimal HDBSCAN params: {optimal_params} "
                        f"(AMI: {hdbscan_optimization['best_ami']:.4f})")

                min_cluster_size = optimal_params['min_cluster_size']
                min_samples = optimal_params['min_samples']
            else:
                min_cluster_size = kwargs.get('min_cluster_size', hdbscan_config.get('min_cluster_size', 5))
                min_samples = kwargs.get('min_samples', hdbscan_config.get('min_samples'))

            metric = kwargs.get('metric', hdbscan_config.get('metric', 'euclidean'))
            cluster_selection_method = kwargs.get('cluster_selection_method',
                                                  hdbscan_config.get('cluster_selection_method', 'eom'))

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_method=cluster_selection_method
            )

        elif method == 'hierarchical':
            hierarchical_config = self.config.get('clustering', {}).get('hierarchical', {})
            if n_clusters is None:
                n_clusters = hierarchical_config.get('n_clusters')
                if n_clusters is None:
                    n_clusters = min(5, len(rules)//3)

            linkage = kwargs.get('linkage', hierarchical_config.get('linkage', 'ward'))

            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        # 執行聚類
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # 處理 DBSCAN 和 HDBSCAN 的 noise 點
        if method in ['dbscan', 'hdbscan']:
            cluster_labels = self._handle_dbscan_noise(cluster_labels, embeddings, **kwargs)
        
        # 組織結果
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append({
                'rule': rules[i],
                'index': i
            })
        
        # 計算聚類統計
        cluster_stats = {}
        for label, items in clusters.items():
            cluster_embeddings = embeddings[[item['index'] for item in items]]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # 計算內聚度 (聚類內相似度)
            if len(cluster_embeddings) > 1:
                intra_similarities = cosine_similarity(cluster_embeddings)
                # 去除對角線元素（自己和自己的相似度）
                mask = np.ones(intra_similarities.shape, dtype=bool)
                np.fill_diagonal(mask, 0)
                cohesion = np.mean(intra_similarities[mask])
            else:
                cohesion = 1.0
            
            cluster_stats[label] = {
                'size': len(items),
                'cohesion': float(cohesion),
                'centroid': centroid.tolist()
            }
        
        result = {
            'clusters': dict(clusters),
            'cluster_labels': cluster_labels.tolist(),
            'cluster_stats': cluster_stats,
            'n_clusters': len(clusters),
            'method': method,
            'embeddings': embeddings
        }
        
        # 如果使用了參數優化，添加優化結果
        if optimize_params:
            if method == 'dbscan' and 'eps' not in kwargs:
                result['eps_optimization'] = eps_optimization
                result['used_eps'] = eps
            elif method == 'hdbscan' and ('min_cluster_size' not in kwargs or 'min_samples' not in kwargs):
                result['hdbscan_optimization'] = hdbscan_optimization
                result['used_params'] = optimal_params
        
        logger.info(f"Clustering completed. Found {len(clusters)} clusters")
        return result

# =================================0710: for DBSCAN eps search========================================
    def find_optimal_eps_with_ami(self, embeddings: np.ndarray, 
                                min_samples: int = 2,
                                eps_range: Tuple[float, float] = (0.1, 1.0),
                                n_eps_values: int = 20,
                                reference_method: str = 'kmeans',
                                n_reference_clusters: int = None) -> Dict:
        """
        使用AMI找到DBSCAN的最佳eps參數
        
        Args:
            embeddings: 嵌入向量
            min_samples: DBSCAN的min_samples參數
            eps_range: eps搜索範圍
            n_eps_values: 測試的eps值數量
            reference_method: 參考聚類方法用於AMI計算
            n_reference_clusters: 參考聚類的cluster數量
            
        Returns:
            包含最佳eps和評估結果的字典
        """
        logger.info("Finding optimal eps for DBSCAN using AMI")
        
        # 生成eps候選值
        eps_candidates = np.linspace(eps_range[0], eps_range[1], n_eps_values)
        
        # 獲取參考聚類結果
        if reference_method == 'kmeans':
            if n_reference_clusters is None:
                n_reference_clusters = self._find_optimal_clusters(embeddings, max_k=len(embeddings)//2)
            
            reference_clusterer = KMeans(n_clusters=n_reference_clusters, random_state=42)
            reference_labels = reference_clusterer.fit_predict(embeddings)
        
        elif reference_method == 'hierarchical':
            if n_reference_clusters is None:
                n_reference_clusters = min(5, len(embeddings)//3)
            
            reference_clusterer = AgglomerativeClustering(n_clusters=n_reference_clusters, linkage='ward')
            reference_labels = reference_clusterer.fit_predict(embeddings)
        
        else:
            raise ValueError(f"Unsupported reference method: {reference_method}")
        
        # 評估每個eps值
        results = []
        best_ami = -1
        best_eps = eps_candidates[0]
        
        for eps in eps_candidates:
            try:
                # 執行DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                dbscan_labels = dbscan.fit_predict(embeddings)
                
                # 計算clusters數量（排除噪音點-1）
                n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                n_noise = list(dbscan_labels).count(-1)
                
                # 如果沒有形成有效聚類，跳過
                if n_clusters < 2:
                    results.append({
                        'eps': eps,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'ami_score': -1,
                        'silhouette_score': -1,
                        'valid': False
                    })
                    continue
                
                # 計算AMI分數
                ami_score = adjusted_mutual_info_score(reference_labels, dbscan_labels)
                
                # 計算輪廓係數（排除噪音點）
                if n_clusters > 1 and n_noise < len(embeddings):
                    # 只對非噪音點計算輪廓係數
                    non_noise_mask = dbscan_labels != -1
                    if np.sum(non_noise_mask) > 1:
                        silhouette_avg = silhouette_score(
                            embeddings[non_noise_mask], 
                            dbscan_labels[non_noise_mask]
                        )
                    else:
                        silhouette_avg = -1
                else:
                    silhouette_avg = -1
                
                results.append({
                    'eps': eps,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'ami_score': ami_score,
                    'silhouette_score': silhouette_avg,
                    'valid': True
                })
                
                # 更新最佳eps
                if ami_score > best_ami:
                    best_ami = ami_score
                    best_eps = eps
                    
            except Exception as e:
                logger.warning(f"Error evaluating eps={eps}: {e}")
                results.append({
                    'eps': eps,
                    'n_clusters': 0,
                    'n_noise': len(embeddings),
                    'ami_score': -1,
                    'silhouette_score': -1,
                    'valid': False
                })
        
        return {
            'best_eps': best_eps,
            'best_ami': best_ami,
            'results': results,
            'reference_method': reference_method,
            'reference_clusters': n_reference_clusters
        }

    def find_optimal_eps_knee_method(self, embeddings: np.ndarray, 
                                    min_samples: int = 2,
                                    k: int = None) -> float:
        """
        使用k-distance圖和膝蓋法找到最佳eps
        
        Args:
            embeddings: 嵌入向量
            min_samples: DBSCAN的min_samples參數
            k: k-distance中的k值，默認為min_samples
            
        Returns:
            推薦的eps值
        """
        if k is None:
            k = min_samples
        
        # 計算k-distance
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # 取第k個鄰居的距離並排序
        k_distances = distances[:, k-1]
        k_distances = np.sort(k_distances)[::-1]  # 降序排列
        
        # 使用膝蓋法找到最佳eps
        # 計算曲線的二階導數來找到膝蓋點
        if len(k_distances) > 4:
            # 使用差分近似二階導數
            first_diff = np.diff(k_distances)
            second_diff = np.diff(first_diff)
            
            # 找到二階導數的最大值點（膝蓋點）
            knee_idx = np.argmax(second_diff) + 1
            optimal_eps = k_distances[knee_idx]
        else:
            # 如果數據點太少，使用中位數
            optimal_eps = np.median(k_distances)
        
        return optimal_eps
    
    def plot_eps_optimization_results(self, optimization_results: Dict, 
                                            figsize: Tuple[int, int] = (12, 8)):
        """
        繪製eps優化結果圖表
        
        Args:
            optimization_results: eps優化結果
            figsize: 圖表大小
        """
        results = optimization_results['results']
        valid_results = [r for r in results if r['valid']]
        
        if not valid_results:
            logger.warning("No valid results to plot")
            return
        
        eps_values = [r['eps'] for r in valid_results]
        ami_scores = [r['ami_score'] for r in valid_results]
        silhouette_scores = [r['silhouette_score'] for r in valid_results]
        n_clusters = [r['n_clusters'] for r in valid_results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # AMI分數
        ax1.plot(eps_values, ami_scores, 'b-o', markersize=4)
        ax1.axvline(x=optimization_results['best_eps'], color='r', linestyle='--', 
                    label=f'Best eps: {optimization_results["best_eps"]:.4f}')
        ax1.set_xlabel('eps')
        ax1.set_ylabel('AMI Score')
        ax1.set_title('AMI Score vs eps')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 輪廓係數
        ax2.plot(eps_values, silhouette_scores, 'g-o', markersize=4)
        ax2.axvline(x=optimization_results['best_eps'], color='r', linestyle='--')
        ax2.set_xlabel('eps')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs eps')
        ax2.grid(True, alpha=0.3)
        
        # 聚類數量
        ax3.plot(eps_values, n_clusters, 'purple', marker='s', markersize=4)
        ax3.axvline(x=optimization_results['best_eps'], color='r', linestyle='--')
        ax3.set_xlabel('eps')
        ax3.set_ylabel('Number of Clusters')
        ax3.set_title('Number of Clusters vs eps')
        ax3.grid(True, alpha=0.3)
        
        # 噪音點數量
        n_noise = [r['n_noise'] for r in valid_results]
        ax4.plot(eps_values, n_noise, 'orange', marker='^', markersize=4)
        ax4.axvline(x=optimization_results['best_eps'], color='r', linestyle='--')
        ax4.set_xlabel('eps')
        ax4.set_ylabel('Number of Noise Points')
        ax4.set_title('Noise Points vs eps')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# =========================0710: HDBSCAN========================        
    def find_optimal_hdbscan_params(self, embeddings: np.ndarray,
                                min_cluster_size_range: Tuple[int, int] = (2, 20),
                                min_samples_range: Tuple[int, int] = (1, 10),
                                reference_method: str = 'kmeans',
                                n_reference_clusters: int = None) -> Dict:
        """
        使用AMI找到HDBSCAN的最佳參數
        
        Args:
            embeddings: 嵌入向量
            min_cluster_size_range: min_cluster_size參數範圍
            min_samples_range: min_samples參數範圍
            reference_method: 參考聚類方法
            n_reference_clusters: 參考聚類數量
            
        Returns:
            包含最佳參數和評估結果的字典
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        
        logger.info("Finding optimal HDBSCAN parameters using AMI")
        
        # HDBSCAN parameter optimization using default euclidean distance
        
        # 獲取參考聚類結果
        if reference_method == 'kmeans':
            if n_reference_clusters is None:
                n_reference_clusters = self._find_optimal_clusters(embeddings, max_k=min(10, len(embeddings)//2))
            reference_clusterer = KMeans(n_clusters=n_reference_clusters, random_state=42)
            reference_labels = reference_clusterer.fit_predict(embeddings)
        elif reference_method == 'hierarchical':
            if n_reference_clusters is None:
                n_reference_clusters = min(5, len(embeddings)//3)
            reference_clusterer = AgglomerativeClustering(n_clusters=n_reference_clusters, linkage='ward')
            reference_labels = reference_clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"Unsupported reference method: {reference_method}")
        
        # 生成參數候選值
        min_cluster_sizes = range(min_cluster_size_range[0], 
                                min(min_cluster_size_range[1], len(embeddings)//2) + 1)
        min_samples_values = range(min_samples_range[0], min_samples_range[1] + 1)
        
        results = []
        best_ami = -1
        best_params = None
        
        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_values:
                try:
                    # 執行HDBSCAN with default euclidean distance
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric='euclidean',  # Use default euclidean distance
                        cluster_selection_method='eom'  # Excess of Mass
                    )
                    
                    cluster_labels = clusterer.fit_predict(embeddings)
                    
                    # 計算clusters數量（排除噪音點-1）
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    n_noise = list(cluster_labels).count(-1)
                    
                    # 如果沒有形成有效聚類，跳過
                    if n_clusters < 2:
                        results.append({
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'n_noise': n_noise,
                            'ami_score': -1,
                            'silhouette_score': -1,
                            'cluster_persistence': -1,
                            'valid': False
                        })
                        continue
                    
                    # 計算AMI分數
                    ami_score = adjusted_mutual_info_score(reference_labels, cluster_labels)
                    
                    # 計算輪廓係數
                    silhouette_avg = -1
                    if n_clusters > 1 and n_noise < len(embeddings):
                        non_noise_mask = cluster_labels != -1
                        if np.sum(non_noise_mask) > 1:
                            silhouette_avg = silhouette_score(
                                embeddings[non_noise_mask], 
                                cluster_labels[non_noise_mask]
                            )
                    
                    # 計算聚類持久性（HDBSCAN特有）
                    cluster_persistence = 0
                    if hasattr(clusterer, 'cluster_persistence_') and clusterer.cluster_persistence_ is not None:
                        cluster_persistence = np.mean(clusterer.cluster_persistence_)
                    
                    results.append({
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'ami_score': ami_score,
                        'silhouette_score': silhouette_avg,
                        'cluster_persistence': cluster_persistence,
                        'valid': True
                    })
                    
                    # 更新最佳參數
                    if ami_score > best_ami:
                        best_ami = ami_score
                        best_params = {
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples
                        }
                        
                except Exception as e:
                    logger.warning(f"Error with min_cluster_size={min_cluster_size}, "
                                f"min_samples={min_samples}: {e}")
                    results.append({
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'n_clusters': 0,
                        'n_noise': len(embeddings),
                        'ami_score': -1,
                        'silhouette_score': -1,
                        'cluster_persistence': -1,
                        'valid': False
                    })
        
        return {
            'best_params': best_params,
            'best_ami': best_ami,
            'results': results,
            'reference_method': reference_method,
            'reference_clusters': n_reference_clusters
        }

    def compare_clustering_methods(self, embeddings: np.ndarray,
                                reference_labels: np.ndarray = None,
                                methods: List[str] = ['kmeans', 'dbscan', 'hdbscan'],
                                **kwargs) -> Dict:
        """
        比較不同聚類方法的性能
        
        Args:
            embeddings: 嵌入向量
            reference_labels: 參考標籤（如果有的話）
            methods: 要比較的方法列表
            **kwargs: 各方法的參數
            
        Returns:
            比較結果字典
        """
        results = {}
        
        for method in methods:
            try:
                if method == 'kmeans':
                    n_clusters = kwargs.get('n_clusters', self._find_optimal_clusters(embeddings))
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = clusterer.fit_predict(embeddings)
                    
                elif method == 'dbscan':
                    if 'optimal_eps' in kwargs:
                        eps = kwargs['optimal_eps']
                    else:
                        eps_opt = self.find_optimal_eps_with_ami(embeddings)
                        eps = eps_opt['best_eps']
                    
                    min_samples = kwargs.get('min_samples', 2)
                    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                    labels = clusterer.fit_predict(embeddings)
                    
                elif method == 'hdbscan':
                    if not HDBSCAN_AVAILABLE:
                        logger.warning("HDBSCAN not available, skipping")
                        continue
                    
                    if 'optimal_hdbscan_params' in kwargs:
                        params = kwargs['optimal_hdbscan_params']
                    else:
                        hdbscan_opt = self.find_optimal_hdbscan_params(embeddings)
                        params = hdbscan_opt['best_params']
                    
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=params['min_cluster_size'],
                        min_samples=params['min_samples'],
                        metric='euclidean',  # Use default euclidean distance
                        cluster_selection_method='eom'
                    )
                    labels = clusterer.fit_predict(embeddings)
                    
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue
                
                # 計算評估指標
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # 輪廓係數
                silhouette_avg = -1
                if n_clusters > 1 and n_noise < len(embeddings):
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 1:
                        silhouette_avg = silhouette_score(
                            embeddings[non_noise_mask], 
                            labels[non_noise_mask]
                        )
                
                # AMI分數（如果有參考標籤）
                ami_score = -1
                if reference_labels is not None:
                    ami_score = adjusted_mutual_info_score(reference_labels, labels)
                
                # 聚類持久性（僅HDBSCAN）
                cluster_persistence = -1
                if method == 'hdbscan' and hasattr(clusterer, 'cluster_persistence_'):
                    if clusterer.cluster_persistence_ is not None:
                        cluster_persistence = np.mean(clusterer.cluster_persistence_)
                
                results[method] = {
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette_score': silhouette_avg,
                    'ami_score': ami_score,
                    'cluster_persistence': cluster_persistence,
                    'clusterer': clusterer
                }
                
            except Exception as e:
                logger.error(f"Error with method {method}: {e}")
                results[method] = {
                    'error': str(e),
                    'labels': None,
                    'n_clusters': 0,
                    'n_noise': len(embeddings),
                    'silhouette_score': -1,
                    'ami_score': -1,
                    'cluster_persistence': -1
                }
        
        return results 

def perform_dbscan_grid_search(inferencer, rules, true_labels):
    """
    Perform comprehensive grid search for DBSCAN parameters
    """
    logger.info("Starting DBSCAN grid search...")
    
    # Encode rules first
    embeddings = inferencer.encode_rules(rules)
    
    # Define parameter ranges
    eps_range = np.linspace(0.05, 0.9, 20)  # 20 different eps values
    min_samples_range = range(2, 11)  # min_samples from 2 to 10
    
    results = []
    best_ami = -1
    best_params = None
    
    total_combinations = len(eps_range) * len(min_samples_range)
    current_combination = 0
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            current_combination += 1
            logger.info(f"Testing combination {current_combination}/{total_combinations}: eps={eps:.3f}, min_samples={min_samples}")
            
            try:
                # Run DBSCAN
                clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                cluster_labels = clusterer.fit_predict(embeddings)
                
                # Handle noise points
                cluster_labels = inferencer._handle_dbscan_noise(cluster_labels, embeddings, noise_strategy='individual')
                
                # Calculate metrics
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = list(cluster_labels).count(-1)
                
                if n_clusters >= 2:
                    # Calculate AMI score
                    ami_score = adjusted_mutual_info_score(true_labels, cluster_labels)
                    
                    # Calculate silhouette score
                    non_noise_mask = cluster_labels != -1
                    if np.sum(non_noise_mask) > 1 and n_clusters > 1:
                        silhouette_avg = silhouette_score(embeddings[non_noise_mask], cluster_labels[non_noise_mask])
                    else:
                        silhouette_avg = -1
                    
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'ami_score': ami_score,
                        'silhouette_score': silhouette_avg,
                        'valid': True
                    })
                    
                    # Update best parameters
                    if ami_score > best_ami:
                        best_ami = ami_score
                        best_params = {'eps': eps, 'min_samples': min_samples}
                        logger.info(f"New best AMI: {best_ami:.4f} with eps={eps:.3f}, min_samples={min_samples}")
                
                else:
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'ami_score': -1,
                        'silhouette_score': -1,
                        'valid': False
                    })
                    
            except Exception as e:
                logger.warning(f"Error with eps={eps}, min_samples={min_samples}: {e}")
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': 0,
                    'n_noise': len(embeddings),
                    'ami_score': -1,
                    'silhouette_score': -1,
                    'valid': False
                })
    
    logger.info(f"Grid search completed. Best AMI: {best_ami:.4f} with parameters: {best_params}")
    
    return {
        'best_params': best_params,
        'best_ami': best_ami,
        'results': results,
        'method': 'dbscan'
    }

def perform_hdbscan_grid_search(inferencer, rules, true_labels):
    """
    Perform comprehensive grid search for HDBSCAN parameters
    """
    if not HDBSCAN_AVAILABLE:
        logger.error("HDBSCAN not available. Install with: pip install hdbscan")
        return None
    
    logger.info("Starting HDBSCAN grid search...")
    
    # Encode rules first
    embeddings = inferencer.encode_rules(rules)
    
    # HDBSCAN grid search using default euclidean distance
    
    # Define parameter ranges
    min_cluster_size_range = range(2, min(21, len(embeddings)//3))  # 2 to 20 or 1/3 of data
    min_samples_range = range(1, 11)  # 1 to 10
    
    results = []
    best_ami = -1
    best_params = None
    
    total_combinations = len(min_cluster_size_range) * len(min_samples_range)
    current_combination = 0
    
    for min_cluster_size in min_cluster_size_range:
        for min_samples in min_samples_range:
            current_combination += 1
            logger.info(f"Testing combination {current_combination}/{total_combinations}: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
            
            try:
                # Run HDBSCAN with default euclidean distance
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',  # Use default euclidean distance
                    cluster_selection_method='eom'
                )
                cluster_labels = clusterer.fit_predict(embeddings)
                
                # Handle noise points
                cluster_labels = inferencer._handle_dbscan_noise(cluster_labels, embeddings, noise_strategy='individual')
                
                # Calculate metrics
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = list(cluster_labels).count(-1)
                
                if n_clusters >= 2:
                    # Calculate AMI score
                    ami_score = adjusted_mutual_info_score(true_labels, cluster_labels)
                    
                    # Calculate silhouette score
                    non_noise_mask = cluster_labels != -1
                    if np.sum(non_noise_mask) > 1 and n_clusters > 1:
                        silhouette_avg = silhouette_score(embeddings[non_noise_mask], cluster_labels[non_noise_mask])
                    else:
                        silhouette_avg = -1
                    
                    # Calculate cluster persistence if available
                    cluster_persistence = -1
                    if hasattr(clusterer, 'cluster_persistence_') and clusterer.cluster_persistence_ is not None:
                        cluster_persistence = np.mean(clusterer.cluster_persistence_)
                    
                    results.append({
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'ami_score': ami_score,
                        'silhouette_score': silhouette_avg,
                        'cluster_persistence': cluster_persistence,
                        'valid': True
                    })
                    
                    # Update best parameters
                    if ami_score > best_ami:
                        best_ami = ami_score
                        best_params = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples}
                        logger.info(f"New best AMI: {best_ami:.4f} with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
                
                else:
                    results.append({
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'ami_score': -1,
                        'silhouette_score': -1,
                        'cluster_persistence': -1,
                        'valid': False
                    })
                    
            except Exception as e:
                logger.warning(f"Error with min_cluster_size={min_cluster_size}, min_samples={min_samples}: {e}")
                results.append({
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'n_clusters': 0,
                    'n_noise': len(embeddings),
                    'ami_score': -1,
                    'silhouette_score': -1,
                    'cluster_persistence': -1,
                    'valid': False
                })
    
    logger.info(f"Grid search completed. Best AMI: {best_ami:.4f} with parameters: {best_params}")
    
    return {
        'best_params': best_params,
        'best_ami': best_ami,
        'results': results,
        'method': 'hdbscan'
    }

def plot_parameter_optimization_results(optimization_result, method, output_dir='results'):
    """
    Create comprehensive visualizations for parameter optimization results

    Args:
        optimization_result: Optimization results dictionary
        method: Clustering method ('dbscan' or 'hdbscan')
        output_dir: Directory to save plots (default: 'results')
    """
    results = optimization_result['results']
    valid_results = [r for r in results if r['valid']]

    if not valid_results:
        logger.warning("No valid results to plot")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if method == 'dbscan':
        plot_dbscan_results(valid_results, optimization_result['best_params'], output_dir=output_dir)
    elif method == 'hdbscan':
        plot_hdbscan_results(valid_results, optimization_result['best_params'], output_dir=output_dir)

def plot_dbscan_results(valid_results, best_params, output_dir='results'):
    """
    Create visualizations for DBSCAN parameter optimization

    Args:
        valid_results: List of valid optimization results
        best_params: Best parameters found
        output_dir: Directory to save plots (default: 'results')
    """
    # Convert to DataFrame for easier manipulation
    df_results = pd.DataFrame(valid_results)
    
    # Create a pivot table for heatmap
    pivot_ami = df_results.pivot_table(values='ami_score', index='min_samples', columns='eps', aggfunc='mean')
    pivot_clusters = df_results.pivot_table(values='n_clusters', index='min_samples', columns='eps', aggfunc='mean')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. AMI Score Heatmap
    im1 = axes[0, 0].imshow(pivot_ami.values, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('AMI Score vs Parameters')
    axes[0, 0].set_xlabel('eps')
    axes[0, 0].set_ylabel('min_samples')
    axes[0, 0].set_xticks(range(len(pivot_ami.columns)))
    axes[0, 0].set_xticklabels([f'{x:.2f}' for x in pivot_ami.columns])
    axes[0, 0].set_yticks(range(len(pivot_ami.index)))
    axes[0, 0].set_yticklabels(pivot_ami.index)
    
    # Add colorbar
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Mark best parameters
    best_eps_idx = list(pivot_ami.columns).index(min(pivot_ami.columns, key=lambda x: abs(x - best_params['eps'])))
    best_min_samples_idx = list(pivot_ami.index).index(best_params['min_samples'])
    axes[0, 0].scatter([best_eps_idx], [best_min_samples_idx], color='red', s=100, marker='x', linewidths=3)
    
    # 2. Number of Clusters Heatmap
    im2 = axes[0, 1].imshow(pivot_clusters.values, cmap='plasma', aspect='auto')
    axes[0, 1].set_title('Number of Clusters vs Parameters')
    axes[0, 1].set_xlabel('eps')
    axes[0, 1].set_ylabel('min_samples')
    axes[0, 1].set_xticks(range(len(pivot_clusters.columns)))
    axes[0, 1].set_xticklabels([f'{x:.2f}' for x in pivot_clusters.columns])
    axes[0, 1].set_yticks(range(len(pivot_clusters.index)))
    axes[0, 1].set_yticklabels(pivot_clusters.index)
    
    plt.colorbar(im2, ax=axes[0, 1])
    axes[0, 1].scatter([best_eps_idx], [best_min_samples_idx], color='red', s=100, marker='x', linewidths=3)
    
    # 3. AMI vs eps for different min_samples
    unique_min_samples = sorted(df_results['min_samples'].unique())
    for min_samples in unique_min_samples:
        subset = df_results[df_results['min_samples'] == min_samples]
        axes[1, 0].plot(subset['eps'], subset['ami_score'], 'o-', label=f'min_samples={min_samples}', alpha=0.7)
    
    axes[1, 0].axvline(x=best_params['eps'], color='red', linestyle='--', alpha=0.7, label='Best eps')
    axes[1, 0].set_xlabel('eps')
    axes[1, 0].set_ylabel('AMI Score')
    axes[1, 0].set_title('AMI Score vs eps for different min_samples')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 3D scatter plot
    from mpl_toolkits.mplot3d import Axes3D
    
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    scatter = ax3d.scatter(df_results['eps'], df_results['min_samples'], df_results['ami_score'], 
                          c=df_results['ami_score'], cmap='viridis', s=50)
    
    # Mark best point
    ax3d.scatter([best_params['eps']], [best_params['min_samples']], [df_results[
        (df_results['eps'] == best_params['eps']) & 
        (df_results['min_samples'] == best_params['min_samples'])
    ]['ami_score'].iloc[0]], color='red', s=100, marker='x')
    
    ax3d.set_xlabel('eps')
    ax3d.set_ylabel('min_samples')
    ax3d.set_zlabel('AMI Score')
    ax3d.set_title('3D Parameter Space')
    
    plt.tight_layout()
    from datetime import datetime
    date_str = datetime.now().strftime('%m%d')
    plt.savefig(f'{output_dir}/dbscan_parameter_optimization_{date_str}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    logger.info(f"DBSCAN Parameter Optimization Summary:")
    logger.info(f"Best parameters: eps={best_params['eps']:.3f}, min_samples={best_params['min_samples']}")
    logger.info(f"Best AMI score: {df_results[(df_results['eps'] == best_params['eps']) & (df_results['min_samples'] == best_params['min_samples'])]['ami_score'].iloc[0]:.4f}")

def plot_hdbscan_results(valid_results, best_params, output_dir='results'):
    """
    Create visualizations for HDBSCAN parameter optimization

    Args:
        valid_results: List of valid optimization results
        best_params: Best parameters found
        output_dir: Directory to save plots (default: 'results')
    """
    # Convert to DataFrame for easier manipulation
    df_results = pd.DataFrame(valid_results)
    
    # Create a pivot table for heatmap
    pivot_ami = df_results.pivot_table(values='ami_score', index='min_samples', columns='min_cluster_size', aggfunc='mean')
    pivot_clusters = df_results.pivot_table(values='n_clusters', index='min_samples', columns='min_cluster_size', aggfunc='mean')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. AMI Score Heatmap
    im1 = axes[0, 0].imshow(pivot_ami.values, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('AMI Score vs Parameters')
    axes[0, 0].set_xlabel('min_cluster_size')
    axes[0, 0].set_ylabel('min_samples')
    axes[0, 0].set_xticks(range(len(pivot_ami.columns)))
    axes[0, 0].set_xticklabels(pivot_ami.columns)
    axes[0, 0].set_yticks(range(len(pivot_ami.index)))
    axes[0, 0].set_yticklabels(pivot_ami.index)
    
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Mark best parameters
    best_min_cluster_size_idx = list(pivot_ami.columns).index(best_params['min_cluster_size'])
    best_min_samples_idx = list(pivot_ami.index).index(best_params['min_samples'])
    axes[0, 0].scatter([best_min_cluster_size_idx], [best_min_samples_idx], color='red', s=100, marker='x', linewidths=3)
    
    # 2. Number of Clusters Heatmap
    im2 = axes[0, 1].imshow(pivot_clusters.values, cmap='plasma', aspect='auto')
    axes[0, 1].set_title('Number of Clusters vs Parameters')
    axes[0, 1].set_xlabel('min_cluster_size')
    axes[0, 1].set_ylabel('min_samples')
    axes[0, 1].set_xticks(range(len(pivot_clusters.columns)))
    axes[0, 1].set_xticklabels(pivot_clusters.columns)
    axes[0, 1].set_yticks(range(len(pivot_clusters.index)))
    axes[0, 1].set_yticklabels(pivot_clusters.index)
    
    plt.colorbar(im2, ax=axes[0, 1])
    axes[0, 1].scatter([best_min_cluster_size_idx], [best_min_samples_idx], color='red', s=100, marker='x', linewidths=3)
    
    # 3. AMI vs min_cluster_size for different min_samples
    unique_min_samples = sorted(df_results['min_samples'].unique())
    for min_samples in unique_min_samples:
        subset = df_results[df_results['min_samples'] == min_samples]
        axes[1, 0].plot(subset['min_cluster_size'], subset['ami_score'], 'o-', label=f'min_samples={min_samples}', alpha=0.7)
    
    axes[1, 0].axvline(x=best_params['min_cluster_size'], color='red', linestyle='--', alpha=0.7, label='Best min_cluster_size')
    axes[1, 0].set_xlabel('min_cluster_size')
    axes[1, 0].set_ylabel('AMI Score')
    axes[1, 0].set_title('AMI Score vs min_cluster_size for different min_samples')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 3D scatter plot
    from mpl_toolkits.mplot3d import Axes3D
    
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    scatter = ax3d.scatter(df_results['min_cluster_size'], df_results['min_samples'], df_results['ami_score'], 
                          c=df_results['ami_score'], cmap='viridis', s=50)
    
    # Mark best point
    ax3d.scatter([best_params['min_cluster_size']], [best_params['min_samples']], [df_results[
        (df_results['min_cluster_size'] == best_params['min_cluster_size']) & 
        (df_results['min_samples'] == best_params['min_samples'])
    ]['ami_score'].iloc[0]], color='red', s=100, marker='x')
    
    ax3d.set_xlabel('min_cluster_size')
    ax3d.set_ylabel('min_samples')
    ax3d.set_zlabel('AMI Score')
    ax3d.set_title('3D Parameter Space')
    
    plt.tight_layout()
    from datetime import datetime
    date_str = datetime.now().strftime('%m%d')
    plt.savefig(f'{output_dir}/hdbscan_parameter_optimization_{date_str}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    logger.info(f"HDBSCAN Parameter Optimization Summary:")
    logger.info(f"Best parameters: min_cluster_size={best_params['min_cluster_size']}, min_samples={best_params['min_samples']}")
    logger.info(f"Best AMI score: {df_results[(df_results['min_cluster_size'] == best_params['min_cluster_size']) & (df_results['min_samples'] == best_params['min_samples'])]['ami_score'].iloc[0]:.4f}")

# 使用示例
def example_usage(inferencer,rules):
    """
    使用示例：比較DBSCAN和HDBSCAN
    """
    # 比較不同方法
    methods_to_compare = ['dbscan', 'hdbscan', 'kmeans']
    comparison_results = {}
    
    for method in methods_to_compare:
        try:
            result = inferencer.cluster_rules(rules, method=method, optimize_params=True)
            comparison_results[method] = result
            print(f"\n{method.upper()} Results:")
            print(f"  Clusters: {result['n_clusters']}")
            print(f"  Method: {result['method']}")
            
            if method == 'dbscan' and 'used_eps' in result:
                print(f"  Optimal eps: {result['used_eps']:.4f}")
            elif method == 'hdbscan' and 'used_params' in result:
                print(f"  Optimal params: {result['used_params']}")
                
        except Exception as e:
            print(f"Error with {method}: {e}")
    
    # 找到最佳方法
    best_method = None
    best_score = -1
    
    for method, result in comparison_results.items():
        # 可以使用不同的評估指標
        if 'eps_optimization' in result:
            score = result['eps_optimization']['best_ami']
        elif 'hdbscan_optimization' in result:
            score = result['hdbscan_optimization']['best_ami']
        else:
            score = 0  # 為其他方法設置默認分數
            
        if score > best_score:
            best_score = score
            best_method = method
    
    print(f"\nBest method: {best_method} (Score: {best_score:.4f})")
    return comparison_results

def fetch_data(data_path):
    # 讀取測試數據
    df = pd.read_csv(data_path)
    df.rename(columns={'Description': 'description', 'Ground Truth': 'group_id'}, inplace=True)
    
    # filter out the rows that has empty value in the column of group_id
    df = df[df['group_id'].notna()]        
    rules = df['description'].tolist()
    true_labels = df['group_id'].tolist()
    
    return df, rules, true_labels

def main_find_top5(model_path=None, data_path=None):
    # 自動檢測模型路徑
    if model_path is None:
        model_path = find_latest_model()
        logger.info(f"Auto-detected model: {model_path}")
    
    # 默認數據路徑
    if data_path is None:
        data_path = os.path.expanduser('~/111111ForDownload/csv_TLR001_22nm_with_init_cluster_with_gt_class_name.csv')
    
    # 顯示模型信息
    model_info = get_model_info(model_path)
    logger.info(f"Using model: {model_info['name']}")
    logger.info(f"Model created: {model_info['created_time']}")
    logger.info(f"Model modified: {model_info['modified_time']}")
    
    # 初始化推理器
    inferencer = RuleEmbeddingInference(model_path)
    
    # 加載模型
    if not inferencer.load_model():
        logger.error("Failed to load model")
        return
    
    try:
        # 讀取測試數據
        df, rules, true_labels = fetch_data(data_path)
        
        logger.info(f"Loaded {len(rules)} rules for inference")
        
        # 1. 相似規則查找
        query_rule = rules[0]
        similar_rules = inferencer.find_similar_rules(
            query_rule, 
            rules[1:], 
            top_k=5, 
            threshold=0.3
        )
        
        logger.info(f"\nTop 5 similar rules to: {query_rule}")
        for i, item in enumerate(similar_rules):
            logger.info(f"  {i+1}. {item['rule'][:60]}... (similarity: {item['similarity']:.3f})")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

def main_visualize(model_path=None, data_path=None):
    """示例使用"""
    # 自動檢測模型路徑
    if model_path is None:
        model_path = find_latest_model()
        logger.info(f"Auto-detected model: {model_path}")
    
    # 默認數據路徑
    if data_path is None:
        data_path = os.path.expanduser('~/111111ForDownload/csv_TLR001_22nm_with_init_cluster_with_gt_class_name.csv')
    
    # 顯示模型信息
    model_info = get_model_info(model_path)
    logger.info(f"Using model: {model_info['name']}")
    logger.info(f"Model created: {model_info['created_time']}")
    logger.info(f"Model modified: {model_info['modified_time']}")
    
    # 初始化推理器
    inferencer = RuleEmbeddingInference(model_path)
    
    # 加載模型
    if not inferencer.load_model():
        logger.error("Failed to load model")
        return
    
    try:
        # 讀取測試數據
        df, rules, true_labels = fetch_data(data_path)
        
        logger.info(f"Loaded {len(rules)} rules for inference")
        
        # 分析規則組
        if true_labels:
            group_analysis = inferencer.analyze_rule_groups(rules, true_labels)
            
            logger.info("\nRule group analysis:")
            for group_id, analysis in group_analysis['group_analysis'].items():
                logger.info(f"  Group {group_id}: {analysis['size']} rules, "
                           f"intra-similarity: {analysis['intra_similarity']:.3f}")
        
        # 可視化
        inferencer.visualize_embeddings(
            rules, 
            labels=true_labels if true_labels else None,
            method='tsne',
            title="Rule Embeddings Visualization"
        )
        logger.info("Visualization completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

def main_inference(model_path=None, data_path=None):
    """示例使用"""
    # 自動檢測模型路徑
    if model_path is None:
        model_path = find_latest_model()
        logger.info(f"Auto-detected model: {model_path}")
    
    # 默認數據路徑
    if data_path is None:
        data_path = os.path.expanduser('~/111111ForDownload/csv_TLR001_22nm_with_init_cluster_with_gt_class_name.csv')
    
    # 顯示模型信息
    model_info = get_model_info(model_path)
    logger.info(f"Using model: {model_info['name']}")
    logger.info(f"Model created: {model_info['created_time']}")
    logger.info(f"Model modified: {model_info['modified_time']}")
    
    # 初始化推理器
    inferencer = RuleEmbeddingInference(model_path)
    
    # 加載模型
    if not inferencer.load_model():
        logger.error("Failed to load model")
        return
    
    try:
        # 讀取測試數據
        df = pd.read_csv(data_path)
        df.rename(columns={'Description': 'description', 'Ground Truth': 'group_id'}, inplace=True)
        
        # filter out the rows that has empty value in the column of group_id
        df = df[df['group_id'].notna()]        
        rules = df['description'].tolist()
        logger.info(f"Loaded {len(rules)} rules for inference")
        
        # 批次推理
        from datetime import datetime
        date_str = datetime.now().strftime('%m%d')
        output_file = f'inference_results_{date_str}.json'
        batch_results,df = inferencer.batch_inference(
            data_path, 
            output_file, 
            clustering_method='dbscan',
            n_clusters=5
        )
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def main_evaluation(model_path=None, data_path=None,
                    method='dbscan',
                    optimize_params=False,
                    config_path=None,
                    **kwargs):
    """Enhanced evaluation with comprehensive parameter grid search and visualization"""

    # Load config first to get default paths
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'inference_config.json')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load config: {e}, using defaults")
        config = {}

    # 自動檢測模型路徑
    if model_path is None:
        # First try config file
        model_path = config.get('paths', {}).get('model_path')

        if model_path is None:
            # Try to auto-detect
            try:
                base_dir = config.get('paths', {}).get('model_base_dir', './')
                pattern = config.get('paths', {}).get('model_pattern', 'rule-embedder-*')
                model_path = find_latest_model(base_dir=base_dir, model_pattern=pattern)
                logger.info(f"Auto-detected model: {model_path}")
            except FileNotFoundError as e:
                logger.error(f"No model found. Please specify model_path in config or as argument.")
                logger.error(f"Error: {e}")
                raise

    # 默認數據路徑
    if data_path is None:
        # First try config file
        data_path = config.get('paths', {}).get('data_path')

        if data_path is None:
            # Use default from config or hardcoded fallback
            data_path = config.get('paths', {}).get('data_default',
                os.path.expanduser('~/111111ForDownload/csv_TLR001_22nm_with_init_cluster_with_gt_class_name.csv'))

    # 顯示模型信息
    model_info = get_model_info(model_path)
    logger.info(f"Using model: {model_info['name']}")
    logger.info(f"Model created: {model_info['created_time']}")
    logger.info(f"Model modified: {model_info['modified_time']}")

    # 初始化推理器 (pass config_path to use the same config)
    inferencer = RuleEmbeddingInference(model_path, config_path=config_path)

    # 加載模型
    if not inferencer.load_model():
        logger.error("Failed to load model")
        return

    try:
        # 讀取測試數據
        df, rules, true_labels = fetch_data(data_path)

        logger.info(f"Loaded {len(rules)} rules for inference")

        # Perform comprehensive parameter grid search
        if method == 'dbscan':
            optimization_result = perform_dbscan_grid_search(inferencer, rules, true_labels)
        elif method == 'hdbscan':
            optimization_result = perform_hdbscan_grid_search(inferencer, rules, true_labels)
        else:
            logger.warning(f"Grid search not implemented for method: {method}")
            optimization_result = None
        
        # Use the best parameters found
        if optimization_result and optimization_result['best_params']:
            best_params = optimization_result['best_params']
            logger.info(f"Best parameters found: {best_params}")
            logger.info(f"Best AMI score: {optimization_result['best_ami']:.4f}")
            
            # Run clustering with best parameters
            clustering_result = inferencer.cluster_rules(rules, method=method, optimize_params=False, **best_params)
            used_params = best_params
        else:
            # Fallback to original approach
            clustering_result = inferencer.cluster_rules(rules, method=method, optimize_params=optimize_params,
                                eps_range=(0.1, 0.8), n_eps_values=25,
                                reference_method='kmeans',
                                **kwargs,
                                )
            used_params = 'Default/Optimized'
        
        df['predicted_cluster'] = clustering_result['cluster_labels']
        print(f"Found {clustering_result['n_clusters']} clusters")
        print(f"Used parameters: {used_params}")
        
        # Create visualizations
        if optimization_result:
            output_dir = config.get('output', {}).get('results_dir', 'results')
            plot_parameter_optimization_results(optimization_result, method, output_dir=output_dir)
        
        # 繪製優化結果
        if 'eps_optimization' in clustering_result:
            inferencer.plot_eps_optimization_results(clustering_result['eps_optimization'])
        
        from pathlib import Path
        from datetime import datetime
        output_dir = config.get('output', {}).get('results_dir', 'results')
        Path.mkdir(Path(output_dir), exist_ok=True)
        df.rename(columns={'description': 'Description', 'group_id':'Ground Truth'}, inplace=True)
        date_str = datetime.now().strftime('%m%d')
        df.to_csv(f'{output_dir}/inference_results_{date_str}.csv', index=False)
        
        # 3. 評估聚類效果
        if true_labels:
            predicted_labels = clustering_result['cluster_labels']
            metrics = inferencer.evaluate_clustering(rules, true_labels, predicted_labels)
        
        logger.info("evaluation completed successfully!")
        
        return optimization_result
        
    except Exception as e:
        logger.error(f"evaluation failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='DBSCAN and HDBSCAN parameter optimization with AMI score evaluation')
    parser.add_argument('--method', type=str, default='dbscan', choices=['dbscan', 'hdbscan', 'kmeans'],
                        help='Clustering method to use (default: dbscan)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model. If None, will auto-detect the latest model')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the CSV data file. If None, uses default path')
    parser.add_argument('--optimize_params', action='store_true', default=True,
                        help='Whether to optimize parameters (default: True)')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='DBSCAN eps parameter (default: 0.1)')
    parser.add_argument('--min_samples', type=int, default=2,
                        help='DBSCAN/HDBSCAN min_samples parameter (default: 2)')
    parser.add_argument('--min_cluster_size', type=int, default=2,
                        help='HDBSCAN min_cluster_size parameter (default: 2)')
    
    args = parser.parse_args()
    
    # Display configuration
    print("=== Configuration ===")
    print(f"Method: {args.method}")
    print(f"Model path: {args.model_path if args.model_path else 'Auto-detect latest'}")
    print(f"Data path: {args.data_path if args.data_path else 'Default path'}")
    print(f"Optimize parameters: {args.optimize_params}")
    print(f"DBSCAN eps: {args.eps}")
    print(f"Min samples: {args.min_samples}")
    print(f"HDBSCAN min_cluster_size: {args.min_cluster_size}")
    print("=" * 50)
    
    # Run evaluation with specified parameters
    result = main_evaluation(
        model_path=args.model_path,
        data_path=args.data_path,
        method=args.method,
        optimize_params=args.optimize_params,
        eps=args.eps,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size
    )
    
    # Display results summary
    if result:
        print("\n=== Results Summary ===")
        print(f"Method: {args.method}")
        print(f"Best parameters: {result.get('best_params', 'N/A')}")
        print(f"Best AMI score: {result.get('best_ami', 'N/A'):.4f}")
        print("Optimization completed successfully!")
    else:
        print("No optimization results available.")
