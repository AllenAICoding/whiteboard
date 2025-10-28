# Cluster Embedding Module

A comprehensive embedding training and clustering framework for semiconductor design rule analysis, featuring advanced negative mining, real-time validation analysis, and automatic model management.

## Overview

This module implements a complete pipeline for training embedding models specifically designed for clustering semiconductor design rules. It combines traditional machine learning clustering with deep learning embeddings to achieve high-quality rule grouping with AMI scores of 0.86-0.89.

## Architecture

### Core Components

- **`train.py`**: Main training pipeline with enhanced metrics tracking and validation analysis
- **`data_prepare.py`**: Advanced data preparation with hard negative mining and text augmentation
- **`inference.py`**: Comprehensive inference engine with parameter optimization and automatic model detection
- **`model_manager.py`**: Model management utilities for validation and comparison

### Key Features

- **Hard Negative Mining**: Intelligent selection of challenging negative examples during training
- **Real-time Validation Analysis**: Deep clustering analysis during training with problem diagnosis
- **Automatic Model Detection**: Smart model selection based on modification time and validation
- **Parameter Optimization**: Grid search for DBSCAN and HDBSCAN clustering parameters
- **Comprehensive Metrics**: AMI, NMI, Silhouette scores with detailed logging

## Quick Start

### Basic Training
```bash
# Train with default configuration
python train.py

# Monitor training progress in real-time
tail -f training.log
```

### Inference and Clustering
```bash
# Run inference with automatic model detection
python inference.py

# Use specific clustering method
python inference.py --method dbscan

# Specify model path
python inference.py --model_path ./rule-embedder-0717_trained_MPnet_HNM
```

### Model Management
```bash
# List all available models
python model_manager.py --list

# Validate a specific model
python model_manager.py --validate ./model_path

# Clean up old models (keep 3 latest)
python model_manager.py --cleanup 3
```

### Training Visualization
```bash
# Plot training metrics with dynamic scaling
python plot_training_metrics.py training_metrics.xlsx

# Compare multiple training runs
python plot_training_metrics.py file1.xlsx file2.xlsx --compare

# Use fixed scale instead of dynamic
python plot_training_metrics.py training_metrics.xlsx --fixed-scale

# Auto-detect and plot existing metrics
python plot_training_metrics.py
```

## Configuration

### Training Parameters
```python
config = {
    'model_name': 'sentence-transformers/all-mpnet-base-v2',
    'num_epochs': 10,
    'learning_rate': 1e-5,  # Reduced to prevent embedding collapse
    'enhanced_triplet': True,
    'difficulty_strategy': 'mixed',
    'loss_type': 'cached_multiple_negatives',  # New: memory-efficient cached loss
    'output_path': './rule-embedder-MMDD_trained_MPnet_CachedMN'
}
```

### Loss Function Options
- **`triplet`**: Traditional triplet loss with cosine distance
- **`circle`**: Circle loss (if available in sentence-transformers)
- **`multi_similarity`**: Multi-Similarity loss from pytorch-metric-learning
- **`multiple_negatives`**: Standard Multiple Negatives Ranking Loss
- **`cached_multiple_negatives`**: Memory-efficient cached version (recommended)
- **`cosine`**: Simple cosine similarity loss

### Data Preparation Settings
- **Hard negative ratio**: 0.7 (70% hard negatives)
- **Max triplets per anchor**: 24
- **Top-k hard negatives**: 10
- **Difficulty ratio**: 0.6

## Validation Analysis

The system provides comprehensive validation analysis every 2 epochs or when AMI < 0.1:

### Automatic Problem Detection
- **Embedding Collapse**: Same-class similarity = 1.0, std = 0.0
- **Clustering Collapse**: Only 1 cluster predicted
- **Over-fragmentation**: Too many small clusters
- **Poor Separation**: Same/different class similarities too close

### Saved Analysis Data
All validation analysis is automatically saved to `./validation_analysis/`:
- **CSV files**: Sentence-level analysis with embeddings and similarities
- **NumPy arrays**: Raw embeddings and similarity matrices
- **JSON metrics**: Complete clustering statistics and diagnostics

## Performance Optimization

### Clustering Parameter Grid Search
```python
# DBSCAN optimization
eps_range = np.linspace(0.05, 0.9, 20)
min_samples_range = range(2, 11)

# HDBSCAN optimization  
min_cluster_size_range = range(2, 21)
min_samples_range = range(1, 11)
```

### Embedding Quality Metrics
- **Same-class similarity**: Target > 0.8, std > 0.05
- **Different-class similarity**: Target < 0.6
- **Similarity margin**: Target > 0.2

## Version History

### Version 0722 (Current) - Cached Loss & Dynamic Visualization
**Major Features:**
- **CachedMultipleNegativesRankingLoss**: Memory-efficient cached loss function supporting larger batch sizes (16-24 vs 8)
- **Dynamic Loss Visualization**: Automatic loss scale adaptation for different loss functions (triplet: 0.2-1.0, multi-similarity: 1.0-3.0, circle: 0.01-0.05)
- **Enhanced Training Plotting**: Real-time training progress with adaptive Y-axis scaling and selective annotations
- **Memory-Optimized Training**: Mini-batch processing within cached loss to handle larger effective batch sizes

**Key Improvements:**
- Added `cached_multiple_negatives` loss type with intelligent batch sizing (8-24 based on GPU memory)
- Implemented `plot_training_metrics.py` for post-training analysis with dynamic scaling
- Enhanced `MetricsTracker` with automatic loss range detection and appropriate tick intervals
- Memory-efficient similarity computation using `mini_batch_size=16` parameter
- Selective annotation display (every other point) to reduce plot clutter
- Conservative memory management with fallback mechanisms for different GPU sizes

**Technical Details:**
- **Batch Size Logic**: 24 (16GB+ GPU), 16 (8GB+ GPU), 8 (<8GB GPU) for cached loss
- **Mini-batch Processing**: Internal chunking of similarity calculations for memory efficiency
- **Dynamic Scaling**: Automatic Y-axis range calculation with 10-15% margins around min/max values
- **Visualization Enhancements**: Separate plotting utility supporting single/multiple training run comparisons

### Version 0718 - Multi-Loss Support & Batch Optimization
**Major Features:**
- **Multiple Loss Functions**: Support for TripletLoss, CircleLoss, MultipleNegativesRankingLoss, CosineSimilarityLoss, and Multi-Similarity Loss
- **Dynamic Batch Sizing**: Automatic batch size optimization based on GPU memory and loss type (2-4x larger batches for MS Loss)
- **Enhanced Loss Configuration**: Flexible loss type selection with fallback mechanisms
- **Improved Training Stability**: Better error handling and JSON serialization fixes

**Key Improvements:**
- Multi-Similarity Loss integration with pytorch-metric-learning
- GPU memory adaptive batch sizing (4-64 range)
- Automatic hard negative mining for MS Loss
- Enhanced parameter configuration and logging
- Fixed validation analysis data serialization issues

### Version 0717 - Enhanced Analysis & Stability
**Major Features:**
- **Fixed Embedding Collapse**: Reduced learning rate from 6e-5 to 1e-5
- **Enhanced Validation Analysis**: Real-time clustering diagnosis with problem detection
- **Automatic Data Saving**: Complete validation analysis saved for offline examination
- **Duplicate Logging Fix**: Removed redundant log messages, clean callback-only reporting
- **Improved Diagnostics**: Detailed similarity analysis and clustering problem identification

**Key Improvements:**
- Embedding collapse detection and prevention
- Comprehensive validation set analysis with similarity statistics
- Automatic problem diagnosis (clustering collapse, over-fragmentation, poor separation)
- Per-sentence similarity margin calculation
- Cluster purity analysis and confusion mapping

### Version 0716 - Advanced Infrastructure
**Major Features:**
- **Parameter Grid Search**: Comprehensive DBSCAN and HDBSCAN optimization
- **Automatic Model Detection**: Smart model selection with `find_latest_model()`
- **Command Line Interface**: Argparse support for method and model_path
- **Enhanced Visualization**: Parameter optimization plots and heatmaps
- **Model Management**: Validation, comparison, and cleanup utilities

**Key Improvements:**
- Grid search for eps (0.05-0.9) and min_samples (2-11) in DBSCAN
- Grid search for min_cluster_size (2-21) and min_samples (1-11) in HDBSCAN
- Automatic model detection based on file modification time
- Model validation with required file checking
- Comprehensive visualization of parameter vs AMI relationships

### Version 0715 - Hard Negative Mining
**Major Features:**
- **Hard Negative Mining**: Advanced negative example selection
- **Text Augmentation**: Enhanced data diversity through text transformations
- **Enhanced Triplet Generation**: Improved anchor-positive-negative triplet quality
- **Similarity-based Mining**: Sentence transformer-based hard negative selection

**Key Improvements:**
- Hard negative ratio of 0.7 for challenging training examples
- Top-k (10) hard negative selection per anchor
- Text augmentation with paraphrasing and layer name extraction
- Triplet quality analysis with margin statistics
- Enhanced data splitting ensuring no rule overlap between train/validation

### Version 0713 - Foundation
**Major Features:**
- **Basic Training Pipeline**: SentenceTransformer-based embedding training
- **TripletLoss Implementation**: Cosine distance with 0.5 margin
- **Simple Clustering**: Basic DBSCAN and K-means support
- **Evaluation Framework**: AMI-based model evaluation

**Key Improvements:**
- MPNet base model integration
- Basic triplet loss training
- Simple clustering evaluation
- Initial metrics tracking

## Troubleshooting

### Common Issues

1. **Embedding Collapse (Same_Class_Std = 0)**
   - **Cause**: Learning rate too high
   - **Solution**: Reduce learning rate (use 1e-5 instead of 6e-5)

2. **Clustering Collapse (Only 1 cluster)**
   - **Cause**: DBSCAN eps too small or embeddings too similar
   - **Solution**: Use parameter grid search or increase eps

3. **Poor AMI Scores (< 0.3)**
   - **Cause**: Wrong clustering parameters or poor embeddings
   - **Solution**: Enable parameter optimization or retrain with different loss

4. **Training Memory Issues**
   - **Cause**: Batch size too large for GPU
   - **Solution**: Reduce batch size (8 for <8GB GPU, 16 for 8GB+, 24 for 16GB+)
   - **For Cached Loss**: Reduce `mini_batch_size` parameter if needed

5. **Loss Scale Visualization Issues**
   - **Cause**: Fixed Y-axis scale inappropriate for loss function
   - **Solution**: Use `plot_training_metrics.py` with dynamic scaling

### Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Monitor Validation**: Watch for embedding collapse early in training
3. **Parameter Search**: Enable grid search for optimal clustering
4. **Data Quality**: Ensure balanced training data across clusters

## File Structure

```
cluster_embedding/
├── README.md                    # This file
├── train.py                     # Main training pipeline with cached loss support
├── data_prepare.py             # Data preparation and hard negative mining
├── inference.py                # Inference engine with parameter optimization
├── plot_training_metrics.py    # Training visualization with dynamic scaling
├── model_manager.py            # Model management utilities
├── AUTO_MODEL_DETECTION.md     # Automatic model detection documentation
├── training.log                # Training logs with detailed metrics
├── training_metrics.xlsx       # Training metrics Excel file
├── validation_analysis/        # Saved validation analysis data
│   ├── validation_analysis_epoch_*.csv
│   ├── embeddings_epoch_*.npy
│   ├── similarity_matrix_epoch_*.npy
│   └── clustering_metrics_epoch_*.json
└── rule-embedder-*/           # Trained model directories
```

## Dependencies

### Core Requirements
- `sentence-transformers>=2.0.0`
- `torch>=1.9.0`
- `scikit-learn>=1.0.0`
- `pandas>=1.3.0`
- `numpy>=1.21.0`

### Optional (for enhanced features)
- `hdbscan>=0.8.27` (for HDBSCAN clustering)
- `matplotlib>=3.5.0` (for visualization)
- `faiss-gpu` (for fast similarity search)

## Contributing

When contributing to this module:

1. **Maintain backward compatibility** in public APIs
2. **Add comprehensive logging** for new features
3. **Include validation analysis** for clustering changes
4. **Update version history** in this README
5. **Test with small datasets** before full runs

## Performance Benchmarks

### Typical Results
- **AMI Score**: 0.86-0.89 (with optimal parameters)
- **Training Time**: ~2-3 hours (10 epochs, 590 samples)
- **Memory Usage**: ~4-8GB GPU memory
- **Validation Analysis**: ~30MB per epoch

### Hardware Recommendations
- **GPU**: 8GB+ VRAM (RTX 3070/4060 or better)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space for models and analysis data
