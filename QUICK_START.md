# Quick Start Guide - Inference Configuration

## Current Status

✓ Configuration system is ready
✓ Model auto-detection is working
✓ Found 2 models in `./weights/`:
  - `rule-embedder-0722_trained_MPnet_CachedMN`
  - `rule-embedder-0828_trained_MPnet_CachedMN` (latest)

## Running Inference (3 Ways)

### 1. Default - Auto-Detect Model

```bash
cd /home/htilab04/DRC_GUI/ai
python inference.py --method dbscan
```

Uses:
- Auto-detected latest model from `./weights/` directory
- Default settings from `inference_config.json`

### 2. Specify Model Path

```bash
python inference.py \
  --model_path ./weights/rule-embedder-0828_trained_MPnet_CachedMN \
  --method dbscan
```

### 3. Full Configuration

```bash
python inference.py \
  --model_path ./weights/rule-embedder-0828_trained_MPnet_CachedMN \
  --data_path ../data/your_data.csv \
  --method dbscan \
  --optimize_params
```

## Configuration Files

| File | Purpose |
|------|---------|
| `inference_config.json` | Main config (paths, clustering params) |
| `CONFIG_README.md` | Full parameter documentation |
| `SETUP_PATHS.md` | Path configuration guide |
| `bug_history/` | Bug fixes and issue resolution documentation |

## Common Tasks

### Check Current Configuration

```bash
python3 << 'EOF'
import json
with open('inference_config.json') as f:
    config = json.load(f)
print(json.dumps(config.get('paths', {}), indent=2))
EOF
```

### Find Available Models

```bash
ls -la weights/ | grep rule-embedder
```

### Update Model Path in Config

Edit `inference_config.json`:
```json
{
  "paths": {
    "model_path": "/full/path/to/your/model"
  }
}
```

### Change Clustering Method

Edit `inference_config.json`:
```json
{
  "clustering": {
    "default_method": "hdbscan"
  }
}
```

Or use command line:
```bash
python inference.py --method hdbscan
```

## Parameter Customization

### Quick: Edit Config File

```bash
nano inference_config.json
```

Common changes:
- `clustering.default_method`: "dbscan", "hdbscan", "kmeans", "hierarchical"
- `clustering.dbscan.eps`: 0.1 to 1.0 (smaller = more clusters)
- `clustering.dbscan.min_samples`: 2 to 10 (minimum points per cluster)
- `model.batch_size`: 16, 32, 64 (affects memory usage)

### Advanced: Create Custom Config

```bash
cp inference_config.json my_config.json
nano my_config.json
python inference.py --config_path my_config.json
```

## Troubleshooting

### Error: No model found
```bash
# Set explicit path in config
nano inference_config.json
# Add: "model_path": "/full/path/to/model"
```

### Error: Data file not found
```bash
# Option 1: Set in config
nano inference_config.json
# Add: "data_path": "/path/to/data.csv"

# Option 2: Use command line
python inference.py --data_path /path/to/data.csv
```

### Error: Config file not found
```bash
# Check you're in the right directory
cd /home/htilab04/DRC_GUI/ai
ls -la inference_config.json
```

## Getting Help

```bash
# View all parameters
cat CONFIG_README.md

# View path setup guide
cat SETUP_PATHS.md

# View bug fix history
ls bug_history/

# Check current setup
python3 -c "
import json
with open('inference_config.json') as f:
    config = json.load(f)
print('Paths:', config.get('paths'))
print('Clustering:', config.get('clustering', {}).get('default_method'))
"
```

## What's Configured

All previously hardcoded values are now configurable:

✓ Model and data paths
✓ Clustering methods (KMeans, DBSCAN, HDBSCAN, Hierarchical)
✓ Optimization parameters
✓ Similarity search settings
✓ Noise handling strategies
✓ Output formats and locations

## Priority System

When a parameter is needed:
1. **Command line argument** (if provided) ← Highest
2. **Config file value** (if set)
3. **Default value** (fallback) ← Lowest

Example:
```bash
# Config has: eps=0.3
# Command: python inference.py --eps 0.5
# Result: Uses 0.5 (command line wins)
```

## Next Steps

1. **For production**: Set explicit `model_path` in config
2. **For experiments**: Create multiple config files
3. **For automation**: Use command line arguments

---

**Ready to use!** The system is fully configured and tested.
