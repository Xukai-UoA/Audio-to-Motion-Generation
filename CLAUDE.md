# CLAUDE.md - Audio-to-Motion-Generation

This document provides guidance for AI assistants working with this codebase.

## Project Overview

**Audio-to-Motion-Generation** is a deep learning system that generates 2D human body motion (pose sequences) from audio input. It uses a GAN-based architecture with Graph Convolutional Networks (GCN) to create realistic gestures synchronized with speech.

**Status**: Test version / Research prototype

## Repository Structure

```
Audio-to-Motion-Generation/
├── pats/                           # Data handling package
│   ├── data/                       # Data storage
│   │   ├── cmu_intervals_df.csv    # Main metadata (84K rows)
│   │   ├── h5_loader.py            # HDF5 loading utilities
│   │   └── h5_processor.py         # HDF5 processing
│   ├── data_loading/               # Modular data loaders
│   │   ├── dataUtils.py            # Core Data_Loader class
│   │   ├── common.py               # Base Modality abstract class
│   │   ├── skeleton.py             # Skeleton2D (52 joints)
│   │   ├── audio.py                # Audio feature extraction
│   │   ├── text.py                 # NLP features (BERT/Word2Vec)
│   │   └── skeleton_tree.txt       # Joint hierarchy definition
│   └── argsUtils.py                # CLI argument parsing
│
├── pose_video/                     # Visualization & utilities
│   ├── consts.py                   # Speaker configs & constants
│   ├── audio_lib.py                # Audio extraction
│   ├── mel_features.py             # Mel-spectrogram features
│   ├── evaluation.py               # PCK metric implementation
│   ├── pose_plot_lib.py            # Pose visualization
│   └── speaker_config.py           # Speaker-specific configs
│
├── Core Model Files
│   ├── real_motion_model.py        # Generator & Discriminator
│   ├── model_layers.py             # Neural network components
│   ├── version5_model_train.py     # Main training pipeline
│   └── generate_motion_video.py    # Inference & video generation
│
└── Utilities
    ├── normalization_tools.py      # Data normalization
    ├── motion_evaluation.py        # PCK evaluation
    └── dataset_exception_error_diagnosis.py  # Data debugging
```

## Key Architecture

### Model Components

1. **Generator (`SelfAttention_G`)**:
   - Input: Audio mel-spectrogram (B, T, 128)
   - Pipeline: AudioEncoder → UNet1D → Body GCN + Hand GCN
   - Output: 2D Pose sequences (B, T, 104) - 52 joints × 2 coords

2. **Discriminator (`SelfAttention_D`)**:
   - Input: Motion sequences (temporal difference of poses)
   - Uses spectral normalization and self-attention
   - Multi-scale classification

3. **Graph Neural Networks**:
   - Body Decoder: 10 body joints with GAT/GraphConv layers
   - Hand Decoder: 42 hand joints (21 per hand)
   - Internal losses: bone length + joint angle constraints

### Data Specifications

- **Audio**: 16kHz, log-mel spectrogram (128 bins, 512 hop)
- **Pose**: 52 joints (10 body + 42 hands) in 2D coordinates
- **Time**: 64 frames per window (~4.3s at 15Hz)
- **Normalization**: Neck-subtraction method (pose relative to neck joint)

### Supported Speakers

8 speakers from CMU dataset: oliver, noah, seth, shelly, ellen, angelica, almaram, chemistry

## Development Workflows

### Training

```bash
python version5_model_train.py
```

Key parameters in `version5_model_train.py`:
- `SPEAKER = 'multi_speaker'` - train on multiple speakers
- `N_EPOCHS = 500`
- `BATCH_SIZE = 128`
- `LR_GENERATOR = 5e-6`
- `LR_DISCRIMINATOR = 10e-6`

Models saved to: `./save/{SPEAKER}/`

### Inference

```bash
python generate_motion_video.py
```

Generates motion sequences from audio and visualizes them.

### Data Debugging

```bash
python dataset_exception_error_diagnosis.py
```

Validates data loading pipeline and batch shapes.

## Code Conventions

### Import Structure

Core dependencies (no requirements.txt - managed implicitly):
```python
# Deep Learning
torch, torch.nn, torch.nn.functional
torch_geometric (GraphConv, GATConv)
transformers (BertModel, BertTokenizer)

# Data Processing
numpy, pandas, h5py
librosa, webrtcvad

# Utilities
scipy, matplotlib, PIL, nltk, tqdm
```

### Naming Conventions

- Model classes: `PascalCase` (e.g., `SelfAttention_G`, `UNet1D`)
- Functions: `snake_case` (e.g., `get_mean_std_necksub`)
- Constants: `UPPER_CASE` (e.g., `SPEAKER`, `N_EPOCHS`)
- Private methods: `_underscore_prefix`

### Documentation Style

- Code comments are mixed English/Chinese
- Chinese comments explain complex architectural decisions
- Mathematical operations documented inline

### Data Processing Patterns

1. **Lazy loading**: HDF5-based with interval indexing
2. **Modular design**: Abstract `Modality` base class
3. **Multi-speaker**: Concatenated datasets with speaker tags
4. **Normalization**: Always normalize pose relative to neck joint

## Common Tasks

### Adding a New Modality

1. Create new class in `pats/data_loading/` inheriting from `Modality`
2. Implement `__init__`, `load`, and `normalize` methods
3. Register in `Data_Loader` class in `dataUtils.py`

### Modifying Network Architecture

Key files:
- `model_layers.py` - Reusable components (ConvNormRelu, ResBlock, etc.)
- `real_motion_model.py` - Generator/Discriminator architecture

### Adding New Loss Functions

Modify training loop in `version5_model_train.py`:
- Generator losses: Line ~200-250
- Discriminator losses: Line ~300-350
- Internal losses (bone/angle constraints): Built into model forward pass

### Changing Training Strategy

`DynamicGANTraining` class in `version5_model_train.py`:
- Adjusts G/D training frequency based on loss ratios
- Implements label smoothing and noise injection
- Controls learning rate adaptation

## Important Technical Details

### Skeleton Structure (52 joints)

Body (10 joints): Neck (root), shoulders, elbows, wrists, nose, eyes
Hands (42 joints): 5 fingers × 4 segments × 2 hands

Joint hierarchy defined in: `pats/data_loading/skeleton_tree.txt`

### Graph Construction

Adjacency matrix built from parent-child relationships:
- Body: Upper body kinematic chain
- Hands: Finger joint chains with fixed angles (0-180°)

### Training Dynamics

The system uses **dynamic GAN training**:
- Monitors rolling loss windows
- Auto-adjusts training frequencies to balance G/D
- Applies label smoothing that anneals over epochs
- Injects noise for stability

## Pitfalls & Warnings

1. **No requirements.txt**: Dependencies must be installed manually
2. **Large data files**: CSV metadata files can be 16MB+
3. **GPU memory**: Batch size 128 with 64 timesteps requires significant VRAM
4. **Speaker-specific**: Each speaker has different normalization statistics
5. **File paths**: Some paths may be hardcoded - check before running

## Testing & Validation

- **PCK (Percentage of Correct Keypoints)**: Primary evaluation metric
- Located in `pose_video/evaluation.py` and `motion_evaluation.py`
- Validation happens per-epoch during training

## File Size Reference

Largest files by code complexity:
- `model_layers.py`: ~44KB (all network components)
- `pats/data_loading/dataUtils.py`: ~36KB (data loader)
- `real_motion_model.py`: ~23KB (model architecture)
- `version5_model_train.py`: ~20KB (training loop)

## Quick Reference

| Task | File | Function/Class |
|------|------|----------------|
| Load data | `pats/data_loading/dataUtils.py` | `Data_Loader` |
| Define model | `real_motion_model.py` | `SelfAttention_G`, `SelfAttention_D` |
| Train model | `version5_model_train.py` | Main script |
| Generate motion | `generate_motion_video.py` | Main script |
| Evaluate | `motion_evaluation.py` | `compute_pck()` |
| Normalize pose | `normalization_tools.py` | `get_mean_std_necksub()` |

## Environment Setup (Recommended)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install core dependencies
pip install torch torchvision
pip install torch-geometric
pip install transformers
pip install numpy pandas h5py
pip install librosa webrtcvad
pip install scipy matplotlib pillow
pip install nltk tqdm resampy
```

## Notes for AI Assistants

- This is research code with mixed documentation quality
- Comments in Chinese explain complex ML concepts
- No formal testing framework - validation through training metrics
- Data paths may need adjustment for your environment
- HDF5 files not included in repo - must be downloaded separately
- Focus on understanding the GAN training dynamics when debugging
