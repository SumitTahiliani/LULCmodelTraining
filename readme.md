# LULC Model Training Project

A machine learning project for training custom Land Use Land Cover (LULC) classification models using Sentinel-2 satellite imagery. This project is an evolution of the **LandCoverExplorer-India** project, moving from pre-loaded rasters to custom model training for improved accuracy and flexibility.

---

## Project Overview

This project focuses on training deep learning models for land cover classification using Sentinel-2 satellite imagery. Unlike the previous **LandCoverExplorer-India** project that relied on pre-loaded Google Dynamic World rasters, this approach trains custom models that can potentially achieve better accuracy for specific regions and use cases.

**Work in Progress**: This project is currently in active development. I am experimenting with different neural network architectures, loss functions, and training strategies to optimize model performance.

---

## Data Pipeline

### Data Sources
- **Sentinel-2 L2A**: High-resolution satellite imagery (10m resolution)
- **Dynamic World**: Google's land cover dataset for ground truth labels
- **Coverage**: Multiple Indian states (Uttar Pradesh, West Bengal, etc.)

### Data Processing
1. **Data Collection**: Automated Sentinel-2 data fetching using Sentinel Hub
2. **Patch Generation**: 256×256 pixel patches sampled from larger scenes
3. **Label Extraction**: Corresponding Dynamic World labels for each patch
4. **Data Splitting**: Train/validation splits for model training
5. **Format Conversion**: Conversion to NPZ format for efficient training

---

## Project Structure

```
LULCmodelTraining/
├── Scripts/
│   ├── getS2Data.py          # Sentinel-2 data collection
│   ├── trainUnet.py          # Main training script
│   ├── npzPrepData.py        # Data preparation utilities
│   ├── focalLoss.py          # Custom loss function implementation
│   ├── computeClassWeights.py # Class weight calculation
│   ├── splitData.py          # Data splitting utilities
│   ├── quickVis.py           # Visualization tools
│   └── dataQualityCheck.py   # Data validation scripts
├── dataset/                  # Training data (generated)
├── class_weights.json        # Pre-computed class weights
├── splits_unordered.json     # Data split indices
├── best_6cl_focalce_rarenpz_highlr.pt  # Best trained model
└── readme.md                 # This file
```

---

## Current Status & Future Work

### Completed
- Basic U-Net architecture implementation
- Sentinel-2 data collection pipeline
- Dynamic World label integration
- Focal Loss implementation
- Class weight balancing
- Training pipeline with early stopping

### In Progress
- Architecture experimentation (different encoders)
- Loss function optimization
- Data augmentation strategies
- Hyperparameter tuning

### Planned
- Model deployment pipeline
- Real-time inference capabilities
- Integration with LandCoverExplorer-India
- Multi-temporal analysis capabilities
- Transfer learning experiments
