# Experimental Setup

## Dataset
- **Source:** [Aloe Vera Images](https://www.kaggle.com/datasets/rubab123/aloe-vera-images) (Kaggle)
- **Classes:** healthy (1,033), rot (1,122), rust (1,340) — 3,495 images total
- **Split:** stratified 70% train / 15% validation / 15% test, seed=42, applied identically across every experiment for fair comparison
- **No overlap:** train/val/test sets are disjoint at the image level; verified programmatically before any training began

## Preprocessing & Augmentation
- **Normalization:** ImageNet mean/std (standard for transfer learning from ImageNet-pretrained weights)
- **Training augmentation:** random horizontal flip, random rotation (±20°), color jitter (brightness/contrast/saturation ±0.2), random affine translation (±10%) and shear (±10°)
- **Validation/test:** resize only, no augmentation

## Models & Training
| Setting | Value |
|---|---|
| Backbones | MobileNetV2, EfficientNet-B0, ResNet18, EfficientNetB3 (all ImageNet-pretrained, fully fine-tuned) |
| Image size | 192×192 (MobileNetV2, EfficientNet-B0, ResNet18); 250×250 (EfficientNetB3) |
| Batch size | 32 (192px models); 24 (EfficientNetB3) |
| Optimizer | Adam, learning rate 1e-4, weight decay 1e-4 |
| Loss | Cross-entropy with inverse-frequency class weighting |
| LR schedule | ReduceLROnPlateau (factor 0.5, patience 2 epochs, on validation macro-F1) |
| Early stopping | Patience 5 epochs, on validation macro-F1 |
| Max epochs | 20 |
| Checkpoint selection | Best validation macro-F1 (not final epoch, not accuracy) |
| Mixed precision | Enabled (CUDA only) |
| Seed | 42 (Python, NumPy, PyTorch, CUDA) |

## Two-Stage Pipeline
- **Stage 1:** binary classifier, healthy vs. diseased (rot+rust merged), same backbone/hyperparameters as flat models
- **Stage 2:** binary classifier, rot vs. rust, trained only on diseased-class images from the same train split
- Both stages evaluated on the exact same held-out test indices as the flat models, for direct comparability

## Ensemble
- Simple averaging of softmax output probabilities across all 4 independently-trained flat models, no additional training required

## Explainability
- Grad-CAM applied to the final convolutional layer of the flat MobileNetV2 model

## Hardware & Software
- **Training environment:** Kaggle Notebooks, NVIDIA Tesla T4 GPU (CUDA 12.8)
- **Python:** 3.12.13
- **Key package versions:** torch 2.10.0+cu128, torchvision 0.25.0+cu128, timm 1.0.26, scikit-learn 1.6.1, numpy 2.0.2, matplotlib 3.10.0
- **Local baseline environment:** Intel i5-8265U, NVIDIA MX250 (2GB VRAM) — used for initial pipeline validation only, not final results

## Reproducibility
Full code, the executed Kaggle notebook, and all output figures are available at: [github.com/ArghoChandaLikhon/aloe-vera-leaf-disease-classification](https://github.com/ArghoChandaLikhon/aloe-vera-leaf-disease-classification)
