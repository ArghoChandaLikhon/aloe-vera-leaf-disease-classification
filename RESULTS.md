# Results

Full experimental results for the Aloe Vera Leaf Disease Classification project. All numbers below are from a single, fixed test set (525 images, stratified 15% holdout, seed=42), so every model in these tables was evaluated under identical conditions.

## Table 1 — Overall Model Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| ResNet18 | 97.14% | 0.9731 | 0.9715 | 0.9722 |
| EfficientNet-B0 | 97.33% | 0.9744 | 0.9747 | 0.9743 |
| EfficientNetB3 | 97.52% | 0.9761 | 0.9752 | 0.9755 |
| MobileNetV2 | 97.71% | 0.9777 | 0.9771 | 0.9772 |
| **Ensemble (all 4)** | **98.86%** | **0.9895** | **0.9883** | **0.9889** |

## Table 2 — Per-Class Breakdown (Precision / Recall / F1)

| Model | Healthy | Rot | Rust |
|---|---|---|---|
| ResNet18 | 0.9935 / 0.9935 / 0.9935 | 0.9695 / 0.9408 / 0.9550 | 0.9563 / 0.9801 / 0.9681 |
| EfficientNet-B0 | 1.0000 / 0.9806 / 0.9902 | 0.9435 / 0.9882 / 0.9653 | 0.9796 / 0.9552 / 0.9673 |
| EfficientNetB3 | 1.0000 / 0.9742 / 0.9869 | 0.9483 / 0.9763 / 0.9621 | 0.9800 / 0.9751 / 0.9776 |
| MobileNetV2 | 0.9747 / 0.9935 / 0.9840 | 0.9877 / 0.9527 / 0.9699 | 0.9706 / 0.9851 / 0.9778 |
| **Ensemble** | 1.0000 / 0.9935 / 0.9968 | 0.9880 / 0.9763 / 0.9821 | 0.9804 / 0.9950 / 0.9877 |

## Table 3 — Two-Stage Hierarchical Pipeline Investigation

Motivated by confusion-matrix evidence that rot/rust was the persistent hard boundary across all four backbones.

| Configuration | Accuracy | Notes |
|---|---|---|
| Stage 1 alone (healthy vs. diseased) | 98.67% | Very strong — this split is easy |
| Stage 2 alone (rot vs. rust) | 93.51% | The real bottleneck |
| Full pipeline (hard cascade) | 94.10% | Underperforms flat models |
| Full pipeline (soft, probability-weighted) | 94.10% | Identical to hard cascade — confirms the issue is Stage 2's raw discriminative ability, not how the two stages combine |

**Conclusion:** the hierarchical approach underperformed flat classification, most likely because the Stage 2 specialist loses access to healthy images during training — images that may otherwise help the model learn general leaf-texture features that indirectly aid rot/rust discrimination.

## Statistical Note

The ensemble's improvement over the best individual model (MobileNetV2) did not reach conventional statistical significance (McNemar's exact test, p = 0.109) — likely because only 10 of 525 test images were cases where the two models disagreed. Reported as a numerically higher, honestly-caveated improvement rather than a statistically proven one.

## Figures

- `kaggle/results/gradcam_examples.png` — Grad-CAM visualizations on correct and misclassified predictions
- `kaggle/results/mobilenetv2_100_confusion_matrix.png`
- `kaggle/results/efficientnet_b0_confusion_matrix.png`
- `kaggle/results/resnet18_confusion_matrix.png`
- `kaggle/results/efficientnet_b3_confusion_matrix.png`
- `kaggle/results/ensemble_confusion_matrix.png`
