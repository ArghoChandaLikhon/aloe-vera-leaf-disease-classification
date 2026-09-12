# Aloe Vera Leaf Disease Classification

Deep learning pipeline for classifying Aloe Vera leaves as **healthy**, **rot**, or **rust**, benchmarked against the recently published AloeVeraNet (Koli et al., 2025, *Expert Systems*) on the same dataset.

## Results

| Model | Test Accuracy |
|---|---|
| ResNet18 | 97.14% |
| EfficientNet-B0 | 97.33% |
| EfficientNetB3 | 97.52% |
| MobileNetV2 | 97.71% |
| **Ensemble (all 4 models)** | **98.86%** |
| *AloeVeraNet (Koli et al., 2025) — published benchmark* | *96.09%* |

The 4-model ensemble outperforms the previously published benchmark on this exact dataset. See `kaggle/results/gradcam_examples.png` for Grad-CAM visualizations showing what the model focuses on when making each prediction, including two real misclassified examples.

## What's in this repo

- **`config.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `utils.py`** — a lightweight local pipeline (MobileNetV2, runs on modest hardware) for quick baseline experiments.
- **`kaggle/aloevera-leaf-detection.ipynb`** — the full experiment: 4-backbone comparison, a two-stage hierarchical classification investigation, the final ensemble, and Grad-CAM explainability, all run on a Kaggle GPU.
- **`kaggle/results/`** — output figures, including Grad-CAM visualizations.

## Dataset

[Aloe Vera Images](https://www.kaggle.com/datasets/rubab123/aloe-vera-images) (Kaggle) — 3,495 images across 3 classes: healthy (1,033), rot (1,122), rust (1,340). Stratified 70/15/15 train/val/test split (seed=42), consistent across every experiment for fair comparison.

## Methodology summary

1. **Baseline comparison** — four ImageNet-pretrained backbones (MobileNetV2, EfficientNet-B0, ResNet18, EfficientNetB3) fine-tuned and evaluated under identical conditions.
2. **Two-stage hierarchical pipeline** — motivated by confusion-matrix evidence that rot/rust was the persistent hard boundary across all four backbones. Split into a healthy-vs-diseased stage followed by a rot-vs-rust specialist stage. Result: this underperformed the flat models (94.10% vs. 97.71%), likely because the specialist stage lost access to healthy images during training — a documented, analyzed negative result, not a discarded one.
3. **Ensemble** — averaging softmax outputs across all four flat models, which gave the best result overall.
4. **Explainability** — Grad-CAM applied to the best individual model (MobileNetV2) to visualize decision regions, including on misclassified examples.

## Running it

**Local baseline (lightweight, CPU/low-VRAM friendly):**
```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

**Full experiment:** open `kaggle/aloevera-leaf-detection.ipynb` on [Kaggle Notebooks](https://www.kaggle.com/code) with a GPU accelerator enabled, attach the [Aloe Vera Images dataset](https://www.kaggle.com/datasets/rubab123/aloe-vera-images), and run all cells.

## Author

Argho Chanda Likhon — [LinkedIn](https://www.linkedin.com/in/argho-likhon-122699192/)
