# Clothes Segmentation — DeepLabV3+ (ResNet50)

A deep-learning pipeline that segments clothes and body parts from images of people.

The model performs **18-class human/clothing parsing** (background, clothing items, accessories, body parts) using a **DeepLabV3+ architecture with a ResNet50 backbone**, trained on the [`mattmdjaga/human_parsing_dataset`](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset) (ATR dataset, 17,706 images).

See [`REPORT.md`](REPORT.pdf) for the full write-up (dataset choice, architecture, loss function, evaluation, and limitations).

---

## Repository contents

| File | Description |
|---|---|
| `clothes-segmentation.ipynb` | End-to-end notebook: data loading, preprocessing, augmentation, model, training, evaluation, inference |
| `REPORT.md` | Full technical report required by the assignment |
| `requirements.txt` | Python dependencies |
| `human_parsing_unet_18_classes.keras` | Final trained model weights (produced after training — see note below) |
| `best_unet.keras` | Best checkpoint saved during training (lowest `val_loss`) |

> **Note on weights:** the `.keras` weight files are large binaries. If they are not committed to this repository (e.g. due to GitHub file-size limits), re-run the notebook end-to-end to reproduce them, or download them from the release/link provided separately, and place them in the repository root before running inference.

---

## 1. Setup

**Requirements:** Python 3.10+, and ideally a CUDA-capable GPU (training on CPU is possible but slow).

```bash
git clone <your-repo-url>
cd <your-repo-name>

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt`:
```
tensorflow>=2.15.0
datasets>=2.19.0
numpy>=1.26.0
pillow>=10.0.0
matplotlib>=3.8.0
```

The dataset is streamed automatically from the Hugging Face Hub the first time the notebook runs (no manual download needed):

```python
from datasets import load_dataset
dataset = load_dataset("mattmdjaga/human_parsing_dataset")
```

## 2. Running the pipeline

Launch Jupyter and run `clothes-segmentation.ipynb` top to bottom:

```bash
jupyter notebook clothes-segmentation.ipynb
```

The notebook is organized into clearly separated sections:

1. **Dataset loading & exploration** — loads the dataset, inspects class IDs, visualizes sample image/mask pairs.
2. **Preprocessing & splitting** — 80% / 10% / 10% train/val/test split, resize to 512×512, normalize images to `[0, 1]`, nearest-neighbor resize for masks.
3. **Augmentation** — random horizontal flip, brightness, and contrast jitter (training split only).
4. **Model** — builds `DeepLabV3+` with a `ResNet50` (ImageNet-pretrained) backbone and an ASPP module.
5. **Loss & metrics** — a combined weighted focal + weighted multiclass Dice loss, plus a custom mean-IoU metric.
6. **Training** — `model.fit(...)` with checkpointing, learning-rate reduction, and early stopping.
7. **Evaluation** — loss/accuracy/mIoU curves, per-class IoU on the held-out test set, qualitative predictions.
8. **Inference on a new image** — `predict_image(model, image_path)` runs the trained model on any external photo.

## 3. Reproducing the results

To retrain from scratch, simply run all cells. Key configuration (top of the notebook, easy to change):

```python
IMAGE_SIZE   = (512, 512)
NUM_CLASSES  = 18
BATCH_SIZE   = 8
EPOCHS       = 6
LEARNING_RATE = 1e-3
SEED         = 42
```

The random seed (`42`) is fixed for NumPy, Python's `random`, and TensorFlow, and the dataset split uses the same seed, so re-running the notebook reproduces the same train/val/test partition and closely-comparable results.

Training saves two checkpoints automatically:
- `best_unet.keras` — best epoch by `val_loss` (via `ModelCheckpoint`)
- `human_parsing_unet_18_classes.keras` — final model saved at the end of training

## 4. Running inference on your own image

```python
import tensorflow as tf
from predict_utils import predict_image   # or copy the function from the notebook

model = tf.keras.models.load_model(
    "human_parsing_unet_18_classes.keras",
    custom_objects={"combined_loss": combined_loss, "mean_iou_metric": mean_iou_metric}
)

original, pred_mask = predict_image(model, "path/to/your_photo.jpg")
```

`pred_mask` is a `512×512` array of class IDs (0–17); see [`REPORT.md`](REPORT.md) for the full label map and recommended capture conditions for reliable results.

## 5. Headline results (test set)

| Metric | Value |
|---|---|
| Test loss (combined focal + Dice) | 0.396 |
| Test pixel accuracy | 93.1% |
| Test mean IoU (Keras metric, per-batch) | 0.513 |
| Test mean IoU (per-class, full test set) | 0.534 |

Full per-class breakdown, strengths, weaknesses, and limitations are in [`REPORT.md`](REPORT.md).
