# UAMSF-Net: Uncertainty-Aware Visual Inspection of Pharmaceutical Tablets with Deep Ensembles

Code for the paper *Uncertainty-Aware Visual Inspection of Pharmaceutical Tablets with Deep Ensembles* (Pharmaceutics, 2026).

UAMSF-Net classifies a tablet image as normal or defective. Each network is a ConvNeXt-Tiny backbone with a multi-scale module (MSM) on stages 2–4 and an MLP head, trained with cross-entropy under the PharmaVision (PV) protocol. Three networks trained with seeds 7, 42 and 123 form a deep ensemble; its predictive entropy ranks tablets by uncertainty, and the most uncertain can be deferred for review.

## Files

| File | Contents |
|---|---|
| `augment.py` | Offline augmentation of the PV training images (2 copies per image, seed 7) |
| `data.py` | PV and pillQC loading, 224 × 224 resize, ImageNet normalization |
| `model.py` | UAMSF-Net (31,338,686 parameters) |
| `train.py` | Training under the PV protocol |
| `test.py` | Testing one network, the ensemble analysis, and latency |

## Installation

```bash
pip install -r requirements.txt
```

## Data

- **PV:** https://www.kaggle.com/datasets/tahirlee/pharmaceutical-tablets-dataset, arranged as `data/PV/{train,val,test}/{normal,defective}/`.
- **pillQC:** `git clone https://github.com/matlab-deep-learning/pillQC data/pillQC` (the normal and chip images are used; dirt is excluded).

Create the augmented training set, then copy the validation and test folders unchanged:

```bash
python augment.py --input_dir data/PV/train --output_dir data/PV_augmented/train
```

The result is `data/PV_augmented/{train,val,test}` with 1,140 / 60 / 200 images.

## Training

```bash
python train.py --data data/PV_augmented --seed 7
python train.py --data data/PV_augmented --seed 42
python train.py --data data/PV_augmented --seed 123
```

Adam (learning rate 1e-4, weight decay 1e-4), batch size 16, one warm-up epoch and cosine decay, up to 100 epochs. The checkpoint with the best validation F2-score at a decision threshold of 0.40 is kept, with early stopping after 15 epochs without improvement.

## Testing

```bash
python test.py --ckpt runs/seed7/best.pt   --data data/PV_augmented --pillqc data/pillQC/images
python test.py --ckpt runs/seed42/best.pt  --data data/PV_augmented --pillqc data/pillQC/images
python test.py --ckpt runs/seed123/best.pt --data data/PV_augmented --pillqc data/pillQC/images
python test.py --ensemble runs/seed7 runs/seed42 runs/seed123
python test.py --latency runs/seed7/best.pt runs/seed42/best.pt runs/seed123/best.pt
```

The ensemble analysis chooses the deferral rate on the validation set (0%, 10% or 20%) and writes `ensemble_results.txt`.

## Main results

| Setting | Result |
|---|---|
| PV test, one network (mean ± SD over seeds 7, 42, 123) | 92.33 ± 1.26% accuracy, 92.27 ± 1.11% F1 |
| PV test, ensemble | 94.00% accuracy, 93.88% F1 |
| pillQC, ensemble (no fine-tuning) | 94.27% accuracy, 93.02% recall, 87.91% F1 |
| Error-detection AUROC of the ensemble entropy | 0.837 (PV), 0.889 (pillQC) |
| Accuracy on the remaining tablets after deferring the 20% most uncertain | 97.50% (PV), 98.05% (pillQC) |
| Latency, RTX 3060, batch size 1 | 14.54 ms (one network), 47.01 ms (ensemble) |

GPU kernels are not bit-for-bit deterministic, so retraining can change the results slightly.

## Citation

```
[BibTeX entry to be added on publication]
```

## License

MIT License. See `LICENSE`.

## Contact

Tahir Mahmood, tahirmahmood@dongguk.edu
Division of Electronics and Electrical Engineering, Dongguk University, Seoul, Republic of Korea
