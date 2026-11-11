# UAMSF-Net: Uncertainty-Aware Defect Detection for Pharmaceutical Tablet Quality Control

Code for the paper *Uncertainty-Aware Defect Detection for Pharmaceutical Tablet Quality Control with Deep Ensembles and Selective Review* ([citation to be added on publication]).

UAMSF-Net is a ConvNeXt-Tiny classifier with an adaptive multi-scale fusion module (AMSM). It labels a tablet image as normal or defective. Uncertainty comes from a three-seed deep ensemble, and tablets can be routed to manual review by that uncertainty.

The code follows the PharmaVision (PV) protocol: fixed splits, 224 × 224 inputs, the same optimiser and early stopping for every model, and the same metrics. It also evaluates zero-adaptation transfer to the pillQC dataset.

## Main results

| Setting | Result |
|---|---|
| PV test, UAMSF-Net single model, 3 seeds (7, 42, 123) | 92.33 ± 1.26% accuracy, 92.27 ± 1.11% F1 |
| PV test, 3-model ensemble | 94.00% accuracy (12 errors) |
| Error ranking (AUROC), ensemble entropy vs. mean single seed, PV test | 0.837 vs. 0.762 |
| PV test, reviewing the 20% most uncertain tablets | 97.50% accuracy on the rest |
| pillQC transfer, ensemble | 94.27% accuracy; 98.05% on the 80% kept after review |
| Latency, RTX 3060, batch 1, 224 × 224 | 14.5 ms (one model), 47.0 ms (ensemble) |

The AMSM module does not significantly improve accuracy over the plain ConvNeXt-Tiny backbone. The paired comparison is in `all_results.txt`.

## Installation

A CUDA GPU is needed to reproduce the latency numbers. Training also runs on CPU, but it is much slower. Results in the paper were produced on an NVIDIA RTX 3060.

```bash
pip install -r requirements.txt
```

Before running anything else, check the installation:

```bash
python smoke_test.py
```

Optional, for GFLOPs and the Holm cross-check in `test_metrics.py`:

```bash
pip install fvcore statsmodels
```

## Data

The code does not include the datasets. Download them from the sources below and place them in the `data/` folder, which is git-ignored.

**PV dataset (PharmaVision):** [Kaggle: tahirlee/pharmaceutical-tablets-dataset](https://www.kaggle.com/datasets/tahirlee/pharmaceutical-tablets-dataset)

With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) and an API token in place:

```bash
pip install kaggle
kaggle datasets download -d tahirlee/pharmaceutical-tablets-dataset --unzip -p data/PV_raw
```

Then arrange the images so the layout is:

```
data/PV/
  train/{normal,defective}/*.jpg     # 190 + 190
  val/{normal,defective}/*.jpg       #  30 +  30
  test/{normal,defective}/*.jpg      # 100 + 100
```

Check the folder names after download and rename or move them as needed. `data.py` accepts the aliases `training`, `valid`, `validation`, `testing`, `good` and `bad`, and accepts either `{split}/{class}` or `{class}/{split}` layouts. Folders with other names, such as `train_aug`, are ignored.

**pillQC:** [matlab-deep-learning/pillQC](https://github.com/matlab-deep-learning/pillQC)

```bash
git clone https://github.com/matlab-deep-learning/pillQC data/pillQC
```

The images are in `images/normal/`, `images/chip/` and `images/dirt/`. The study uses normal (149 images) and chip (43 images) only. Dirt is excluded on purpose, as explained in `data.py`, so pass the `images` folder as `--pillqc`. The loader raises an error if the counts differ from 149 normal and 43 chip.

Generate the offline augmentation of the training set once:

```bash
python data.py --pv-root data/PV --out-root data/PV_augmented
```

This expands the training set from 380 to 1,140 images. The augmentation uses seed 7 and is fixed across all runs.

## Reproducing the study

One command trains every configuration, evaluates it, and writes the summary:

```bash
python run_all.py --train-root data/PV_augmented --pillqc data/pillQC/images --candidate convnext_tiny
```

It runs these stages in order:

0. Self-checks: `test_metrics.py` and `verify_amsm_params.py`. The run stops if either fails.
1. Backbone family sweep at seed 7. Skip it with `--skip-family` when a backbone is pinned.
2. Backbone selection. `--candidate` pins the backbone. Without it, the backbone with the best **validation** F2 among the EDL seed-7 runs is chosen. Test metrics are never used for selection.
3. Training of six configurations at seeds 7, 42 and 123.
4. Testing on PV and pillQC, with paired comparisons against the proposed model.
5. Summary: `all_results.txt` and `seeds.csv`.

Finished steps are skipped, so after a failure, fix the cause and run the same command again. Add `--dry-run` to print the commands without running them.

Then run the uncertainty and latency analysis:

```bash
python save_val_predictions.py --train-root data/PV_augmented   # validation predictions, needed for deferral
python ensemble_uncertainty.py                                   # CPU; writes ensemble_uncertainty.txt
python ensemble_latency.py                                       # GPU; writes ensemble_latency.json
```

`ensemble_uncertainty.py` chooses the deferral rate (0%, 10% or 20%) on the validation set. It then applies that rate once to the PV test set and pillQC.

Individual steps can also be run on their own:

```bash
python train.py --train-root data/PV_augmented --backbone convnext_tiny --variant amsm_only --head ce --seed 7
python evaluate.py --ckpt runs/convnext_tiny_amsm_only_ce_seed7/best.pt --train-root data/PV_augmented --pillqc data/pillQC/images
python efficiency.py --ckpt runs/convnext_tiny_amsm_only_ce_seed7/best.pt
python smoke_test.py   # end-to-end check on synthetic data; run this first on a new machine
```

Run `python smoke_test.py` before anything else to confirm the installation works.

## Reproducibility notes

- The seed controls weight initialisation, batch order and dropout. The augmented training set is fixed.
- The reported spread across seeds reflects training randomness only.
- GPU kernels are not bit-for-bit reproducible, so re-running a seed can change results slightly.

## Repository layout

| File | Contents |
|---|---|
| `config.py` | Frozen PV protocol, model and loss settings, paths, run-directory naming |
| `data.py` | Split discovery and checks, offline augmentation, data loaders, pillQC loading |
| `backbones.py` | Encoder registry (ConvNeXt family and others), multi-stage feature extraction |
| `modules.py` | AMSM, stage projection, MLP head, Dirichlet outputs for the EDL variant |
| `model.py` | `TabletDefectNet` and model construction for each variant |
| `losses.py` | Cross-entropy (PV protocol), focal cross-entropy, evidential loss |
| `metrics.py` | Accuracy, F1, F2, Wilson and bootstrap intervals, McNemar test, Holm correction, AUROC, ECE |
| `train.py` | Training loop; writes `best.pt`, `history.json` and `test_predictions.npz` |
| `evaluate.py` | Test and pillQC evaluation, MC-Dropout baseline, paired comparisons |
| `save_val_predictions.py` | Saves validation predictions for the ensemble deferral rule |
| `ensemble_uncertainty.py` | Ensemble uncertainty, error AUROC, bootstrap, deferral analysis |
| `ensemble_latency.py` | Latency and throughput of the single model and the ensemble |
| `efficiency.py` | Parameters, GFLOPs, latency, throughput, peak memory |
| `run_all.py` | Full pipeline in one command (see above) |
| `seed_summary.py` | Mean ± SD across seeds for each configuration |
| `summarize.py` | Table of finished runs, ranked by validation F2 |
| `param_budget.py` | Parameter totals per backbone |
| `test_metrics.py` | Checks `metrics.py` against published reference values |
| `verify_amsm_params.py` | Checks AMSM parameter counts against the source paper |
| `smoke_test.py` | End-to-end run on synthetic data |
| `run_family.bat`, `run_seeds.bat` | Windows helpers for the backbone family sweep and the three-seed runs |

## Limitations

- Three seeds for the proposed model. The comparison with the baseline is paired but small.
- The deferral rate is chosen on 60 validation images, which contain only 4 ensemble errors. Treat the 20% choice as provisional.
- pillQC is a single product imaged under one fixed setup. Transfer results do not show generalisation to other products or cameras.
- pillQC dirt images are excluded from the transfer evaluation, so the transfer test covers normal versus chip only.
- Labels describe visual condition only. They do not certify chemical, dissolution or microbiological properties.

## Citation

```
[BibTeX entry to be added on publication]
```

## License

[Choose a license, e.g. MIT or Apache-2.0, and add a LICENSE file.]

## Contact

Tahir Mahmood, tahirmahmood@dongguk.edu
Division of Electronics and Electrical Engineering, Dongguk University, Seoul, South Korea
