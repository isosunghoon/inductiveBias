# Experimental

Work that was explored during the study but does not appear in the final report.

It is kept because it documents what was tried and where the work was heading, but it is
deliberately **not wired into `train.py` / `run_analysis.py`** — the main tree contains
exactly the code paths behind the reported numbers. Nothing here is needed to reproduce
the report.

---

## Scaling to a larger dataset

The report's limitation section notes that its conclusions were drawn on small datasets
and should be re-checked at ImageNet scale. This is how far that got:

| File | What it does |
|---|---|
| `select_imagenet100_classes.py` | Picks a 100-class subset of ImageNet-1k by per-class F1 of a pretrained ResNet-50 (from `timm`), and writes a manifest JSON |
| `config/base_imagenet100.yaml` | Base config for that subset (300 epochs, ImageNet normalization) |

The loader side is already merged into the main tree: `utils/dataset.py` supports
`dataset: imagenet-100`, including download and manifest handling. To run it:

```bash
python experimental/select_imagenet100_classes.py --data_path ./data
python train.py --base_config experimental/config/base_imagenet100.yaml \
                --config config/vit.yaml --run_name vit --device_type 0
```

No ImageNet-100 run made it into the report.

## Alternative token mixers

| File | What it does |
|---|---|
| `models/token_mixers_extra.py` | `PoolFormer` — the average-pool-minus-identity mixer from the MetaFormer paper |

The four reported mixers were chosen because they cleanly factor the two properties of
Table 1 (long range dependency × data specific weight); PoolFormer does not sit on that
grid. To use it, import the class and add a branch to `setup()` in `train.py` alongside
the other mixers.

## Alternative baselines and optimizers

| File | What it does |
|---|---|
| `models/resnet.py` | ResNet-18/50 baseline built outside the MetaFormer framework |
| `config/resnet.yaml` | SGD config for that baseline |
| `models/sam.py` | Sharpness-Aware Minimization optimizer |

A CNN built outside the MetaFormer framework reintroduces the confounds the controlled
setup exists to remove (downsampling, stage widths, stem design), which is what
Convformer is for inside the framework. Note this ResNet is unrelated to the ResNet-50
used for ImageNet-100 class selection above — that one comes from `timm`.

SAM is relevant to the loss-landscape analysis of Section 5.3, since it optimizes for
flatness explicitly. It is costly to run here: it needs two forward/backward passes per
step, and the implementation forces fp32 because the double pass does not compose
cleanly with AMP's `unscale_`.

Both need re-wiring in `train.py` (a `--model resnet18` branch and an `--optimizer sam`
branch) before they will run; their argparse options are still registered in
`utils/config.py`, marked `[experimental]`.

## Analyses not in the report

| File | What it does |
|---|---|
| `analysis/dis_occ_sensitivity.py` | Occlusion sensitivity vs. patch distance — a perturbation-based way to measure how far a model actually looks |
| `analysis/erf_cka_boundary.py` | Searches for a layer index where ERF and CKA both change regime, i.e. a "local → global" boundary |
| `analysis/loss_landscape_2d.py` | 2D filter-normalized loss surface (Li et al. 2018) |
| `analysis/visualize_loss_landscape_2d.py` | Plotting for the above |
| `results/dis_occ_mlpmixer*.png` | Occlusion-sensitivity output for the MLP mixer (saved under its old `denseformer` name) |

Two of these overlap with analyses that the report does use. The occlusion study
measures the same quantity ERD does in Section 5.1 — the spatial range a trained model
relies on — but from perturbations rather than from gradients. The 2D loss surface
targets the same object as the Hessian spectrum in Section 5.3, but as one slice through
a random plane rather than as the extreme curvature at the convergence point; the
reported Fig 8 comes from `analysis/loss_hessian.py` and `analysis/hessian_spectrum.py`
in the main tree.

These scripts were written against an earlier state of the pipeline and are **not kept
working** against the current `analysis/pipeline.py` interface.
