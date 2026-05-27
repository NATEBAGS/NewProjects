# Hierarchical Image Classifier — 100-class Fine-Grained Recognition

A PyTorch image classification pipeline for a 100-class problem where the
classes form four contiguous superclasses (**food / flowers / cars / planes**,
25 fine classes each). Built for the CSE 144 final project at UC Santa Cruz.

Instead of training one big 100-way classifier, the system decomposes the
problem hierarchically:

1. A **router** picks the superclass (4-way).
2. A **specialist** trained only on that superclass picks the fine class (25-way).
3. A **flat 100-way model** acts as a fallback for superclasses where no
   specialist was trained (its logits are masked to the predicted superclass).

A `HybridPredictor` stitches the three together and adds multi-seed ensembling
and test-time augmentation on top.

---

## Architecture

```
                              ┌──────────────────────────┐
                              │   Input image (224×224)  │
                              └────────────┬─────────────┘
                                           │
                                           ▼
                              ┌──────────────────────────┐
                              │   Router (ConvNeXt-Tiny) │
                              │     4-way superclass     │
                              └────────────┬─────────────┘
                                           │
              ┌──────────────┬─────────────┴─────────────┬──────────────┐
              ▼              ▼                           ▼              ▼
       ┌────────────┐ ┌────────────┐             ┌─────────────┐ ┌─────────────┐
       │ Food spec. │ │ Flowers    │             │ Cars        │ │ Planes      │
       │ ensemble   │ │ specialist │             │ (no spec.)  │ │ (no spec.)  │
       │ (3 seeds)  │ │ ensemble   │             │             │ │             │
       └─────┬──────┘ └─────┬──────┘             └──────┬──────┘ └──────┬──────┘
             │              │                           │               │
             │   25-way local logits (avg. + TTA)       │  Flat 100-way │
             │              │                           │  logits, masked
             │              │                           │  to superclass id range
             ▼              ▼                           ▼               ▼
             └──────────────┴───────────────────────────┴───────────────┘
                                           │
                                           ▼
                                    Final 0–99 label
```

---

## Why this design?

- **Routing first** narrows the problem from 100 classes to 25, letting each
  specialist learn discriminative features only relevant to its superclass
  (e.g. a flower specialist can lean on color/texture; a car specialist
  on shape/silhouette).
- **Per-superclass augmentation recipes** — flowers get aggressive color
  jitter and rotation, cars get plain center crops (random crops destroy
  make/model cues), food gets a mild resized crop. See
  `hierarchy_data.get_specialist_transforms`.
- **Per-superclass batch augmentation** — MixUp, CutMix, or a random mix,
  chosen per superclass. See `SPECIALIST_CFG` in [`main_specialist.py`](main_specialist.py).
- **Two-phase fine-tuning** — Phase 1 trains only the new classifier head
  (everything else frozen). Phase 2 unfreezes the final ConvNeXt stage(s)
  with **layer-wise learning rates** (highest at the head, lowest deepest
  in the backbone) and a **cosine LR schedule**.
- **Multi-seed ensembling + TTA** at inference — each specialist is averaged
  over 3 seeds, and each forward pass is averaged over 3 deterministic
  TTA transforms (default eval, larger-resize center crop, and a
  horizontal-flip variant).
- **Stratified split** — `_stratified_split_indices` splits over the
  *original* 100 classes (not the 4 router labels), so every fine class is
  represented in both train and val even when training the 4-way router.

---

## Repo layout

```
.
├── engine.py              Training/eval loops, MixUp/CutMix, checkpointing
├── utils.py               set_seed() for reproducibility
├── data_setup.py          Flat 100-class data loaders (baseline)
├── hierarchy.py           Label-space conversions (original ↔ router ↔ local)
├── hierarchy_data.py      Router & specialist datasets / dataloaders / aug
├── model.py               ResNet-18 flat baseline (early experiment)
├── main_router.py         Train the ConvNeXt-Tiny router (4-way)
├── main_router_resnet.py  ResNet-50 router baseline for comparison
├── main_specialist.py     Train one 25-way specialist (set SUPERCLASS_NAME)
├── inference.py           Standalone TTA helper for the flat baseline
├── hybrid_inference.py    HybridPredictor: router + specialists + flat fallback
├── sample_submission.csv  Kaggle-style format reference
└── convnext_5fold_ensemble_submission.csv   Best submission (kept for reference)
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

GPU strongly recommended (CUDA-enabled PyTorch). CPU works but training is slow.

### Data

The dataset isn't checked in. Place it at:

```
data/
├── train/
│   ├── 0/    # folder name = original class id (0–99)
│   ├── 1/
│   └── ...
│       └── 99/
└── test/
    ├── 0.jpg
    ├── 1.jpg
    └── ...
```

Class folders **must** be named with their original integer id, since the
hierarchy code uses the folder name to determine which superclass each
sample belongs to (see [`hierarchy.py`](hierarchy.py)).

---

## How to train

Each script saves a phase-1 checkpoint and a best-val phase-2 checkpoint under
`./checkpoints/`, plus a `<phase>_curves.png` plot.

```bash
# 1. Router (4-way superclass classifier)
python main_router.py

# 2. One specialist per superclass — edit SUPERCLASS_NAME inside the file,
#    then run it once per superclass you want a specialist for:
python main_specialist.py

# 3. (Optional) Multi-seed ensembles — change SEED at the top of
#    main_specialist.py and re-run to produce additional checkpoints.
#    The default hybrid_inference.CONFIG expects seeds 42 / 123 / 999
#    for food and flowers.
```

For multi-seed ensembles, the easiest pattern is to change `SEED` and `RUN_NAME`
inside `main_specialist.py` before each run, e.g. saving as
`phase1_specialist_food_convnext_seed42.pt` etc.

---

## How to predict

Edit the checkpoint paths in `hybrid_inference.CONFIG`, then:

```bash
python hybrid_inference.py
```

This writes `submission.csv` with two columns (`id`, `label`) in numeric
filename order, ready for Kaggle-style submission.

---

## Tech stack

- **Framework:** PyTorch, torchvision
- **Backbones:** ConvNeXt-Tiny (primary), ResNet-50 / ResNet-18 (baselines), all ImageNet-pretrained
- **Optimization:** AdamW, cosine annealing LR, layer-wise learning rates, weight decay
- **Regularization:** MixUp, CutMix, label smoothing, dropout, RandomErasing
- **Augmentation:** RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation (per-superclass recipes)
- **Inference:** multi-seed model ensembling, deterministic test-time augmentation (TTA), logit averaging
- **Other:** stratified train/val splitting, two-phase transfer learning, gradient-frozen backbone warm-up

---

## Notes

- This was a class project; checkpoints aren't committed (they're large).
  Run the training scripts above to regenerate them.
- `model.py` and `main_router_resnet.py` represent earlier experiments kept
  for comparison — the production pipeline is the ConvNeXt-Tiny router plus
  per-superclass ConvNeXt-Tiny specialists.
