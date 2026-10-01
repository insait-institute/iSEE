# iSEE: Object Permanence through Self-Supervision

[Pramish Paudel](https://pramishp.github.io/), [Ajad Chhatkuli](https://ajadchhatkuli.github.io/), Luc Van Gool,
Danda Pani Paudel · INSAIT, Sofia University "Kliment Ohridski" ·
[Project page](https://insait-institute.github.io/iSEE/)

<p align="center">
  <img src="assets/teaser.gif" width="480" alt="iSEE on a LA-CATER clip: the target is covered by nested cones and carried while hidden">
</p>
<p align="center"><em>iSEE on LA-CATER: the target (white) is covered by nested cones and carried, out of sight for
7.0 s. Green: iSEE's estimate of the target, dashed while it is hidden.</em></p>

## Abstract

Object permanence, keeping track of an object's identity and position while it is occluded, is central to video
representations that track, predict and plan. Trackers that achieve it learn from boxes, track identities and
visibility labels. On the other hand, self-supervised object-centric methods discover objects without labels: through
slot attention, it represents a video as slots that bind to objects and follow them across frames. However, these
slots are lost under occlusion, making the desired permanence impossible. Reasoning permanence is a hard problem
because it requires to detect when an object becomes occluded, re-identify when object reappears, and keep the
object's hidden position continuous, using reappearance as the only learning cue. To address this, we propose iSEE, a
novel framework that offers all three aforementioned requirements, without any labels whatsoever. We built iSEE using
the following three proposed components: (i) Object evidence modelling: a slot's attention, compared with its own
past, reveals when its object is hidden. (ii) Appearance-position separation: two slot streams let the appearance be
held for re-identification while the position keeps changing. (iii) Permanence from reappearance: a walker follows the
hidden object's position, trained only on where the object reappears. On LA-CATER static, iSEE returns a reappearing
object to its own slot after 86 % of occlusions, against 32 % for SlotContrast, and localises it while hidden within
4.1 mAP of the label-trained SoTA RAM. The two streams also allow downstream planning, with the position stream as the
action of a world model.

## Setup

```bash
conda create -n isee python=3.11 -y && conda activate isee
pip install -r requirements.txt        # tested with PyTorch 2.14 (CUDA 13.0) and timm 1.0.30
```

**Backbone: DINOv2 NoPE.** iSEE reads frozen features of DINOv2 ViT-S/14 with four registers (timm
`vit_small_patch14_reg4_dinov2`) whose positional embedding has been removed and which was fine-tuned on COCO to
reproduce the original DINOv2 patch features ([DINO-SAW](https://github.com/tldr-group/dino-saw), Pawlowsky et al.,
2026). We use the authors' checkpoint `nope_coco_dv2_vits14_reg4.pth` unchanged: `isee/model/backbone.py` downloads
it from [huggingface.co/rmdocherty/dino-saw](https://huggingface.co/rmdocherty/dino-saw) into `checkpoints/` on first
use and checks its sha256. iSEE reads the 384-d patch tokens of its last layer ([CLS] and registers dropped) at
518 × 518 px on LA-CATER (37 × 37 patches) and 336 × 336 px on MOVi-C (24 × 24 patches). The backbone is never
trained.

**Checkpoints** (`checkpoints/`): the encoders `isee_lacater_static`, `isee_lacater_moving` and `isee_movi_c`, and
the walkers `walker_lacater_{static,moving}_seed{0,1,2}` (three per camera, run as an ensemble). Each config in
`configs/` names its checkpoints and holds every setting of the model, its training and its evaluation.

**Data.** The code downloads no dataset.
- LA-CATER, static camera: `test_data.zip`, `train_data_part_1.zip` and `train_data_part_2.zip` from the
  [LA-CATER page](https://avivsham.github.io/op_net/).
- LA-CATER, moving camera:
  [`la_cater_moving.tar.gz`](https://tri-ml-public.s3.amazonaws.com/datasets/la_cater_moving.tar.gz) from
  [RAM](https://github.com/TRI-ML/RAM).
- LA-CATER masks, for evaluation: `lacater_masks.zip` from the
  [v1.0 release](https://github.com/insait-institute/iSEE/releases/tag/v1.0), the visible mask of every object and
  its pixels rendered alone, which we rendered from the dataset's scene files.
- MOVi-C: `movi_c/256x256:1.0.0` from TensorFlow Datasets (`pip install tensorflow tensorflow-datasets`).

```bash
python scripts/prepare_lacater.py static --downloads <folder with the zips> --out data/la_cater
python scripts/prepare_lacater.py moving --tarball la_cater_moving.tar.gz --out data/la_cater_moving
unzip lacater_masks.zip -d data/
python scripts/prepare_movi_c.py --out data/movi_c      # --splits validation: evaluation only (1.5 GB of 48 GB)
```

## Evaluation

```bash
python scripts/eval_lacater.py configs/lacater_static.yaml     # -> outputs/lacater_static/results.json
python scripts/eval_lacater.py configs/lacater_moving.yaml     # -> outputs/lacater_moving/results.json
python scripts/eval_movi_c.py configs/movi_c.yaml              # -> outputs/movi_c/results.json
```

Results with the released checkpoints (percent, higher is better):

| | LA-CATER static | LA-CATER moving | MOVi-C |
|---|---|---|---|
| own-box mAP@[0.1:0.3] while hidden: occluded / contained / carried | 80.6 / 84.2 / 78.7 | 54.3 / 51.6 / 46.3 | |
| SAME, all occlusions | 86.3 | 75.8 | |
| video FG-ARI / video mBO | 96.9 / 24.0 | 89.3 / 17.6 | 71.8 / 36.1 |

mAP places the box of the slot that followed the target at that slot's position on every hidden frame (195 static /
388 moving occlusions, `eval_sets/`). SAME is the share of occlusions after which the target returns to the slot that
owned it (183 / 360 scored). FG-ARI and mBO score the decoder's masks over whole clips (the first 150 LA-CATER test
clips, the 250 MOVi-C validation clips). The definitions are in `isee/eval/`, and `results.json` also holds the
per-label counts, SAME by occlusion length and the *frozen* baseline. Computed on an NVIDIA RTX A6000 with each
config's `evaluation` settings; TEN's hold decisions are discrete, so other hardware or precision moves the numbers
slightly.

**On your own video:** `python scripts/infer.py configs/lacater_static.yaml --video clip.mp4 --out outputs/infer --tf32`
writes TEN's held flags, the slots, the masks and every slot's position (`.npz`) and a video of them (`.mp4`).

## Training

```bash
python scripts/train_encoder.py configs/lacater_static.yaml --out runs/lacater_static     # rerun to resume
python scripts/build_walker_data.py configs/lacater_static.yaml                           # walker: LA-CATER only
python scripts/train_walker.py configs/lacater_static.yaml --seed 0 --out runs/walker_lacater_static_seed0
```

Every setting is in the config's `training` and `walker_training` blocks (train the walker for seeds 0, 1 and 2). The
released models were trained on NVIDIA H200 GPUs; the LA-CATER encoder's batch of 32 and the walker's batch of 16 do
not fit in 48 GB. To evaluate retrained models, point the config's `checkpoint` and `walker.checkpoints` at them.

## Repository structure

```
configs/       one config per dataset            isee/model/   encoder, TEN, walker
checkpoints/   trained encoders and walkers      isee/train/   losses
eval_sets/     the LA-CATER occlusions scored    isee/eval/    metrics
assets/        the teaser                        isee/data/    reading and preparing clips
scripts/       prepare_*, train_*, build_walker_data, eval_*, infer
```

## Citation

```bibtex
@article{paudel2026isee,
  title   = {iSEE: Object Permanence through Self-Supervision},
  author  = {Paudel, Pramish and Chhatkuli, Ajad and Van Gool, Luc and Paudel, Danda Pani},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgements

iSEE builds on [SlotContrast](https://github.com/martius-lab/slotcontrast) (Manasyan et al., CVPR 2025): the learned
slot initialisation, the feature MLP, the slot-attention update, the predictor's transformer block and the slot-slot
contrastive loss follow its implementation. We thank its authors for releasing their code.

## License

MIT (`LICENSE`); third-party components and their licenses are listed in `THIRD_PARTY_NOTICES.md`.
