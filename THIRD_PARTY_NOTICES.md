# Third-party notices

## SlotContrast

iSEE builds on SlotContrast (Manasyan et al., CVPR 2025, https://github.com/martius-lab/slotcontrast). The
learned slot initialisation, the feature MLP, the slot-attention update and the predictor's transformer block follow
its implementation (re-implemented here; its attention's query scaling is kept, see `isee/model/predictor.py`).
SlotContrast is released under the MIT License:

```
MIT License

Copyright (c) 2023 Maximilian Seitzer and Andrii Zadaianchuk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## DINO-SAW (DINOv2 NoPE backbone)

The frozen backbone is the DINOv2 NoPE checkpoint `nope_coco_dv2_vits14_reg4.pth` released by the DINO-SAW authors
(https://github.com/tldr-group/dino-saw, https://huggingface.co/rmdocherty/dino-saw), MIT License,
Copyright (c) 2025 tldr group. It is not redistributed here: `isee/model/backbone.py` downloads it and checks its sha256.

## DINOv2 and timm

The backbone architecture is DINOv2 ViT-S/14 with registers (https://github.com/facebookresearch/dinov2, Apache 2.0),
instantiated through timm (https://github.com/huggingface/pytorch-image-models, Apache 2.0).
