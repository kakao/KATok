# KATok: Keep-or-Drop? Adaptive Tokenizer for Compact Video Representation

**ECCV 2026**

Official repository for **"Keep-or-Drop? Adaptive Tokenizer for Compact Video Representation"** (ECCV 2026).

[![arXiv](https://img.shields.io/badge/arXiv-2608.24293-b31b1b.svg)](https://arxiv.org/abs/2608.24293)
[![Project Page](https://img.shields.io/badge/Project_Page-KATok-1c7ed6)](https://kakao.github.io/KATok/)

Yeonkyeong Lee, Hyunsung Go, Jongmin Kim, Sewoong Lim, Donghoon Lee<sup>†</sup>

Kakao Corp. — <sup>†</sup>Corresponding author

Conventional video VAEs compress at a fixed ratio, so a static shot and a fast-moving
one cost exactly the same number of tokens. KATok is a transformer VAE whose **adaptive
token selector** is learned jointly with the latents: it learns a keep-or-drop probability 
for each latent token and discards uninformative ones.
The token count therefore becomes an outcome of the content rather than a preset budget
— no inference-time search, no per-sample tuning.

On Panda-70M at 256²×16, KATok reconstructs at **31.24 PSNR / 5.12 rFVD using 366 tokens
on average**.

This repository contains **inference code** for the tokenizer and for the two diffusion
variants built on top of it. Training code is not included.

---



## Install

```bash
pip install -r requirements.txt
```

PyTorch's built-in SDPA is the default attention backend, so nothing needs to be
compiled. If `flash-attn` is installed KATok picks it up automatically for the faster
kernels used in the paper's experiments; the two paths compute the same thing.
Set `ATTN_BACKEND=sdpa` to force the pure-PyTorch path.

## Weights

> Weight release is under internal review.


## Quickstart

Any video file works — the scripts decode it, center-crop and resize to the requested
resolution, and take the first `--frames` frames.

```bash
# reconstruct a clip and report what it cost
python scripts/reconstruct.py your_clip.mp4 -w weights/ -o out/

# render which patches survived
python scripts/visualize_tokens.py your_clip.mp4 -w weights/ -o out/
```

```
input  : (1, 3, 16, 256, 256)
tokens : 370 / 514   (72.0% kept, registers included)
comp.  : 132.8x   at 64 channels per token
psnr   : 30.05 dB
```

In `visualize_tokens.py` output, black tiles are dropped tokens. A near-static clip
shows the pattern the method is built around: 186 tokens for the first 8 frames, then
only 25 for the next 8, because the second half is almost entirely redundant.

### Python API

```python
import torch
from katok.tokenizer import KATok, token_counts, token_positions
from katok.utils import prepare, read_video

model = KATok.from_pretrained("weights/", device="cuda")

video = read_video("your_clip.mp4", n_frames=16)        # (C, T, H, W) in [0, 1]
x = prepare(video, resolution=256, n_frames=16).cuda()  # (1, C, T, H, W) in [-1, 1]

latents = model.encode(x)          # sparse latent tokens
recon = model.decode(latents)      # back to pixels, in [-1, 1]

latents.tokens.tensor  # (B, L, 64) gated latents; dropped rows are exactly zero
latents.mask           # (B, L, 1) keep mask; the first 2 rows are register tokens
latents.logit          # (B, L, 2) raw keep/drop logits, alpha_i

token_counts(latents)     # N_eff per sample, registers excluded
token_positions(latents)  # (t, h, w) grid coordinates of the kept tokens
```

`encode` is deterministic: it uses the posterior mean. Pass `sample=True` to draw from
the posterior instead.

### Long videos

How many frames fit in one pass depends on the resolution: multi-resolution training
paired large spatial sizes with short clips, so the envelope is 64 frames at 256² but
only 32 at 512² (the full table lives in `katok/tokenizer/chunked.py` as
`KNOWN_GOOD_SHAPES`). Past it the temporal position encodings were never trained, so
longer clips are reconstructed with a sliding window instead — automatically, sized to
fit the resolution's envelope:

```bash
python scripts/reconstruct.py long_clip.mp4 -w weights/tokenizer/ --frames 80
```

`visualize_tokens.py` takes the same flags. Both scripts accept `--chunk` to set the
window length (`0` forces a single pass) and `--overlap` to cross-fade neighbouring
windows, which defaults to none.

```python
from katok.tokenizer import reconstruct_video

recon, info = reconstruct_video(model, x, overlap=0)
info["chunked"]          # whether the clip had to be split
info["plan"]             # (start, end) of each window
info["token_counts"]     # tokens spent per window
info["tokens_per_frame"]
```

Windows are tokenized independently, so a shorter window sees less temporal redundancy
to remove and costs more tokens per frame, and any overlap is tokenized twice. Window
length must be a multiple of the encoder's temporal patch size (8).

## Generation

```bash
python scripts/generate.py -g weights/cascade/ -t weights/tokenizer/ -o out/ --classes 7 21 --tokens 370
```

The released generators are class-conditional on UCF-101 at 256²×16.

```python
import torch
from katok.diffusion import load_generator
from katok.tokenizer import KATok

gen = load_generator("weights/cascade/", device="cuda")
tokenizer = KATok.from_pretrained("weights/tokenizer/", device="cuda")

labels = torch.tensor([7, 21], device="cuda")
latents, counts = gen.sample(labels, tokens=370, cfg_scale=4.0, num_steps=50)
videos = gen.decode(latents, counts, tokenizer)   # (B, 3, 16, 256, 256) in [-1, 1]
```



### Choosing the token budget

A token budget must be chosen — there is no default. `tokens` is how many tokens to
generate per sample, registers included; it accepts either a single number or one
count per sample, and the two serve different purposes.

**A fixed count is a control signal.** Lowering it yields simpler, low-motion clips;
raising it yields more dynamic, visually richer ones — no retraining, no extra
conditioning:

```bash
python scripts/generate.py -g weights/cascade/ -t weights/tokenizer/ --tokens 200 --classes 7
python scripts/generate.py -g weights/cascade/ -t weights/tokenizer/ --tokens 400 --classes 7
```

**Per-sample counts match a dataset.** KATok spends a different number of tokens on
every clip, and the generator was trained on that joint distribution of class and
count. Drawing each sample's budget from real clips of its own class keeps generation
in that distribution; this is how the reported FVD numbers were produced:

```bash
python scripts/token_counts.py path/to/videos -w weights/tokenizer/ -o counts.json
python scripts/generate.py -g weights/cascade/ -t weights/tokenizer/ --token-counts counts.json
```

`token_counts.py` records which class each clip came from when the video folder is
organized by class (`root/<ClassName>/clip.avi`, UCF-style); `generate.py` then pairs
every sample with a count from its own class, falling back to the global pool (with a
warning) for classes the scan never reached.

```python
counts = torch.tensor([322, 307, 369], device="cuda")   # e.g. from real clips
latents, valid_counts = gen.sample(labels, tokens=counts)
```

Sequence length is padded to the largest count in the batch; padded slots are excluded
from attention and dropped when decoding, so they cost nothing but compute.

### The two variants

Sparse latents lose their spatial layout, so both variants pair content generation
with explicit position handling: **cascaded** (the paper's default) first generates an
occupancy mask with a small prior and conditions the content model on the selected
positions, while **joint** denoises positions as three extra channels alongside the
content. See the paper for the comparison between them.

## Results

*Ground truth / reconstruction / kept tokens on a 96-frame clip at 512², reconstructed
in 32-frame windows with no overlap — 4,057 of 12,294 possible tokens, 34.84 dB. Black
tiles are dropped tokens. Full-resolution video: [assets/cat_512_96.mp4](assets/cat_512_96.mp4).*

Reconstruction on the Panda-70M validation set. `Comp.` is
`H·W·T·3 / (#tokens · channels)`; for KATok `#Tokens` is the average over the set.


| Method            | Resolution | #Tokens     | Channels | Comp.↑     | PSNR↑     | LPIPS↓   | SSIM↑    | rFVD↓    |
| ----------------- | ---------- | ----------- | -------- | ---------- | --------- | -------- | -------- | -------- |
| OmniTokenizer-VAE | 256²×17    | 5120        | 8        | 96         | 28.10     | 0.05     | 0.88     | 7.84     |
| ElasticTok-KL     | 256²×16    | 3845.56     | 8        | 102.25     | 30.52     | 0.06     | 0.91     | 12.37    |
| **KATok**         | 256²×16    | **366.24**  | 64       | **134.21** | **31.24** | **0.04** | **0.94** | **5.12** |
| OmniTokenizer-VAE | 512²×33    | 36864       | 8        | 96         | 24.07     | 0.06     | 0.80     | 16.85    |
| **KATok**         | 512²×32    | **1554.24** | 64       | **253.00** | **33.23** | 0.05     | **0.95** | **6.40** |


The compression ratio *rises* with resolution (134× → 253×) because larger clips carry
more spatio-temporal redundancy to remove. Token count tracks temporal entropy
(Pearson r ≈ 0.87) far more strongly than spatial entropy (r ≈ 0.62).

## License
This software is licensed under the Apache 2 license, quoted below.

Copyright 2026 Kakao Corp. http://www.kakaocorp.com

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Citation

```bibtex
@inproceedings{lee2026katok,
  title     = {Keep-or-Drop? Adaptive Tokenizer for Compact Video Representation},
  author    = {Lee, Yeonkyeong and Go, Hyunsung and Kim, Jongmin and Lim, Sewoong and Lee, Donghoon},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```