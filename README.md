# simsim · Image Similarity Scanner

**Live site: <https://nathanreitinger.github.io/simsim/>**

A VirusTotal-style scanner for image similarity. Drop two images and **36 tests** run side by side — from
byte-level file hashes to Meta's **SSCD** copy detector — and the page reports how many of them flag the pair
as similar, with an explanation of every test.

Everything runs in the browser (WebAssembly). Images are never uploaded, so the site can be hosted as static
files on GitHub Pages.

Built for teaching computer science & law (copyright, memorisation in generative models, content matching).

## What it runs

| Group | Tests |
| --- | --- |
| **Neural copy detection** | SSCD ResNet-50 (`sscd_disc_mixup`, official preprocessing) · SSCD ResNeXt-101 in the Somepalli et al. replication setting (`sscd_disc_large`, resize 256 / centre-crop 224) · DINOv2 ViT-S/14 · CLIP ViT-B/32 · LPIPS (AlexNet) |
| **Objects** | D-FINE object detector (365 Objects365 categories): objects are found in both images, paired up and compared one by one, with marked-up images in the Objects tab |
| **Exact & verbatim** | SHA-256 · SHA-1 · MD5 · pixel-exact match of decoded pixels · verbatim crop search (template matching, verified pixel-for-pixel) |
| **Perceptual hashes** | PDQ (Meta) · PDQ over all 8 rotations/mirrors · pHash · dHash · aHash · wHash · Blockhash |
| **Keypoints & geometry** | ORB, AKAZE and BRISK keypoints with RANSAC homography · SSIM after aligning B onto A |
| **Pixel & structural** | SSIM · MS-SSIM · PSNR · normalised cross-correlation · UQI · GMSD · CIEDE2000 ΔE · changed-pixel ratio |
| **Colour & histograms** | Hue–saturation correlation · χ² · RGB histogram intersection · Bhattacharyya · brightness EMD |
| **Metadata** | EXIF / XMP capture fields (camera, timestamp, unique IDs) |

Each test has a verdict (*Identical*, *Match*, *Partial*, *No match*, *N/A*). Thresholds come from the source
papers where they exist — SSCD ≥ 0.75 (90% precision on DISC2021, per the SSCD authors), SSCD > 0.5 for
replication (Somepalli et al., CVPR 2023 / NeurIPS 2023), PDQ ≤ 31 bits (Meta) — and are labelled as heuristics
elsewhere. All thresholds live in [`js/engines.js`](js/engines.js).

Other features: a spotlight panel with SSCD on its published threshold scale (CLIP alongside for contrast),
visual comparisons (swipe, blink, colour-difference and SSIM heat maps, keypoint-match lines, aligned overlay),
a details table (formats, digests, perceptual hashes, EXIF), a JSON report, built-in public-domain examples, and
a **Transform lab** that makes an edited copy of image A (crop, rotate, mirror, recolour, blur, noise, caption,
JPEG quality) to explore which tests survive which edits.

## Run locally

Any static file server works (module workers need `http://`, not `file://`):

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>. Add `?debug` to the URL to expose the scan state as `window.scanner` in the
console.

## Publishing changes

The workflow in [`.github/workflows/pages.yml`](.github/workflows/pages.yml) publishes the site to GitHub Pages
(Settings → Pages → Source: *GitHub Actions*). Every push to `main` republishes it, usually within a minute or
two:

```bash
git add -A
git commit -m "Describe the change"
git push
```

You can also edit files directly on github.com; committing there republishes the site the same way. The
**Actions** tab shows each deployment ("Deploy to GitHub Pages") and has a **Run workflow** button to redeploy
without a new commit. When GitHub's runners are busy, a deployment waits in the queue rather than failing.

Two caches to know about:

- GitHub Pages lets browsers cache files for about 10 minutes, so hard-refresh (⇧⌘R / Ctrl+Shift+R) to see a
  change immediately.
- Model files are stored in the browser's Cache Storage under the name in
  [`js/lib/modelstore.js`](js/lib/modelstore.js) (`image-compare-models-v1`). If you replace a file in `models/`,
  bump that name (e.g. to `-v2`) so returning visitors download the new version.

To host a copy elsewhere: fork or push this folder to another repository and enable Pages the same way; the site
then appears at `https://<user>.github.io/<repo>/`. All paths are relative, so no configuration is needed.

The repository is about 220 MB because the neural network weights are committed directly (GitHub Pages cannot
serve Git LFS files). Every file is under GitHub's 50 MB warning size; the CLIP encoder is split into two parts
that the page joins after download. `.nojekyll` disables Jekyll processing.

Visitors download the models on their first scan (about 200 MB if every model is enabled) and the browser keeps
them in Cache Storage afterwards. The large ones (CLIP 92 MB, SSCD large 47 MB) can be switched off in
**Settings**.

## How the models were prepared

The `tools/` folder contains the conversion scripts:

- [`tools/export_sscd.py`](tools/export_sscd.py) — exports Meta's official TorchScript SSCD models
  (`sscd_disc_mixup`, `sscd_disc_large`, MIT licence) to ONNX.
- [`tools/export_lpips.py`](tools/export_lpips.py) — exports LPIPS v0.1 (AlexNet) from the `lpips` package.
- [`tools/export_hf_vision.py`](tools/export_hf_vision.py) — takes the full-precision ONNX exports of CLIP ViT-B/32
  (Xenova), DINOv2-small and the D-FINE-M Objects365 detector (onnx-community) from Hugging Face at pinned
  revisions.
- [`tools/quantize_weights.py`](tools/quantize_weights.py) — stores each large weight as int8 with a per-channel
  scale and rebuilds it in fp32 inside the graph, so inference still runs in fp32. The few layers that are
  sensitive to rounding (the stem / first blocks / patch embedding, found by a per-layer sensitivity sweep) stay
  fp32. Files are ~4× smaller.

Accuracy checks against the original full-precision models on about 50 image pairs (crops, rotations,
mirroring, compression, colour edits, captions, unrelated images): the largest change in any similarity score was
0.004 (SSCD), 0.006 (SSCD large), 0.005 (CLIP), 0.007 (DINOv2) and 0.003 (LPIPS). The int8 files published on
Hugging Face moved CLIP/DINOv2 scores by up to 0.06, which is why they are not used.

## Validation

The JavaScript implementations were checked against the standard Python implementations:

- image resizing is a port of Pillow's resampler and is byte-identical to Pillow (bilinear, bicubic, Lanczos, box);
- aHash, dHash, pHash and wHash are bit-identical to the `imagehash` library; Blockhash to `blockhash-js`;
- PDQ (hash, quality score and all dihedral variants) is bit-identical to Meta's reference implementation;
- SSIM matches `skimage.metrics.structural_similarity` and MS-SSIM matches `pytorch_msssim` (within 1e-5);
  CIEDE2000 matches `skimage` and the Sharma et al. test data;
- the neural preprocessing reproduces torchvision / Hugging Face pipelines, and similarities computed in
  JavaScript match Python ONNX Runtime within 1e-6.

## Caveats

- Similarity scores measure resemblance of pixels or learned features. They do not answer legal questions about
  originality, protectable expression, substantial similarity or fair use.
- Thresholds outside the cited papers are heuristics calibrated on a small set of examples.
- Images larger than 4096 px are downscaled before hashing and embedding; pixel-exact and crop checks use the
  full-resolution pixels. Transparent pixels are composited on white. HEIC only decodes in browsers that support
  it (Safari).
- Browsers slow down background tabs, so keep the tab visible while a scan runs.

## Credits and licences

- SSCD: Pizzi et al., *A Self-Supervised Descriptor for Image Copy Detection*, CVPR 2022 — MIT.
- D-FINE: Peng et al., ICLR 2025 — Apache-2.0; trained on Objects365 (Shao et al., ICCV 2019).
- DINOv2: Oquab et al., 2023 — Apache-2.0. CLIP: Radford et al., 2021 — MIT. LPIPS: Zhang et al., CVPR 2018 —
  BSD-2-Clause (AlexNet weights from torchvision, BSD-3-Clause).
  Licence texts are in [`models/licenses/`](models/licenses).
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) 1.30 (MIT),
  [OpenCV.js](https://github.com/TechStark/opencv-js) 4.12 (Apache-2.0),
  [exifr](https://github.com/MikeKovarik/exifr) 7.1 (MIT) — in `vendor/` with their licences.
- Perceptual hash algorithms follow [imagehash](https://github.com/JohannesBuchner/imagehash),
  [blockhash](https://github.com/commonsmachinery/blockhash-js) and
  [PDQ](https://github.com/facebook/ThreatExchange/tree/main/pdq).
- Example photos (via scikit-image): Eileen Collins (NASA, public domain), Falcon 9 launch (SpaceX, public
  domain), Chelsea the cat (Stefan van der Walt, CC0), coffee (Rachel Michetti, CC0).
