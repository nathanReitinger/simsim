# simsim · Image Similarity Scanner

**Live site: <https://nathanreitinger.github.io/simsim/>**

A VirusTotal-style scanner for image similarity. Drop two images and **43 tests** run side by side — from
byte-level file hashes to Meta's **SSCD** copy detector — and the page reports how many of them flag the pair
as similar, with an explanation of every test.

Everything runs in the browser (WebAssembly). Images are never uploaded, so the site can be hosted as static
files on GitHub Pages.

Written by [Nathan Reitinger](https://www.law.northwestern.edu/faculty/profiles/nathanreitinger/) for teaching
computer science & law (copyright, memorisation in generative models, content matching). Corrections and pull
requests welcome.

## What it runs

| Group | Tests |
| --- | --- |
| **Neural copy detection** | SSCD ResNet-50 (`sscd_disc_mixup`, official preprocessing) · SSCD after alignment (crop, rotation, perspective or mirroring undone with the keypoint transform, then SSCD on the shared region) · SSCD ResNeXt-101 in the Somepalli et al. replication setting (`sscd_disc_large`, resize 256 / centre-crop 224) · DINOv2 ViT-S/14 · CLIP ViT-B/32 · LPIPS (AlexNet) |
| **Objects** | D-FINE object detector (365 Objects365 categories): objects are found in both images, paired up and compared one by one, with marked-up images in the Objects tab |
| **Exact & verbatim** | SHA-256 · SHA-1 · MD5 · same image data with metadata set aside (hash of the JPEG scans / PNG image chunks / WebP bitstream) · pixel-exact match of decoded pixels · verbatim crop search (template matching, verified pixel-for-pixel) |
| **Perceptual hashes** | PDQ (Meta) · PDQ over all 8 rotations/mirrors · pHash · dHash · aHash · wHash · Blockhash |
| **Keypoints & geometry** | ORB, AKAZE and BRISK keypoints with RANSAC homography · SSIM after aligning B onto A |
| **Pixel & structural** | SSIM · MS-SSIM · PSNR · normalised cross-correlation · UQI · GMSD · CIEDE2000 ΔE · changed-pixel ratio · Carlini et al.’s tiled ℓ2 extraction test (512², worst of 16 tiles, ≤ 0.15) |
| **Colour & histograms** | Hue–saturation correlation · χ² · RGB histogram intersection · Bhattacharyya · brightness EMD |
| **Metadata & provenance** | EXIF capture fields (camera, timestamp, unique IDs) · copyright management information (XMP/IPTC/EXIF creator, rights, credit, licence — flagged when B drops A’s) · XMP edit history (DocumentID, DerivedFrom, DocumentAncestors; AI-generator settings in PNG text) · JPEG encoder fingerprint (quantization tables, estimated quality, subsampling) · embedded EXIF preview vs both images |

Each test has a verdict (*Identical*, *Match*, *Partial*, *No match*, *N/A*). Thresholds come from the source
papers where they exist — SSCD ≥ 0.75 (90% precision on DISC2021, per the SSCD authors), SSCD > 0.5 for
replication (Somepalli et al., CVPR 2023 / NeurIPS 2023), PDQ ≤ 31 bits (Meta) — and are labelled as heuristics
elsewhere. All thresholds live in [`js/engines.js`](js/engines.js).

## Seeing where images are similar

The **Where it’s similar** tab, and the card at the top of the results, show *where* the resemblance is,
grouped by the question each view answers:

- **Marked up** — the images annotated like a marked-up handout, with marker circles, numbered badges, curved
  arrows and labels, plus a numbered list of close-ups. Pointing at (or tapping) a number singles it out.
  - *Differences* — spot-the-difference for two versions of one picture. B is laid over A with the keypoint
    homography, refined by ECC and then by a smooth field of local shifts (tile-wise normalised cross-correlation,
    median-filtered), so scans and redraws that no single transform explains still line up. A pixel then counts
    as changed only when nothing within a few pixels of it in the other image has a similar colour, which forgives
    leftover misalignment of lines while added, removed and recoloured things stand out. Each change is labelled
    *added in B*, *missing in B*, *recoloured* or *changed*, and named by the object detector when it can.
  - *Matching regions* — what B took from A, even when it was restaged or redrawn. DINOv2 mutual patch matches are
    grouped by the objects D-FINE detects in A (and where their matches land in B), with spatially coherent groups
    for everything else. Each pair is circled in its own colour and joined by an arrow from A to B.
  - *Copy evidence (SSCD)* — SSCD’s score split exactly into **pairs** of locations, one in A and one in B. The
    model pools with GeM, which can be written as a sum over locations, and its last layer is linear, so each
    descriptor is a sum of per-location terms and the cosine score is a double sum over location pairs — a
    second-order explanation in the spirit of BiLRP (Eberle et al., TPAMI 2020; cf. Stylianou et al., WACV 2019).
    Each part of A points at the part of B it pairs with most strongly; parts that move together form one link,
    drawn as matching colour patches joined by an arrow sized by its share of the score. A bar adds the links up
    to the score, and a sentence reads the arrow pattern (in place, mirrored, cropped). On test pairs the
    strongest pairing lands on the true corresponding location for nearly every high-evidence cell (JPEG,
    caption, mirror, rotation) and most cells of a crop. The export that adds the per-location terms is
    `tools/export_sscd.py --explain`.
  - *Point and compare* — hover over (or tap) any spot in either image: the other image lights up wherever
    something resembles it and an arrow lands on the closest match, using DINOv2 patch features or SSCD’s
    pairwise evidence.
  - When B shows only part of A (a crop) or sits inside a larger canvas, the Differences view outlines that part
    and draws arrows from its corners to the other image’s corners.
- **Same content?**
  - *Matching parts* — dense patch correspondences from DINOv2 features (mutual nearest neighbours, as in
    Amir et al., “Deep ViT Features as Dense Visual Descriptors”, 2021). Matched patches are marked with dots of
    the same colour in both images.
  - *Similarity heat map* — for every patch, how close its best match in the other image is.
- **Same details?** — keypoint matches and an aligned overlay.
- **Same pixels?** — colour-difference and SSIM maps, swipe, blink, side by side.

Every view comes with a “what you’re seeing / what it means for copying” note. The results summary also places
each pair on a **similarity spectrum** — same file → same pixels → re-saved → edited copy → shared part →
similar subject → unrelated — and the How it works section explains the seven levels and which tests detect them.

## Other features

A spotlight panel with SSCD on its published threshold scale (CLIP alongside for contrast),
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

### Benchmark of the copy detectors

[`tools/bench`](tools/bench) builds a small copy-detection benchmark from the scikit-image sample images: 19 base
images, each edited 19 ways (JPEG q15, 40% downscale, centre and corner crops, mirror, 15° and 90° rotation, hue
shift, greyscale, blur, noise, meme captions, an emoji-style overlay, a screenshot frame, perspective, pixelation,
contrast, a combined crop+mirror+recolour+caption edit, and a collage that pastes the image into another), plus
2,033 pairs of unrelated images. It scores the site’s own ONNX models:

| Test | AUC | Copies found at 1% false alarms | Copies at the 0.75 copy threshold |
| --- | --- | --- | --- |
| SSCD | 0.997 | 99.4% | weak on collage 5%, rotation 15° 5%, pixelation 11%, corner crop 16% |
| SSCD after alignment (max with SSCD) | — | 99.7% | collage 79%, rotation 15° 84%, corner crop 84%, perspective 84%, memes 84% — and no new false alarms |
| SSCD Somepalli setting | 0.998 | 98.3% | |
| DINOv2 | 0.998 | 97.2% | |
| CLIP | 0.983 | 83.7% | |
| pHash | 0.827 | 52.6% | |

Two other partial-copy strategies were tried and not shipped: the maximum over 18 crops of each image found every
collage but cost 36 extra network runs per pair and raised the false-alarm level; re-scoring the region that
carried SSCD’s evidence did not help.

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
