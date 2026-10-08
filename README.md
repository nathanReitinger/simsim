# simsim · Image Similarity Scanner

**Live site: <https://nathanreitinger.github.io/simsim/>**

**The point of this site: computer similarity and legal similarity are two very different things.** The tests
measure how alike two images are — their bytes, pixels and learned features. Copyright law asks different
questions: was protected expression copied, was it a substantial part, and was the use fair? A pair the scanner
flags as a possible copy can be lawful, and a pair it calls merely similar can infringe; the
[copyright cases](#copyright-cases) show both.

A VirusTotal-style scanner for image similarity. Drop two images and **101 tests** run side by side — from
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
| **Neural copy detection** | SSCD ResNet-50 (`sscd_disc_mixup`, official preprocessing) · SSCD after alignment (crop, rotation, perspective or mirroring undone with the keypoint transform, then SSCD on the shared region) · SSCD ResNeXt-101 in the Somepalli et al. replication setting (`sscd_disc_large`, resize 256 / centre-crop 224) · DINOv2 ViT-S/14 · DINOv2 shared parts (share of patches with a mutual best match; AUC 0.997 on the benchmark) · CLIP ViT-B/32 · DreamSim (OpenCLIP ViT-B/32 tuned on human similarity judgments) · LPIPS (AlexNet) |
| **Objects** | D-FINE object detector (365 Objects365 categories): objects are found in both images, paired up and compared one by one, with marked-up images in the Objects tab · overlap of the kinds of objects found · body pose similarity (ViTPose-B, 17 joints per person, Procrustes + OKS, mirror-aware) |
| **Exact & verbatim** | SHA-256 · SHA-1 · MD5 · SHA-512 · SHA3-256 · BLAKE2b · CRC-32 · same image data with metadata set aside (hash of the JPEG scans / PNG image chunks / WebP bitstream) · pixel-exact match of decoded pixels · verbatim crop search (template matching, verified pixel-for-pixel) |
| **Fuzzy file hashes** | ssdeep (context-triggered piecewise hashing) · TLSH (Trend Micro) · Nilsimsa · LZJD (Lempel-Ziv Jaccard distance) — byte-level similarity of the files, as used in digital forensics and malware analysis |
| **Perceptual hashes** | PDQ (Meta) · PDQ over all 8 rotations/mirrors · pHash · pHash (simple) · dHash · dHash (vertical) · aHash · wHash (Haar) · wHash (Daubechies) · colour hash · crop-resistant hash (segment-wise) · Blockhash · OpenCV img_hash block-mean (two modes), Marr–Hildreth, radial-variance and colour-moment hashes |
| **Keypoints & geometry** | ORB, AKAZE, BRISK and KAZE keypoints with RANSAC homography · XFeat learned keypoints (CVPR 2024) with mutual-nearest-neighbour matching · retry against a mirrored B · Fourier–Mellin (log-polar phase correlation: rotation and scale) · phase correlation (shift) · multi-scale template search · SSIM after aligning B onto A |
| **Pixel & structural** | SSIM · MS-SSIM · PSNR · normalised cross-correlation · UQI · GMSD · multi-scale GMSD · FSIM · VSI · HaarPSI · VIF · DSS · MDSI · normalised mutual information · CIEDE2000 ΔE · changed-pixel ratio · Carlini et al.’s tiled ℓ2 extraction test (512², worst of 16 tiles, ≤ 0.15) · edge agreement (Pratt’s figure of merit) · edge distance (modified Hausdorff) · normalised compression distance |
| **Colour & histograms** | RGB-histogram correlation · χ² · intersection · Bhattacharyya · Kullback–Leibler · Jensen–Shannon · Kolmogorov–Smirnov (brightness) · brightness EMD · colour moments · colour coherence vectors · colour correlogram · dominant-palette EMD · average colour (CIEDE2000) |
| **Texture & shape** | GIST scene descriptor · histogram of oriented gradients · Gabor texture energy · Haralick (GLCM) texture · local binary patterns · Zernike moments · Hu moments |
| **Metadata & provenance** | Content Credentials (C2PA: signer, tool, actions, generative-AI declarations, and whether B lists A as an ingredient — read with the official C2PA SDK, loaded only when a file carries a manifest) · EXIF capture fields (camera, timestamp, unique IDs) · GPS location · copyright management information (XMP/IPTC/EXIF creator, rights, credit, licence — flagged when B drops A’s) · XMP edit history (DocumentID, DerivedFrom, DocumentAncestors; AI-generator settings in PNG text) · JPEG encoder fingerprint (quantization tables, estimated quality, subsampling) · embedded EXIF preview vs both images · ICC colour profile · dimensions & aspect ratio · Stable Diffusion’s invisible watermark (the invisible-watermark “dwtDct” mark written by the SD 1.x/2.x reference scripts and the SDXL pipeline; decoder verified bit-for-bit against the reference library) |

Each test has a verdict (*Identical*, *Match*, *Partial*, *No match*, *N/A*). Thresholds come from the source
papers where they exist — SSCD ≥ 0.75 (90% precision on DISC2021, per the SSCD authors), SSCD > 0.5 for
replication (Somepalli et al., CVPR 2023 / NeurIPS 2023), PDQ ≤ 31 bits (Meta). Where the literature gives no
threshold, the *match* line is set on the benchmark below so that 1% of unrelated pairs pass, and the *partial*
line so that 5% pass; each test’s explanation says which kind of threshold it uses. All thresholds live in
[`js/engines.js`](js/engines.js).

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
  - *Copied or similar?* — red where the copy detector’s evidence and the look-alike model agree, amber where
    the look-alike model finds a counterpart the copy detector does not count (shared subject, pose or idea).
  - *Pose* — ViTPose skeletons for the people in both images, with B’s pose laid over A’s after removing
    position, size, rotation and (if it fits better) mirroring; joints coloured by agreement.
  - *Cover-up test* — paint over parts of either image and SSCD re-scores the covered pair, next to the
    first-order prediction from the pairwise split (filtration by hand; cf. the deletion test of RISE).
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

## Copyright cases

Twelve cases from the course, each with the plaintiff’s work (A) and the defendant’s (B) and the court’s
holding — infringement, fair use, or no infringement, with mixed or unfinished results (Cariou v. Prince,
Sedlik v. Kat Von D) marked as such:

- **Infringement:** Rogers v. Koons · Steinberg v. Columbia Pictures · Friedman v. Guetta · Brammer v. Violent
  Hues · Dr. Seuss v. ComicMix · Andy Warhol Foundation v. Goldsmith
- **Fair use:** Leibovitz v. Paramount · Blanch v. Koons · Cariou v. Prince (25 of 30 works; the one shown was
  remanded, then settled)
- **No infringement:** Harney v. Sony · Rentmeester v. Nike · Sedlik v. Kat Von D (jury verdict; rehearing en
  banc pending)

Pick a case from the drop-down to scan its pair, or press **Run all cases** to scan every pair in turn and fill a
table of the scanner’s verdict, similarity level, tests flagged, SSCD, DINOv2 shared parts and DreamSim beside
the court’s holding. The table makes the site’s point concrete. Koons had Rogers’s photograph re-made as a
sculpture, so the copy detectors see only a similar subject, yet the court found infringement; Prince built his
paintings from Cariou’s actual photographs, which the detectors flag, yet most of the series was held fair use.
Computer similarity is at most evidence of copying; liability also turns on what was protectable, how much was
taken and whether the use was fair. Images for the Warhol, Seuss,
Rentmeester, Brammer and Blanch cases are taken from the published opinions; to add cases, see
[`assets/cases/README.md`](assets/cases/README.md).

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
- [`tools/export_xfeat.py`](tools/export_xfeat.py) — exports the XFeat backbone (Apache-2.0) with dynamic image
  sizes (its `unfold` replaced by an equivalent reshape). The keypoint heat map, NMS, reliability scoring and
  bicubic descriptor sampling are ported to [`js/lib/xfeat.js`](js/lib/xfeat.js) and reproduce the reference
  `detectAndCompute` exactly (same 1,024 keypoints, descriptor cosine 1.00000).
- [`tools/export_dreamsim.py`](tools/export_dreamsim.py) — merges DreamSim's LoRA weights into the OpenCLIP
  ViT-B/32 backbone (MIT) and exports the normalised embedding (cosine to PyTorch 0.9997 after int8 compression).
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
  JavaScript match Python ONNX Runtime within 1e-6;
- SHA-512, SHA3-256, BLAKE2b and CRC-32 equal Python’s `hashlib` / `zlib`; ssdeep equals `ppdeep`, TLSH
  `py-tlsh`, Nilsimsa the `nilsimsa` package, and LZJD `pyLZJD` digest for digest (including pyLZJD’s own
  approximate selection of the smallest hashes);
- the extra `imagehash` hashes (simple pHash, vertical dHash, Daubechies wHash, colour hash and the
  crop-resistant hash, with Pillow’s Gaussian blur and median filter ported for it) are bit-identical on all
  380 benchmark images;
- OpenCV’s img_hash: block-mean hashes are bit-identical; the Marr–Hildreth and radial-variance logic is exact
  when fed OpenCV’s own intermediate images, and colour-moment Hu moments agree to 2e-10. Run end to end in the
  browser the results differ slightly from native OpenCV (85% / 97% of images identical), because OpenCV’s
  WebAssembly build rounds a few pixels of its resizing and filtering differently from the native build;
- colour: Kullback–Leibler matches OpenCV’s `compareHist`, Jensen–Shannon and Kolmogorov–Smirnov match SciPy,
  the palette Earth Mover’s distance matches SciPy’s HiGHS solver to 1e-14, and colour moments, coherence vectors
  and correlograms match NumPy implementations of the papers;
- texture: LBP, HOG and GLCM match scikit-image exactly, Gabor energies match scikit-image kernels with SciPy
  convolution (1e-13), GIST matches a NumPy port of LMgist.m (1e-14), Zernike moments match mahotas (1e-15) and
  Hu-moment matching matches OpenCV; the FFT used throughout matches `numpy.fft` (1e-15);
- VIF, HaarPSI, MDSI, multi-scale GMSD, FSIM and VSI match `piq` in double precision (≤ 4e-12; DSS ≤ 3e-7, from
  PyTorch’s summation order), and normalised mutual information matches scikit-image;
- phase correlation and the Hanning window match OpenCV (1e-12); Canny edges are identical to native OpenCV,
  Pratt’s figure of merit matches a SciPy distance transform and the modified Hausdorff distance scikit-image;
  compressed sizes for NCD are identical to zlib; embedded ICC profiles and their descriptions match Pillow.

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
| SSCD after alignment with XFeat + ORB | — | 99.7% | collage 89%, corner crop 100%, memes 100%, perspective 95%, rotation 15° 95%, rotation 90° 89% |
| SSCD Somepalli setting | 0.998 | 98.3% | |
| DINOv2 | 0.998 | 97.2% | |
| DreamSim | 0.993 | 90.0% | (not a copy detector: it tracks how alike people find the images) |
| CLIP | 0.983 | 83.7% | |
| pHash | 0.827 | 52.6% | |

The classic (non-neural) tests on the same benchmark — AUC, and the share of edited copies each finds at its
*match* line (which lets through at most 1% of the unrelated pairs):

<details>
<summary>All classic tests</summary>

| Group | Test | AUC | Copies found |
| --- | --- | --- | --- |
| Hashes | dHash · pHash · aHash · wHash | 0.86 · 0.83 · 0.83 · 0.85 | 44% · 42% · 45% · 45% |
| | PDQ · PDQ + rotations/mirrors · Blockhash | 0.77 · 0.82 · 0.84 | 39% · 46% · 43% |
| | Marr–Hildreth · crop-resistant · vertical dHash · block mean (×2) | 0.88 · 0.87 · 0.87 · 0.84 | 58% · 49% · 59% · 53% |
| | radial variance · colour moments (Hu) · simple pHash · Daubechies wHash | 0.80 · 0.82 · 0.78 · 0.77 | 48% · 32% · 51% · 50% |
| Geometry | KAZE keypoints (standard rule) | 0.88 | 71% (no false alarms) |
| | Fourier–Mellin · phase correlation · multi-scale template | 0.94 · 0.86 · 0.86 | 80% · 66% · 60% |
| Structure | NCD · VSI · HaarPSI · FSIM | 0.95 · 0.90 · 0.90 · 0.87 | 63% · 58% · 47% · 48% |
| | UQI · ΔE · NCC · MS-SSIM · SSIM · PSNR · GMSD | 0.89 · 0.89 · 0.84 · 0.83 · 0.82 · 0.81 · 0.78 | 26% · 27% · 39% · 37% · 38% · 12% · 32% |
| | VIF · NMI · DSS · MS-GMSD · MDSI | 0.82 · 0.82 · 0.81 · 0.78 · 0.73 | 53% · 55% · 44% · 43% · 26% |
| | Pratt’s FOM · modified Hausdorff | 0.82 · 0.78 | 39% · 31% |
| Colour | coherence vectors · intersection · χ² · Jensen–Shannon · Bhattacharyya | 0.96 · 0.95 · 0.95 · 0.95 · 0.94 | 74% · 70% · 66% · 60% · 60% |
| | correlation · KL · KS · palette EMD · brightness EMD | 0.94 · 0.93 · 0.91 · 0.91 · 0.90 | 43% · 65% · 46% · 48% · 45% |
| | colour moments · correlogram · average colour | 0.89 · 0.88 · 0.88 | 40% · 43% · 44% |
| Texture & shape | GIST · Gabor · GLCM · Hu moments · HOG · Zernike · LBP | 0.94 · 0.88 · 0.88 · 0.86 · 0.84 · 0.81 · 0.72 | 56% · 56% · 38% · 33% · 48% · 47% · 30% |

The byte-level fuzzy hashes are not in the table: a re-encoded image shares almost no bytes with the original,
so they find only the copies whose files kept most of their bytes (ssdeep 29, Nilsimsa 36, LZJD 42 of 361),
while no unrelated pair ever reached their match lines.

</details>

Tried and left out: an affine ECC alignment score (it converged differently in the browser and in native
OpenCV, and found spurious alignments between unrelated images), the spectral angle mapper (any two
black-and-white images score as identical), and OpenCV’s hue–saturation histograms, which flagged a fifth to a
third of unrelated pairs for the same reason; the histogram tests now use 8×8×8 RGB histograms.

ICDiff’s PDF-Embedding (Wang et al., NeurIPS 2024), which predicts a 0–5 “replication level” for diffusion
outputs and correlates with human labels on its own D-Rep data far better than SSCD, was also tested and not
shipped: outside that distribution its six level scores are nearly tied, so it rated near-identical images as
level 2 and placed 14% of unrelated pairs at level 2 or above.

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
- XFeat: Potje et al., *XFeat: Accelerated Features for Lightweight Image Matching*, CVPR 2024 — Apache-2.0.
- ViTPose: Xu et al., *ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation*, NeurIPS 2022 —
  Apache-2.0 (ONNX export by onnx-community, compressed with `tools/quantize_weights.py`).
- DreamSim: Fu et al., *DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data*,
  NeurIPS 2023 — MIT (OpenCLIP backbone, MIT).
- DINOv2: Oquab et al., 2023 — Apache-2.0. CLIP: Radford et al., 2021 — MIT. LPIPS: Zhang et al., CVPR 2018 —
  BSD-2-Clause (AlexNet weights from torchvision, BSD-3-Clause).
  Licence texts are in [`models/licenses/`](models/licenses).
- [C2PA web SDK](https://github.com/contentauth/c2pa-js) 0.15.3 (MIT; bundled with `highgain`, ISC) — in
  `vendor/c2pa/`. C2PA test images for development came from the C2PA public test files (CC BY-SA 4.0).
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) 1.30 (MIT),
  [OpenCV.js](https://github.com/TechStark/opencv-js) 4.12 (Apache-2.0),
  [exifr](https://github.com/MikeKovarik/exifr) 7.1 (MIT) — in `vendor/` with their licences.
- Perceptual hash algorithms follow [imagehash](https://github.com/JohannesBuchner/imagehash) (BSD-2-Clause),
  [blockhash](https://github.com/commonsmachinery/blockhash-js) and
  [PDQ](https://github.com/facebook/ThreatExchange/tree/main/pdq); the block-mean, Marr–Hildreth,
  radial-variance and colour-moment hashes follow OpenCV’s `img_hash` module (Apache-2.0).
- Fuzzy hashes: ssdeep ported from [ppdeep](https://github.com/elceef/ppdeep) (Apache-2.0); TLSH is Trend
  Micro’s [JavaScript port](https://github.com/trendmicro/tlsh) (Apache-2.0 or BSD); LZJD ported from
  [pyLZJD](https://github.com/EdwardRaff/pyLZJD) (Apache-2.0); Nilsimsa from the
  [nilsimsa](https://pypi.org/project/nilsimsa/) package (MIT).
- Quality metrics follow [piq](https://github.com/photosynthesis-team/piq) (Apache-2.0); LBP, HOG, GLCM, Gabor
  kernels and NMI follow [scikit-image](https://scikit-image.org/) (BSD-3-Clause); Zernike moments follow
  [mahotas](https://github.com/luispedro/mahotas) (MIT); GIST follows Oliva & Torralba’s LMgist.m; Pillow’s
  Gaussian blur and median filter are ported for the crop-resistant hash (Pillow licence).
- Example photos (via scikit-image): Eileen Collins (NASA, public domain), Falcon 9 launch (SpaceX, public
  domain), Chelsea the cat (Stefan van der Walt, CC0), coffee (Rachel Michetti, CC0).
