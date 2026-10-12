# Real-Time Detailed Video Analysis of Fruit Flies

A computer-vision and machine-learning pipeline that processes top-down videos of a
male and a female fruit fly in a circular arena and, for every frame, estimates:

- **how many flies** are present (including two flies touching in one blob),
- the **sex** of each separated fly,
- each fly's **orientation** (heading, 0–360°),
- the **wing angles** of the male (left and right).



## Problem

Measuring courtship behaviour (who is where, which way each fly faces, how far the
male extends his wings) by hand is slow. The goal is to extract this per frame,
automatically, fast enough to keep up with the video.

## Method

All models start from the same detector (`src/detection.py`), the one used in the
original Stanford project:

1. A filled circle the size of the arena (radius `(rows − 1) // 2`, centred in the
   square frame) masks out everything outside the arena.
2. A fixed grey-level threshold of 115 keeps only the dark body **core** (head,
   thorax, abdomen). Legs and translucent wings are lighter and drop out, and so do
   the dark arena rim (about 180) and reflections of flies in the wall.
3. External contours with area ≥ 500 px that do not touch the image border are the
   candidate flies.

Training and the video pipeline call the same functions, so their features are
computed identically. The detector was chosen by measurement; see
[Detector choice](#detector-choice).

| Task | Features (from the frame only) | Model |
|---|---|---|
| Fly count | core contour area | `DecisionTreeClassifier(max_depth=2)` predicting 1 / 2 flies per contour; the frame count is the sum over contours. The learned split is 10,442 px, the same as in the original project |
| Sex | core contour area + bounding-box aspect ratio | `StandardScaler` + `LogisticRegression` |
| Orientation | body axis from the image moments of the core; 240 px patch rotated so the axis is vertical; HOG | `PCA(20)` + `LogisticRegression` decides whether the head is at the other end (+180°) |
| Wing angle (male) | 288 px grey patch (wings included) rotated upright, split into right and mirrored left halves; HOG | `PCA(80)` + `LinearRegression` (Ridge also reported), one regressor for both wings |

The orientation and wing models follow the Stanford project's approach: rotate the
fly upright, compute HOG, reduce with PCA, then fit a linear model.

## Data

The full annotated dataset used for training and by `evaluate.py` comes from the
Stanford CS229 project and is **not included** in this repository. Download it here
and place it in `./input` (see [Where to put the data](#where-to-put-the-data)):

https://www.dropbox.com/scl/fo/uz5xtijd3faf1a30c0c8g/AJmDGy_7aXzYYHXqCv_YYCk?rlkey=d2gjrsfjstajwmoeja44jl9x0&e=1&dl=0

Two of the test videos are included in `sample_videos/` (`test5.mp4`, 2.7 MB, and
`test3.mp4`, 3.5 MB; identical copies of the dataset's videos). They are enough to
run the demo with the trained models in `models/`, but not to train or evaluate.

The full dataset contains:

- **326 annotated images** in **5 recording sessions** (one folder per session):

  | Session folder | Images | Content |
  |---|---|---|
  | `12-04_17-54-43` | 100 | sampled uniformly from the original video |
  | `12-05_12-43-00` | 33 | the two flies together |
  | `12-07_16-45-00` | 50 | flies running along the wall |
  | `12-08_11-15-00` | 102 | wing-angle frames from `test4.mp4` |
  | `12-08_22-00-00` | 41 | wing-angle frames from `test1.mp4` |

- Point annotations (labelme JSON) for each image:

  | Label | Meaning | Images |
  |---|---|---|
  | `mp`, `fp` | male / female body point | 326 |
  | `mh`, `ma` | male head / abdomen | 277 |
  | `fh`, `fa` | female head / abdomen | 217 |
  | `mp2` | second male body point (thorax) | 258 |
  | `mw` | male wing tips (two per image) | 257 |

- Five test videos, `test1.mp4` … `test5.mp4` (1530 × 1530, 35 FPS).

## Setup

Python 3.11. With conda:

```bash
conda create -n cs229-project python=3.11
conda activate cs229-project
pip install -r requirements.txt
```

Or with a plain virtual environment (Python 3.11):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Where to put the data

Place the dataset in `./input` at the repository root. A symlink to an existing
copy also works.

```text
input/
├── images/
│   ├── 12-04_17-54-43/   1.bmp, 1.json, 2.bmp, 2.json, ...
│   ├── 12-05_12-43-00/   *.png + *.json
│   ├── 12-07_16-45-00/
│   ├── 12-08_11-15-00/
│   └── 12-08_22-00-00/
└── videos/
    ├── test1.mp4
    └── ...
```

`input/` is git-ignored. Never commit it.

## Run the demo

The trained models and a sample video are in the repository, so the demo runs
straight from a fresh clone, without the dataset:

```bash
python main.py --video sample_videos/test5.mp4
```

This opens a preview window (press Q to stop) and writes the annotated video to
`results/videos/final_demo_test5.mp4`. On a machine without a display, add
`--no-display`; for a quick check, add `--max-frames 50`. Use
`--video sample_videos/test3.mp4` for the second sample.

## How to run

The steps below need the full dataset in `./input` (see [Data](#data)).

Run every command from the repository root, in this order:

```bash
# 1. Train the four models (each writes only its own file in models/)
python train/train_fly_count.py     # also writes results/metrics/fly_count_tree.txt
python train/train_sex.py
python train/train_orientation.py
python train/train_wing_angle.py

# 2. Cross-validated evaluation -> results/metrics/evaluation.{csv,md}, results/plots/*_pca.png
python evaluate.py

# 3. Frame-level fly count on all test videos -> results/metrics/video_counts.md
#    (per-frame CSV and example failure frames in results/debug/, git-ignored)
python eval_video_counts.py

# 4. Video pipeline -> results/videos/final_demo_<video>.mp4,
#    results/metrics/pipeline_stats/<video>.json
python main.py                                  # test1.mp4 with a preview window (Q to stop)
python main.py --video input/videos/test5.mp4   # any other video
python main.py --max-frames 50 --no-display     # quick headless smoke test
python main.py --h264                           # also write an H.264 copy for PowerPoint (needs ffmpeg)

# 5. Regenerate results/metrics/model_results.txt from steps 2 and 4
python make_report.py
```

Each trainer prints a grouped hold-out check, then saves a model fit on all its
data. `src/model_io.py` makes sure each trainer can only write its own model file.

## Results

`evaluate.py` scores every model in three ways, each reported as mean ± std:

- **Random split:** the original style, a random 80/20 split over rows, repeated
  with 10 seeds.
- **GroupKFold by image:** 5 folds; all contours and both wings of one image stay
  in the same fold.
- **Leave-one-session-out:** a whole recording folder is held out. This is the
  strictest test, because frames within a session are near-duplicates.

Three stages are compared:

- **Original:** the numbers hand-written in the first version of this repository.
- **Fixed models, Otsu detector:** the honest models and evaluation (items A–G),
  still using the first detector (Otsu threshold over the whole frame).
- **Final:** the same models on the arena-mask + core-threshold detector (items J–K).

| Metric | Original (one split) | Fixed models, Otsu detector: leave-one-session-out | Final: GroupKFold by image | Final: leave-one-session-out |
|---|---|---|---|---|
| Video frames with count = 2 (all 5 videos) | not measured | 71.3% | — | **100%** (2703 / 2703) |
| Fly count, per contour | 100% (circular labels) | 99.6 ± 0.8% | 100.0 ± 0.0% | 100.0 ± 0.0% |
| Fly count, per image (= 2) | — | 74.0 ± 37.6% | 100.0 ± 0.0% | 100.0 ± 0.0% |
| Sex | 88.46% (one 52-sample split) | 83.7 ± 17.1% | 98.4 ± 0.7% | 98.2 ± 3.1% |
| Orientation, flip error | 0% (keypoint features, unusable on video) | 1.7 ± 2.6% | 0.5 ± 0.9% | 0.9 ± 1.2% |
| Orientation, mean angular error | — | 13.4 ± 10.2° | 2.2 ± 1.4° | 3.3 ± 2.5° |
| Male / female axis error (perfect flip) | — | 17.2° / 3.6° | — | 2.6° / 1.1° |
| Wing angle MAE, true orientation | 4.24° (keypoint features, random split) | 7.5 ± 1.0° | 5.8 ± 0.5° | 7.8 ± 2.6° |
| Wing angle MAE, predicted orientation | — | 13.8 ± 4.3° | 6.0 ± 0.6° | 8.6 ± 2.2° |
| Pipeline speed (test1.mp4, Apple M5) | 45.17 FPS | 193.5 FPS processing | 176.4 FPS processing, 37.2 FPS wall clock with video writing and preview | |

Baselines, final detector, leave-one-session-out:

- **Orientation:** always predicting the training set's majority flip gives a 63.3%
  flip error and a 113.5° mean angular error. That is worse than chance, because the
  majority direction differs between sessions.
- **Wing angle:** predicting the training mean gives a 24.7° MAE.

The only number that got worse with the new detector is the wing MAE with *true*
orientation (7.5° → 7.8°). That is within the spread across sessions and is measured
on more males (236 instead of 137). The end-to-end wing error, which is what the
pipeline produces, improved from 13.8° to 8.6°.

Processing FPS fell from 193 to 176 even though detection got faster (1.7 → 0.7 ms
per frame). The per-fly models and labels now run on two flies in almost every
frame. Most of the time per frame goes to decoding the 1530 × 1530 frame and drawing.

The full tables are in:

- `results/metrics/evaluation.md`
- `results/metrics/video_counts.md`
- `results/metrics/model_results.txt` (includes the original numbers in a
  "before fixes" section)
- `results/metrics/before_detector_fix/` (all metrics with the Otsu detector)

The PCA-size curves are in `results/plots/orientation_pca.png` and
`results/plots/wing_pca.png`. The audit that motivated the fixes is in
`results/metrics/audit.md`.

### Detector choice

The first detector (Gaussian blur + Otsu over the whole frame) marked the dark area
outside the arena floor as one huge blob touching the image border. Flies that
touched the floor's edge merged with it and were discarded, so the video fly count
was wrong in 28.7% of frames (test3: 82.5%).

The comparison below is recorded in `results/metrics/detector_experiments.md`. The
original project's settings were tried one switch at a time; for each, the count
model was retrained and the videos and the leave-one-session-out evaluation were
re-run. These results were produced by earlier experiment scripts that are not part
of the final repository (they are available in the git history).

| Candidate | Video count = 2 | Sex (LOSO) | Orientation error (LOSO) |
|---|---|---|---|
| Otsu (baseline) | 71.3% | 83.7% | 13.4° |
| + arena mask | 71.3% | 86.4% | 16.4° |
| **mask + core threshold 115 (kept)** | **100%** | **98.2%** | **3.3°** |
| mask + "wings" threshold (arena mean − 5) | 41.5% | 50.9% | 24.2° |
| mask + core, no border filter | 100% (identical, not kept) | 98.2% | 3.3° |
| mask + Otsu, no border filter | 60.9% | 67.6% | 31.7° |

No threshold was tuned on the videos.

### Why the "original" numbers were wrong

- **Fly count:** each video frame was labelled `min(#contours, 2)`, and the
  features were the sizes of those same contours. The 100% was guaranteed by
  construction, and the training script was a copy of the sex trainer that
  overwrote `models/sex_model.pkl`.
- **Orientation and wing angle:**
  - The models were trained on distances and angles between the annotated
    keypoints, from which the label is derived.
  - At inference, the pipeline fed them placeholder constants.
  - The random splits put near-duplicate rows (for example, the two wings of one
    image) into both train and test.

## Limitations

- **Small data:** 326 annotated images from only 5 recording sessions. The
  leave-one-session-out standard deviations are large because each session is
  different (for example, sex accuracy is 92% on the wall-running session and
  99–100% on the others).
- **No truly held-out video:**
  - Session `12-08_22-00-00` was cut from `test1.mp4` and `12-08_11-15-00` from
    `test4.mp4`, so those two demo videos overlap the training data.
  - test2, test3 and test5 are probably held out, but session `12-04_17-54-43` was
    "sampled uniformly from video" without saying which one.
  - The 100% video count is therefore partly on training footage.
- **The video count only checks the number:** "count = 2" is correct in every frame,
  but the videos have no per-frame labels for sex, orientation or wings.
- **Touching flies:** when the two flies touch they form one fused contour
  (area above about 10,442 px). It is counted as 2 but not classified: no sex,
  orientation or wing angle is predicted for it. This affects 0–14.5% of frames
  per video (`results/metrics/video_counts.md`):

  | Video | test1 | test2 | test3 | test4 | test5 |
  |---|---|---|---|---|---|
  | Touching-flies frames | 35 / 351 (10.0%) | 51 / 352 (14.5%) | 0 / 736 (0%) | 18 / 632 (2.8%) | 21 / 632 (3.3%) |

  The reported sex, orientation and wing accuracy therefore describes isolated
  flies only.
- **Sex errors remain:** in about 1% of test3's frames both flies are labelled
  female.
- **Orientation labels:** head/abdomen labels exist for males (`mh`/`ma`, 277
  images) and females (`fh`/`fa`, 217 images). Only isolated flies can be used:
  256 males and 178 females.
- **Wing angle:**
  - 236 isolated males have both wing tips annotated, and the model runs only on
    flies predicted male.
  - The target is the angle between the head→abdomen axis and the line from the
    body point `mp` to the wing tip.
- **Fixed threshold:** the core threshold of 115 assumes this camera and lighting.
  A different setup would need it re-checked.
- **Hyperparameters:**
  - The PCA sizes (20 for orientation, 80 for wing) were chosen with GroupKFold by
    image, not with nested cross-validation.
  - The leave-one-session-out curves show that the choice is not critical for
    orientation.
- **Speed:** FPS depends on the machine. The wall-clock number includes writing a
  1530 × 1530 video and the preview window.

## Project structure

```text
fruit-fly-video-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                       entry point (--video, --max-frames, --no-display, --h264)
├── evaluate.py                   random / GroupKFold / leave-one-session-out evaluation
├── eval_video_counts.py          frame-level fly count on the test videos
├── make_report.py                writes results/metrics/model_results.txt
├── sample_videos/                test5.mp4, test3.mp4 for the demo (see Data)
├── src/
│   ├── detection.py              arena mask + thresholding + contours + moment axis
│   ├── features.py               shared count / upright-patch / HOG / wing-half features
│   ├── annotations.py            loading of labelme JSON + matching flies to contours
│   ├── model_io.py               save guard: each trainer may only write its own model
│   ├── pipeline.py               video pipeline
│   ├── analyze_annotations.py    legacy (see below)
│   ├── build_dataset.py          legacy
│   ├── check_dataset.py          legacy
│   ├── feature_engineering.py    legacy
│   ├── fly_count.py              legacy (deprecated, circular labels)
│   ├── inspect_annotation.py     legacy
│   ├── inspect_shapes.py         legacy
│   ├── orientation.py            legacy
│   ├── preprocessing.py          legacy
│   ├── sex_classifier.py         legacy
│   ├── video_test.py             legacy
│   ├── visualize_annotations.py  legacy
│   └── wing_angle.py             legacy
├── train/
│   ├── train_fly_count.py
│   ├── train_sex.py
│   ├── train_orientation.py
│   └── train_wing_angle.py
├── models/
│   ├── fly_count_model.pkl
│   ├── sex_model.pkl
│   ├── orientation_model.pkl
│   └── wing_angle_model.pkl
└── results/
    ├── metrics/
    │   ├── evaluation.csv, evaluation.md      evaluate.py
    │   ├── video_counts.md                    eval_video_counts.py
    │   ├── fly_count_tree.txt                 train/train_fly_count.py
    │   ├── model_results.txt                  make_report.py
    │   ├── pipeline_stats/                    main.py: test1.json, test3.json, test4.json, test5.json
    │   ├── audit.md                           initial audit (earlier script, see git history)
    │   ├── detector_experiments.csv           detector comparison (earlier script)
    │   ├── detector_experiments.md            same, as a table
    │   ├── orientation_by_sex.md              per-sex orientation (earlier script)
    │   └── before_detector_fix/               all metrics with the old Otsu detector:
    │                                          evaluation.csv, evaluation.md, fly_count_tree.txt,
    │                                          model_results.txt, video_counts.md
    ├── plots/
    │   ├── orientation_pca.png
    │   ├── wing_pca.png
    │   └── model_accuracy.png                 original (invalid) accuracies, kept for comparison
    ├── debug/                                 per-frame CSV + failure frames (git-ignored)
    └── videos/                                final_demo_<video>.mp4 (git-ignored)
```

### Legacy files

These files in `src/` are from the original implementation. The current pipeline,
trainers and evaluation scripts do not import or run them; they are kept for
reference only:

`analyze_annotations.py`, `build_dataset.py`, `check_dataset.py`,
`feature_engineering.py`, `fly_count.py`, `inspect_annotation.py`,
`inspect_shapes.py`, `orientation.py`, `preprocessing.py`, `sex_classifier.py`,
`video_test.py`, `visualize_annotations.py`, `wing_angle.py`.

## Credits

The dataset, problem setup and the HOG + PCA approach come from the Stanford CS229
project by **Steven Herbst**:
[github.com/sgherbst/cs229-project](https://github.com/sgherbst/cs229-project),
released under the MIT license (© 2018 Steven Herbst). The original video was
provided to that project by Ryan York.
