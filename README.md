
# The Aging Challenge: Face Recognition Benchmark Under Cross-Age Variation

Companion code for the paper:

> **"The Aging Challenge: A Performance Evaluation on Baseline Face Recognition Models Under Cross Age Variation"**  
> Shreiya Ramaswamy Muthuvelan, Maneesha — BITS Pilani Dubai Campus

Benchmarks four baseline face recognition models — **ArcFace, FaceNet, VGGFace, OpenFace** — on age-variant datasets (**CACD** and **FG-NET**) using closed-set identification and subject-disjoint verification protocols.

---
## Paper

The research paper associated with this repository is currently not publicly available.  
The link will be added here once the paper becomes available.

[Read the Paper](paper/paper.pdf)

## Repository Structure

```
face-aging-benchmark/
│
├── src/                                        
│   ├── __init__.py
│   ├── dataset.py                              
│   ├── dataset_processing.py                   
│   ├── data_loading.py                         
│   ├── identification.py                       
│   ├── age_group_pairing_subject_disjoint.py   
│   └── utils.py                                
│
├── scripts/
│   ├── __init__.py
│   ├── extract_embeddings.py                   
│   └── run_embeddings.py                       
│
├── evaluation/
│   ├── __init__.py
│   ├── identification_closed_set_protocol.py  
│   └── verification_subject_disjoint_protocol.py 
│
├── visualizations/
│   ├── identification_visuals.ipynb            
│   └── verification_results.ipynb              
│
├── data/                                      
│   ├── raw/          ← downloaded by src/dataset.py
│   ├── processed/    ← created by src/dataset_processing.py
│   └── embeddings/   ← created by scripts/run_embeddings.py
│
├── results/                                    # CSV outputs — created at runtime
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shreiya-Muthuvelan/cross-age-face-recognition-benchmark.git
cd face-aging-benchmark
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle credentials

Datasets are downloaded using the Kaggle API.  
Place your `kaggle.json` file in `~/.kaggle/` (Linux/macOS) or `%USERPROFILE%\.kaggle\` (Windows)

---

## Execution Flow

All commands are run from the **project root** directory.

### Step 1 — Download datasets

```bash
python src/dataset.py
```

Downloads **FG-NET** and **CACD** from Kaggle and places raw images under:

```
data/raw/fgnet/     ← flat folder of .jpg images
data/raw/cacd/      ← one sub-folder per celebrity subject
```

### Step 2 — Process datasets

```bash
python src/dataset_processing.py
```

Organises images into per-subject folders and writes `metadata.json` for each subject:

```
data/processed/FGNET/<subject_id>/
    001A05.jpg
    001A10.jpg
    metadata.json          ← {"001A05.jpg": {"age": 5}, ...}

data/processed/CACD/<celebrity_name>/
    32_Aaron_Eckhart_0001.jpg
    metadata.json
```

### Step 3 — Build metadata DataFrame (optional standalone check)

```bash
python src/data_loading.py
```

Prints a summary of all images found across both datasets. This module is also
imported internally by the evaluation scripts.

### Step 4 — Extract embeddings

```bash
python scripts/run_embeddings.py
```

Extracts face embeddings for every image using all four models and saves them as
individual `.npy` files:

```
data/embeddings/<model>/<dataset>/
    001A05.npy            ← per-image embedding vector
    001A10.npy
    fgnet_embeddings.npy  ← combined batch file (for resuming)
    fgnet_metadata.json
```

Extraction **resumes automatically** if interrupted — already-processed images
are skipped.

### Step 5a — Identification evaluation

```bash
# Single model + dataset
python evaluation/identification_closed_set_protocol.py --dataset fgnet --model vggface

# Run all combinations
for dataset in fgnet cacd; do
  for model in facenet arcface vggface openface; do
    python evaluation/identification_closed_set_protocol.py \
      --dataset $dataset --model $model
  done
done
```

Results are saved to `results/results_identification_<dataset>_<model>.csv`.

### Step 5b — Verification evaluation

```bash
python evaluation/verification_subject_disjoint_protocol.py --dataset fgnet --model vggface
```

Results are saved to `results/results_agebins_<dataset>_<model>.csv`.

> The verification script reads from `metadata.csv` (a flat CSV with columns
> `dataset`, `model`, `subject_id`, `age`, `embedding_path`).  
> Generate it by aggregating the per-model `*_metadata.json` files produced in
> Step 4, or adapt `src/data_loading.py` to write this CSV directly.

---

## Evaluation Protocols

| Task | Protocol | Metric |
|---|---|---|
| Identification | Closed-set (gallery = probe set, self-match masked) | Rank-1 / Rank-5 / Rank-10 accuracy |
| Verification | Subject-disjoint, stratified by age bin | AUC, EER |

Age bins: `0–11`, `12–17`, `18–29`, `30–44`, `45–59`, `60+`

---

## Key Results (from paper)

### Identification — FG-NET

| Model | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---|
| VGGFace | 0.000 | 0.433 | 0.453 |
| FaceNet | 0.000 | 0.354 | 0.385 |
| ArcFace | 0.002 | 0.149 | 0.194 |
| OpenFace | 0.000 | 0.060 | 0.096 |

### Verification — average AUC

| Model | FG-NET | CACD |
|---|---|---|
| VGGFace | 0.813 | 0.523 |
| FaceNet | 0.787 | 0.519 |
| ArcFace | 0.664 | 0.505 |
| OpenFace | 0.590 | 0.503 |

---

## Datasets

| Dataset | Images | Subjects |
|---|---|---|
| [FG-NET](https://www.kaggle.com/datasets/aiolapo/fgnet-dataset) | 1002 | 82 |
| [CACD (filtered)](https://www.kaggle.com/datasets/pdombrza/cacd-filtered-dataset) | 152 948 | 2000 | 

---

## Citation

If you use this code or find this work useful in your research, please cite:

```bibtex
@inproceedings{muthuvelan2026aging,
  title  = {The Aging Challenge: A Performance Evaluation on Baseline Face Recognition Models Under Cross Age Variation},
  author = {Muthuvelan, Shreiya Ramaswamy and Maneesha},
  year   = {2026},
  note   = {Paper link coming soon}
}
