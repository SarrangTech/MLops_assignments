# Lab 2 — Docker: Containerised Model Training

MLOps (IE 7374) — Lab Assignment 2
Base lab: `Labs/Docker_Labs/Lab1` in the course repo
(https://github.com/SarrangTech/MLOps/tree/main/Labs/Docker_Labs/Lab1)

The original lab containerises a tiny model-training script: a
`Dockerfile` on `python:3.10` copies everything in `src/` into the
image, installs the requirements, and runs a RandomForest training
script on the Iris dataset when the container starts. This submission
keeps that idea — *train a model inside a Docker container* — and
makes meaningful changes to the Docker work itself (the lab's core
concept), listed in “Changes I made” below.

---

## 1. What you need to install

This README assumes **nothing is installed** on your machine.

### For the Docker run (recommended — this is the lab)

| Tool | Version used | How to get it |
|---|---|---|
| Docker Engine / Docker Desktop | Docker 24 or newer (built and described against the `python:3.10-slim` base image) | Windows / macOS: install **Docker Desktop** from https://www.docker.com/products/docker-desktop and start it. Linux: install **Docker Engine** from https://docs.docker.com/engine/install/ |
| Git | any recent version | https://git-scm.com/downloads — only needed to clone the repo; you can also download the repo as a ZIP from GitHub |

Check Docker is working before you continue:

```bash
docker --version
docker run hello-world
```

The second command should print a “Hello from Docker!” message.

**You do not need Python installed for the Docker run** — Python and
all libraries are installed inside the image by the `Dockerfile`.

### For the no-Docker fallback run (optional, Section 4)

| Tool | Version |
|---|---|
| Python | 3.10 – 3.12 |
| pip packages | exactly as pinned in `requirements.txt`: `scikit-learn==1.5.2`, `joblib==1.4.2`, `numpy==1.26.4` |

---

## 2. Get the code and set up

```bash
git clone https://github.com/SarrangTech/MLops_assignments.git
cd MLops_assignments/lab2-docker
```

No other setup, accounts, API keys, or environment files are needed.
The dataset (Breast Cancer Wisconsin) ships inside scikit-learn, so
nothing is downloaded at run time except the Docker base image and
the pip packages during the image build.

Project structure:

```text
lab2-docker/
├── Dockerfile            <- the modified container definition
├── docker-compose.yml    <- NEW: one-command Compose run
├── .dockerignore         <- NEW: keeps the build context clean
├── requirements.txt      <- pinned versions
├── README.md             <- this file
├── sample_output/        <- a verified run's metrics.json + run_log.txt
└── src/
    └── main.py           <- the training script that runs in the container
```

---

## 3. Run it with Docker (exact steps)

### Step 1 — Build the image

Run from inside the `lab2-docker` folder:

```bash
docker build -t lab2-docker .
```

The first build downloads the `python:3.10-slim` base image and the
pip packages, so it takes a minute or two. Later builds are much
faster because the dependency layer is cached (see “Changes I made”).

### Step 2 — Run the container

```bash
docker run --rm -v "$(pwd)/output:/app/output" lab2-docker
```

- `--rm` removes the container when it finishes.
- `-v "$(pwd)/output:/app/output"` mounts a folder called `output`
  in the current directory on your machine to `/app/output` inside
  the container, so the trained model and metrics are still there
  after the container exits. On Windows PowerShell, use
  `-v "${PWD}/output:/app/output"` instead.

### Step 3 (optional) — Change the run without rebuilding

The image uses an `ENTRYPOINT`, so training arguments pass straight
to the script:

```bash
docker run --rm -v "$(pwd)/output:/app/output" lab2-docker --n-estimators 200 --test-size 0.25
```

### Alternative — Docker Compose

If you have Docker Compose (included with Docker Desktop):

```bash
docker compose up --build
```

This builds the same image, runs the same training job, and writes
the outputs into `./output` on your machine.

---

## 4. Run it without Docker (fallback / sanity check)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/main.py --output-dir output
```

This is exactly the command the container runs, so the console
output and the files produced are the same.

---

## 5. What a successful run looks like

The container prints (values from a verified run with the default
arguments):

```text
Dataset: Breast Cancer Wisconsin - 569 samples, 30 features, 2 classes
Train/test split: 455 train, 114 test
Accuracy: 0.9561
F1 score: 0.9655
Model saved to: /app/output/breast_cancer_rf.joblib
Metrics saved to: /app/output/metrics.json
The model training was successful
```

(In the no-Docker run the two saved paths are `output/…` instead of
`/app/output/…` — everything else is identical. Accuracy/F1 are
deterministic for the default `--random-state 42`.)

After the run, your `output` folder contains:

| File | What it is |
|---|---|
| `breast_cancer_rf.joblib` | the trained RandomForest model |
| `metrics.json` | the run's accuracy, F1, and the parameters used |

`output/metrics.json` looks like this:

```json
{
  "dataset": "breast_cancer_wisconsin",
  "n_samples": 569,
  "n_features": 30,
  "n_estimators": 100,
  "test_size": 0.2,
  "random_state": 42,
  "accuracy": 0.9561,
  "f1_score": 0.9655
}
```

If you see the “The model training was successful” line and both
files are in `output/`, the lab ran correctly.

---

## 6. Changes I made

The point of this lab is **Docker / containerisation**, so the
meaningful changes are to the Docker work itself — not just to the
dataset or model.

### Core changes — the Docker concept

1. **Slim base image.** `FROM python:3.10` → `FROM python:3.10-slim`.
   The original used the full Python image; the slim image contains
   everything this job needs and is far smaller.
2. **Cache-friendly layer order.** The original copied all of `src/`
   into the image *before* installing dependencies, so every code
   change forced a full re-install. Here `requirements.txt` is
   copied and installed **first**, and the source is copied after —
   Docker re-uses the cached dependency layer when only code changes.
3. **Pinned, cache-free installs.** `requirements.txt` now pins exact
   versions (`scikit-learn==1.5.2`, `joblib==1.4.2`, `numpy==1.26.4`
   — the original left the copy in `src/` unpinned in the Dockerfile
   step) and pip runs with `--no-cache-dir` (`PIP_NO_CACHE_DIR=1`),
   so builds are reproducible and the image carries no pip cache.
4. **Non-root user.** The original ran as root. The image now creates
   `appuser`, gives it ownership of `/app`, and switches to it
   (`USER appuser`) before running anything.
5. **Output volume.** The original saved `iris_model.pkl` *inside*
   the container, where it disappeared with the container. Here
   `/app/output` is a declared `VOLUME`, the script writes the model
   and a `metrics.json` there, and the run command mounts it to the
   host — the artifacts survive the container.
6. **ENTRYPOINT + runtime arguments.** The original hard-coded
   `CMD ["python", "main.py"]`. Here `ENTRYPOINT ["python",
   "src/main.py"]` makes training the container's fixed job while
   still letting a checker pass arguments at run time
   (`docker run lab2-docker --n-estimators 200`) with no rebuild.
   `ENV PYTHONUNBUFFERED=1` also makes the training logs appear in
   real time in `docker logs` / the run output.
7. **`.dockerignore` added (new file).** Keeps `__pycache__`, virtual
   environments, the `output/` folder and git files out of the build
   context, so builds are faster and the image contains only what
   the job needs.
8. **`docker-compose.yml` added (new file).** Lab1 had no Compose
   file. The Compose service builds the same Dockerfile, sets
   `OUTPUT_DIR`, and mounts `./output` — a one-command
   (`docker compose up --build`) equivalent of the `docker run`
   command in Section 3.

### Supporting changes — the payload script

9. **Different dataset and evaluation.** The payload now trains on
   the Breast Cancer Wisconsin dataset (569 samples, 30 features,
   binary classification) instead of Iris, and — unlike the original,
   which only printed a success line — it evaluates the model and
   reports accuracy and F1, writing them to `metrics.json`.
10. **Configurable script.** `src/main.py` accepts `--n-estimators`,
    `--test-size`, `--random-state`, and `--output-dir` (or the
    `OUTPUT_DIR` environment variable), which is what makes the
    ENTRYPOINT behaviour in change 6 useful.

---

## 7. Troubleshooting

- **`docker: command not found`** — Docker Desktop / Engine is not
  installed or not running. Install it (Section 1) and, on Desktop,
  wait until it says it is running, then try again.
- **Build fails downloading the base image** — you need an internet
  connection for the first build only.
- **No `output` folder after the run** — make sure you ran the
  `docker run` command from inside the `lab2-docker` folder and kept
  the `-v` mount exactly as written in Section 3.
